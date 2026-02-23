use core::code_chunk::OverlapStrategy;
use core::code_chunk::{
    ChunkOptions, CodeChunk, ContextChunk, IndexChunk, IndexChunkBuildError, IndexChunkOptions,
    RefillOptions,
};
use intelligence::NodeKind;
use intelligence::TreeSitterFile;
use intelligence::TreeSitterFileError;
use intelligence::scope_resolution::EdgeKind;
use petgraph::visit::EdgeRef;
use std::fmt;
use std::ops::Range;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub enum ChunkError {
    Parse(TreeSitterFileError),
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkError::Parse(e) => write!(f, "failed to parse file: {e:?}"),
        }
    }
}

impl std::error::Error for ChunkError {}

/// 基于 `luna/src/intelligence` 的 scope graph 进行“智能切分”。
///
/// - 优先取“直接挂在 root scope 下”的 scope 作为语义块（通常对应函数/类/方法等）。
/// - 如果某个 scope 过大，退化为行滑窗切分。
/// - 如果找不到任何 top-level scope，则对全文件做滑窗。
pub fn chunk_source(
    path: &str,
    src: &[u8],
    lang_id: &str,
    opt: ChunkOptions,
) -> Result<Vec<CodeChunk>, ChunkError> {
    let ts = TreeSitterFile::try_build(src, lang_id).map_err(ChunkError::Parse)?;
    let graph = ts.scope_graph().map_err(ChunkError::Parse)?;

    let line_starts = compute_line_starts(src);
    let root = find_root_scope_idx(&graph);
    let mut chunks = Vec::new();

    if let Some(root_idx) = root {
        let mut top_scopes = top_level_scopes(&graph, root_idx);
        top_scopes.sort_by_key(|(_, r)| r.start.byte);

        for (_, range) in top_scopes {
            // 空范围直接跳过
            if range.end.byte <= range.start.byte {
                continue;
            }

            // 超长 scope -> scope 内滑窗
            if range.size() > opt.max_chunk_bytes {
                chunks.extend(sliding_window_by_lines(
                    path,
                    src,
                    &line_starts,
                    range.start.line,
                    range.end.line,
                    opt.max_chunk_lines,
                    opt.overlap_lines,
                ));
            } else {
                chunks.push(make_chunk(
                    path,
                    src,
                    range.start.line,
                    range.end.line,
                    range.start.byte,
                    range.end.byte,
                ));
            }
        }
    }

    if chunks.is_empty() {
        // 全局 fallback：按行滑窗覆盖全文件
        let total_lines = line_starts.len().saturating_sub(1);
        chunks = sliding_window_by_lines(
            path,
            src,
            &line_starts,
            0,
            total_lines,
            opt.fallback_max_lines,
            opt.overlap_lines,
        );
    }

    // 统一 alias
    for (i, c) in chunks.iter_mut().enumerate() {
        c.alias = i;
    }

    Ok(chunks)
}

/// 生成 IndexChunk（用于向量/关键词检索）。
///
/// 切分策略（modern hybrid）：
/// - 语义边界优先：优先按 top-level scope（函数/类/方法）生成候选块。
/// - 预算归一化：仅对超长 scope 在 scope 内按 token 预算切分（优先 newline，其次 BPE 边界，必要时 overlap）。
/// - 兜底：无 AST/解析失败时退化为全文件 token 切分；tokenizer 不可用再退化为按行切分。
///
/// 说明：
/// - `repo` 用于把 `repo\tpath\n` 前缀计入 token 预算（可以传空字符串）。
/// - `lang_id` 用于解析 top-level scope（例如 "Rust"）。
/// - 若 tokenizer 编码失败，将降级为按行切分（`fallback_lines`）。
pub fn index_chunks(
    repo: &str,
    path: &str,
    src: &[u8],
    lang_id: &str,
    tokenizer: &Tokenizer,
    opt: IndexChunkOptions,
) -> Vec<IndexChunk> {
    // modern hybrid：
    // 1) 优先按 top-level scope 取“语义边界”（函数/类/方法等）
    // 2) 在每个 scope 内再按 token 预算切分（超长才切），保证检索单元尺寸可控
    // 3) 解析失败 / 无 scope 时，退化为全文件 token 切分；tokenizer 不可用则按行切分

    let src = String::from_utf8_lossy(src);

    let encoding = match tokenizer.encode(src.as_ref(), true) {
        Ok(e) => e,
        Err(_) => return by_lines_index_chunks(path, &src, opt.fallback_lines),
    };

    let offsets_all = encoding.get_offsets();
    let ids_all = encoding.get_ids();

    // remove trailing SEP-like token with (0,0) offsets (对齐 Kuaima)
    let offsets_len = offsets_all.len().saturating_sub(1);
    let offsets: &[(usize, usize)] = if offsets_all.get(offsets_len).map(|o| o.0) == Some(0) {
        &offsets_all[..offsets_len]
    } else {
        offsets_all
    };

    // ids 用于 BPE 边界判断；长度与 offsets 不一致时，保守裁剪避免越界
    let ids_len = offsets.len().saturating_add(1).min(ids_all.len());
    let ids = &ids_all[..ids_len];

    let token_bounds = opt.min_chunk_tokens..opt.max_chunk_tokens;
    let min_tokens = token_bounds.start;

    // 计算 `repo\tpath\n` 前缀 token 占用（用于预算扣减）
    let prefix = format!("{}\t{}\n", repo, path);
    let prefix_tokens = match tokenizer.encode(prefix, true) {
        Ok(e) => e,
        Err(_) => return by_lines_index_chunks(path, &src, opt.fallback_lines),
    };
    let prefix_len = prefix_tokens.get_ids().len();
    if token_bounds.end <= DEDUCT_SPECIAL_TOKENS + prefix_len {
        return by_lines_index_chunks(path, &src, opt.fallback_lines);
    }
    let max_tokens = token_bounds.end - DEDUCT_SPECIAL_TOKENS - prefix_len;

    // 语义边界：top-level scopes
    let top_scopes = TreeSitterFile::try_build(src.as_bytes(), lang_id)
        .and_then(|ts| ts.scope_graph())
        .ok()
        .and_then(|graph| find_root_scope_idx(&graph).map(|root| top_level_scopes(&graph, root)))
        .unwrap_or_default();

    let mut out = Vec::new();

    if !top_scopes.is_empty() {
        // 以语义边界为主：每个 scope 内做 token 预算归一化（超长才切）
        for (_, r) in top_scopes {
            if r.end.byte <= r.start.byte {
                continue;
            }

            let Some(token_range) = token_range_for_byte_range(offsets, r.start.byte, r.end.byte)
            else {
                continue;
            };
            let token_len = token_range.end.saturating_sub(token_range.start);

            // scope 太小也保留：语义边界比 min_tokens 更重要（否则会漏掉小函数/小类）
            if token_len <= max_tokens {
                if r.end.byte > r.start.byte {
                    out.push(make_index_chunk_by_bytes(
                        path,
                        &src,
                        r.start.byte,
                        r.end.byte,
                    ));
                }
                continue;
            }

            // scope 过大：在 scope 内按 token 预算切分
            match by_tokens_in_token_range(
                path,
                &src,
                tokenizer,
                ids,
                offsets,
                min_tokens,
                max_tokens,
                opt.overlap,
                token_range,
            ) {
                Ok(mut chunks) => out.append(&mut chunks),
                Err(_) => {
                    // 极端情况下（例如 tokenizer/id 不匹配），退化为按行切分
                    out.extend(by_lines_index_chunks(path, &src, opt.fallback_lines));
                }
            }
        }
    }

    if out.is_empty() {
        // fallback：全文件 token 切分（对齐旧行为）
        let full = 0..offsets.len();
        match by_tokens_in_token_range(
            path,
            &src,
            tokenizer,
            ids,
            offsets,
            min_tokens,
            max_tokens,
            opt.overlap,
            full,
        ) {
            Ok(chunks) => out = chunks,
            Err(_) => return by_lines_index_chunks(path, &src, opt.fallback_lines),
        }
    }

    out
}

const DEDUCT_SPECIAL_TOKENS: usize = 2;

#[allow(clippy::too_many_arguments)]
fn by_tokens_in_token_range(
    path: &str,
    src: &str,
    tokenizer: &Tokenizer,
    ids: &[u32],
    offsets: &[(usize, usize)],
    min_tokens: usize,
    max_tokens: usize,
    strategy: OverlapStrategy,
    token_range: Range<usize>,
) -> Result<Vec<IndexChunk>, IndexChunkBuildError> {
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    if token_range.start >= token_range.end || token_range.end > offsets.len() {
        return Ok(Vec::new());
    }
    if max_tokens == 0 {
        return Err(IndexChunkBuildError::InvalidInput);
    }

    let max_newline_tokens = max_tokens * 3 / 4;
    let max_boundary_tokens = max_tokens * 7 / 8;

    let offsets_last = token_range.end.saturating_sub(1);
    let mut chunks = Vec::new();
    let mut start = token_range.start;
    let (mut last_line, mut last_byte) = (0usize, 0usize);

    loop {
        if start >= offsets_last {
            return Ok(chunks);
        }

        let next_limit = start.saturating_add(max_tokens).min(offsets_last);
        let end_limit = if next_limit >= offsets_last {
            offsets_last
        } else if let Some(next_newline) = (start + max_newline_tokens..next_limit)
            .rfind(|&i| src[offsets[i].0..offsets[i + 1].0].contains('\n'))
        {
            next_newline
        } else if let Some(next_boundary) = (start + max_boundary_tokens..next_limit).rfind(|&i| {
            // ids 与 offsets 不同源时可能越界：越界则认为是边界
            ids.get(i + 1)
                .and_then(|id| tokenizer.id_to_token(*id))
                .is_none_or(|s| !s.starts_with("##"))
        }) {
            next_boundary
        } else {
            next_limit
        };

        let token_len = end_limit.saturating_sub(start);
        // 末尾 chunk 允许小于 min_tokens：保证覆盖完整范围
        let allow_small = end_limit == offsets_last;
        if token_len >= min_tokens || allow_small {
            add_token_range(
                &mut chunks,
                path,
                src,
                offsets,
                start..end_limit + 1,
                &mut last_line,
                &mut last_byte,
            );
        }

        if end_limit == offsets_last {
            return Ok(chunks);
        }

        // overlap：计算新的 start token index（保证严格推进，避免死循环）
        let diff = strategy.next_subdivision(end_limit - start);
        let mut mid = start.saturating_add(diff);
        if mid >= end_limit {
            mid = end_limit.saturating_sub(1);
        }

        let next_newline_diff =
            (mid..end_limit).find(|&i| src[offsets[i].0..offsets[i + 1].0].contains('\n'));
        let prev_newline_diff = (start + (diff / 2)..mid)
            .rfind(|&i| src[offsets[i].0..offsets[i + 1].0].contains('\n'))
            .map(|t| t + 1);

        start = match (next_newline_diff, prev_newline_diff) {
            (Some(n), None) | (None, Some(n)) => n,
            (Some(n), Some(p)) => {
                if n.saturating_sub(mid) < mid.saturating_sub(p) {
                    n
                } else {
                    p
                }
            }
            (None, None) => (mid..end_limit)
                .find(|&i| {
                    ids.get(i + 1)
                        .and_then(|id| tokenizer.id_to_token(*id))
                        .is_none_or(|s| !s.starts_with("##"))
                })
                .unwrap_or(mid),
        };

        // clamp 到 token_range
        if start < token_range.start {
            start = token_range.start;
        }
        if start >= offsets_last {
            return Ok(chunks);
        }
    }
}

fn make_index_chunk_by_bytes(
    path: &str,
    src: &str,
    start_byte: usize,
    end_byte: usize,
) -> IndexChunk {
    if end_byte <= start_byte {
        return IndexChunk {
            path: path.to_string(),
            start_byte,
            end_byte,
            start_line: 0,
            end_line: 0,
            text: String::new(),
        };
    }
    let start = point(src, start_byte, 0, 0);
    let end = point(src, end_byte, 0, 0);
    IndexChunk {
        path: path.to_string(),
        start_byte,
        end_byte,
        start_line: start.line,
        end_line: end.line,
        text: src[start_byte..end_byte].to_string(),
    }
}

fn token_range_for_byte_range(
    offsets: &[(usize, usize)],
    start_byte: usize,
    end_byte: usize,
) -> Option<Range<usize>> {
    if end_byte <= start_byte || offsets.is_empty() {
        return None;
    }

    let start = lower_bound_by(offsets, |&(_, e)| e > start_byte);
    let end = lower_bound_by(offsets, |&(s, _)| s >= end_byte);
    if start >= end { None } else { Some(start..end) }
}

fn lower_bound_by<T, F>(slice: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let mut left = 0usize;
    let mut right = slice.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(&slice[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

fn add_token_range(
    chunks: &mut Vec<IndexChunk>,
    path: &str,
    src: &str,
    offsets: &[(usize, usize)],
    o: Range<usize>,
    last_line: &mut usize,
    last_byte: &mut usize,
) {
    let start_byte = offsets[o.start].0;
    let end_byte = offsets.get(o.end).map_or(src.len(), |&(s, _)| s);
    if end_byte <= start_byte {
        return;
    }

    let start = point(src, start_byte, *last_line, *last_byte);
    let end = point(src, end_byte, *last_line, *last_byte);
    (*last_line, *last_byte) = (start.line, start.byte);

    chunks.push(IndexChunk {
        path: path.to_string(),
        start_byte,
        end_byte,
        // 统一使用 0-based 行号；展示层再做 +1
        start_line: start.line,
        end_line: end.line,
        text: src[start_byte..end_byte].to_string(),
    });
}

fn point(src: &str, byte: usize, last_line: usize, last_byte: usize) -> core::text_range::Point {
    let line = src.as_bytes()[last_byte..byte]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + last_line;
    let column = if let Some(last_nl) = src[..byte].rfind('\n') {
        byte - last_nl
    } else {
        byte
    };
    core::text_range::Point { byte, column, line }
}

fn by_lines_index_chunks(path: &str, src: &str, size: usize) -> Vec<IndexChunk> {
    if size == 0 {
        return Vec::new();
    }
    let ends = std::iter::once(0)
        .chain(src.match_indices('\n').map(|(i, _)| i))
        .enumerate()
        .collect::<Vec<_>>();

    if ends.is_empty() {
        return Vec::new();
    }

    let last = src.len().saturating_sub(1);
    let last_line = *ends.last().map(|(idx, _)| idx).unwrap_or(&0);

    ends.iter()
        .copied()
        .step_by(size)
        .zip(
            ends.iter()
                .copied()
                .step_by(size)
                .skip(1)
                .chain([(last_line, last)]),
        )
        .filter(|((_, start_byte), (_, end_byte))| start_byte < end_byte)
        .map(
            |((start_line0, start_byte), (end_line0, end_byte))| IndexChunk {
                path: path.to_string(),
                start_byte,
                end_byte,
                // 统一使用 0-based 行号；end_line 为包含式
                start_line: start_line0,
                end_line: end_line0,
                text: src[start_byte..end_byte].to_string(),
            },
        )
        .collect()
}

/// 将检索命中的 IndexChunk 扩展为 ContextChunk（函数/类级上下文）。
///
/// MVP：
/// - 解析 scope graph
/// - 找到“覆盖 hit 的最小 top-level scope”，输出对应范围
/// - 若找不到，退化为命中行附近的行窗口
pub fn refill_chunks(
    path: &str,
    src: &[u8],
    lang_id: &str,
    hits: &[IndexChunk],
    opt: RefillOptions,
) -> Result<Vec<ContextChunk>, ChunkError> {
    let ts = TreeSitterFile::try_build(src, lang_id).map_err(ChunkError::Parse)?;
    let graph = ts.scope_graph().map_err(ChunkError::Parse)?;
    let line_starts = compute_line_starts(src);
    let total_lines = line_starts.len().saturating_sub(1);

    let mut out = Vec::new();
    let root = find_root_scope_idx(&graph);
    let top_scopes = root
        .map(|root_idx| top_level_scopes(&graph, root_idx))
        .unwrap_or_default();

    for hit in hits {
        // 1) 优先：找最小 enclosing top-level scope
        let mut best: Option<core::text_range::TextRange> = None;
        for (_, r) in &top_scopes {
            if r.start.byte <= hit.start_byte && r.end.byte >= hit.end_byte {
                match best {
                    None => best = Some(*r),
                    Some(cur) if r.size() < cur.size() => best = Some(*r),
                    _ => {}
                }
            }
        }

        let chunk = if let Some(r) = best {
            let snippet = String::from_utf8_lossy(&src[r.start.byte..r.end.byte]).to_string();
            ContextChunk {
                path: path.to_string(),
                alias: 0,
                snippet,
                // 统一使用 0-based 行号；end_line 为包含式
                start_line: r.start.line,
                end_line: r.end.line,
                reason: "refill from enclosing top-level scope".to_string(),
            }
        } else {
            // 2) fallback：命中行附近窗口
            let hit_line0 = hit.start_line;
            let half = opt.fallback_window_lines / 2;
            let start0 = hit_line0.saturating_sub(half);
            let end0 = (hit_line0 + half).min(total_lines.saturating_sub(1));
            let (b0, b1) = byte_range_for_lines(&line_starts, start0, end0);
            let snippet = if b1 > b0 {
                String::from_utf8_lossy(&src[b0..b1]).to_string()
            } else {
                String::new()
            };
            ContextChunk {
                path: path.to_string(),
                alias: 0,
                snippet,
                // 统一使用 0-based 行号；end_line 为包含式
                start_line: start0,
                end_line: end0,
                reason: "refill fallback window".to_string(),
            }
        };

        out.push(chunk);
    }

    // alias 连续化
    for (i, c) in out.iter_mut().enumerate() {
        c.alias = i;
    }
    Ok(out)
}

fn make_chunk(
    path: &str,
    src: &[u8],
    start_line0: usize,
    end_line0: usize,
    start_byte: usize,
    end_byte: usize,
) -> CodeChunk {
    let snippet = String::from_utf8_lossy(&src[start_byte..end_byte]).to_string();
    CodeChunk {
        path: path.to_string(),
        alias: 0,
        snippet,
        // 统一使用 0-based 行号；end_line 用“包含式”更直观（若 end_line0 < start_line0 则纠正）
        start_line: start_line0,
        end_line: end_line0.max(start_line0),
    }
}

fn compute_line_starts(src: &[u8]) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, b) in src.iter().enumerate() {
        if *b == b'\n' {
            starts.push(i + 1);
        }
    }
    // 末尾 sentinel：方便用 line -> byte range
    starts.push(src.len());
    starts
}

fn byte_range_for_lines(
    line_starts: &[usize],
    start_line0: usize,
    end_line0: usize,
) -> (usize, usize) {
    let start = *line_starts.get(start_line0).unwrap_or(&0);
    // end_line0 为包含式：取 end_line0+1 的起始 offset
    let end_line_exclusive = end_line0.saturating_add(1);
    let end = *line_starts
        .get(end_line_exclusive)
        .unwrap_or(&line_starts[line_starts.len() - 1]);
    (start, end)
}

fn sliding_window_by_lines(
    path: &str,
    src: &[u8],
    line_starts: &[usize],
    start_line0: usize,
    end_line0: usize,
    max_lines: usize,
    overlap_lines: usize,
) -> Vec<CodeChunk> {
    if max_lines == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut cur = start_line0;
    let end = end_line0.max(start_line0);
    let step = max_lines.saturating_sub(overlap_lines).max(1);

    while cur <= end {
        let window_end = (cur + max_lines - 1).min(end);
        let (b0, b1) = byte_range_for_lines(line_starts, cur, window_end);
        if b1 > b0 {
            out.push(make_chunk(path, src, cur, window_end, b0, b1));
        }
        if window_end == end {
            break;
        }
        cur = cur.saturating_add(step);
    }
    out
}

fn find_root_scope_idx(graph: &intelligence::ScopeGraph) -> Option<petgraph::graph::NodeIndex> {
    use intelligence::NodeKind;
    use intelligence::scope_resolution::EdgeKind;

    let mut best: Option<(petgraph::graph::NodeIndex, usize)> = None;
    for idx in graph.graph.node_indices() {
        let Some(NodeKind::Scope(scope)) = graph.graph.node_weight(idx) else {
            continue;
        };

        // root scope 没有 ScopeToScope 的出边
        let has_parent = graph
            .graph
            .edges(idx)
            .any(|e| *e.weight() == EdgeKind::ScopeToScope);

        if has_parent {
            continue;
        }

        let size = scope.range.size();
        match best {
            None => best = Some((idx, size)),
            Some((_, best_size)) if size > best_size => best = Some((idx, size)),
            _ => {}
        }
    }
    best.map(|(idx, _)| idx)
}

fn top_level_scopes(
    graph: &intelligence::ScopeGraph,
    root_idx: petgraph::graph::NodeIndex,
) -> Vec<(petgraph::graph::NodeIndex, core::text_range::TextRange)> {
    use intelligence::NodeKind;
    use intelligence::scope_resolution::EdgeKind;
    use petgraph::visit::EdgeRef;

    graph
        .graph
        .edges_directed(root_idx, petgraph::Direction::Incoming)
        .filter_map(|e| {
            if *e.weight() != EdgeKind::ScopeToScope {
                return None;
            }
            let child = e.source();
            match graph.graph.node_weight(child) {
                Some(NodeKind::Scope(s)) => Some((child, s.range)),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    fn dummy_tokenizer() -> Tokenizer {
        // 一个极简 tokenizer：WordLevel + Whitespace。
        // 仅用于单元测试，让 encode/offsets 可用（未知词会落到 [UNK]）。
        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0u32);
        vocab.insert("fn".to_string(), 1u32);
        vocab.insert("let".to_string(), 2u32);
        vocab.insert("return".to_string(), 3u32);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace));
        tok
    }

    #[test]
    fn chunk_rust_functions_as_top_level_scopes() {
        let code = r#"\
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let _ = add(1, 2);
}
"#;

        let chunks =
            chunk_source("mem.rs", code.as_bytes(), "Rust", ChunkOptions::default()).unwrap();
        // 至少切出 add/main 两个 scope（实现细节可能多一些 scope，这里只保证非空且包含关键字）
        assert!(!chunks.is_empty());
        let merged = chunks
            .iter()
            .map(|c| c.snippet.as_str())
            .collect::<String>();
        assert!(merged.contains("fn add"));
        assert!(merged.contains("fn main"));
    }

    #[test]
    fn refill_from_index_chunk_to_context_chunk() {
        let code = r#"\
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(1, 2);
    println!("{}", result);
}
"#;

        let tok = dummy_tokenizer();
        let opt = IndexChunkOptions {
            min_chunk_tokens: 1,
            max_chunk_tokens: 64,
            overlap: OverlapStrategy::Partial(0.5),
            fallback_lines: 80,
        };
        let idx_chunks = index_chunks("", "mem.rs", code.as_bytes(), "Rust", &tok, opt);
        // hybrid 策略下，IndexChunk 以语义边界（函数）为主，因此这里选择命中 add 函数体
        let hit = idx_chunks
            .iter()
            .find(|c| c.text.contains("a + b") || c.text.contains("fn add"))
            .cloned()
            .unwrap_or_else(|| idx_chunks[0].clone());

        let ctx = refill_chunks(
            "mem.rs",
            code.as_bytes(),
            "Rust",
            &[hit],
            RefillOptions::default(),
        )
        .unwrap();

        assert_eq!(ctx.len(), 1);
        assert!(ctx[0].snippet.contains("fn add"));
    }
}
