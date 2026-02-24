use core::code_chunk::OverlapStrategy;
use core::code_chunk::{
    ChunkOptions, CodeChunk, ContextChunk, IndexChunk, IndexChunkBuildError, IndexChunkOptions,
    RefillOptions,
};
use intelligence::TreeSitterFile;
use std::ops::Range;
use tokenizers::Tokenizer;

// Re-export error from error module
pub use crate::error::ChunkError;

/// Intelligent chunking based on `luna/src/intelligence` scope graph.
///
/// - Prioritizes scopes directly under root scope as semantic chunks (typically functions/classes/methods).
/// - Falls back to sliding window by line for oversized scopes.
/// - Falls back to full-file sliding window if no top-level scopes are found.
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
            // Skip empty ranges
            if range.end.byte <= range.start.byte {
                continue;
            }

            // Oversized scope -> sliding window within scope
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
        // Global fallback: sliding window by line covering the entire file
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

    // Normalize alias
    for (i, c) in chunks.iter_mut().enumerate() {
        c.alias = i;
    }

    Ok(chunks)
}

/// Generate IndexChunk (for vector/keyword retrieval).
///
/// Chunking strategy (modern hybrid):
/// - Semantic boundary first: prioritize top-level scopes (functions/classes/methods) for candidate chunks.
/// - Budget normalization: only split oversized scopes within token budget (prioritize newline, then BPE boundaries, with overlap if necessary).
/// - Fallback: degrade to full-file token chunking when no AST/parse failure; degrade to line-based chunking when tokenizer unavailable.
///
/// Notes:
/// - `repo` is used to account for `repo\tpath\n` prefix in token budget (can be empty string).
/// - `lang_id` is used to parse top-level scopes (e.g., "Rust").
/// - If tokenizer encoding fails, degrades to line-based chunking (`fallback_lines`).
pub fn index_chunks(
    repo: &str,
    path: &str,
    src: &[u8],
    lang_id: &str,
    tokenizer: &Tokenizer,
    opt: IndexChunkOptions,
) -> Vec<IndexChunk> {
    // Modern hybrid strategy:
    // 1) Prioritize top-level scopes for "semantic boundaries" (functions/classes/methods)
    // 2) Split within each scope by token budget (only if oversized), ensuring controllable retrieval unit size
    // 3) Fallback to full-file token chunking on parse failure/no scopes; fallback to line-based chunking if tokenizer unavailable

    let src = String::from_utf8_lossy(src);

    let encoding = match tokenizer.encode(src.as_ref(), true) {
        Ok(e) => e,
        Err(_) => return by_lines_index_chunks(path, &src, opt.fallback_lines),
    };

    let offsets_all = encoding.get_offsets();
    let ids_all = encoding.get_ids();

    let offsets_len = offsets_all.len().saturating_sub(1);
    let offsets: &[(usize, usize)] = if offsets_all.get(offsets_len).map(|o| o.0) == Some(0) {
        &offsets_all[..offsets_len]
    } else {
        offsets_all
    };

    // ids used for BPE boundary detection; conservative trimming to avoid out-of-bounds when lengths differ
    let ids_len = offsets.len().saturating_add(1).min(ids_all.len());
    let ids = &ids_all[..ids_len];

    let token_bounds = opt.min_chunk_tokens..opt.max_chunk_tokens;
    let min_tokens = token_bounds.start;

    // Calculate `repo\tpath\n` prefix token usage (for budget deduction)
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

    // Semantic boundaries: top-level scopes
    let top_scopes = TreeSitterFile::try_build(src.as_bytes(), lang_id)
        .and_then(|ts| ts.scope_graph())
        .ok()
        .and_then(|graph| find_root_scope_idx(&graph).map(|root| top_level_scopes(&graph, root)))
        .unwrap_or_default();

    let mut out = Vec::new();

    if !top_scopes.is_empty() {
        // Prioritize semantic boundaries: normalize token budget within each scope (split only if oversized)
        for (_, r) in top_scopes {
            if r.end.byte <= r.start.byte {
                continue;
            }

            let Some(token_range) = token_range_for_byte_range(offsets, r.start.byte, r.end.byte)
            else {
                continue;
            };
            let token_len = token_range.end.saturating_sub(token_range.start);

            // Keep small scopes: semantic boundaries take priority over min_tokens (otherwise small functions/classes would be missed)
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

            // Oversized scope: split within scope by token budget
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
                    // Extreme cases (e.g., tokenizer/id mismatch), fallback to line-based chunking
                    out.extend(by_lines_index_chunks(path, &src, opt.fallback_lines));
                }
            }
        }
    }

    if out.is_empty() {
        // Fallback: full-file token chunking (aligns with legacy behavior)
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
            // ids and offsets from different sources may be out of bounds: treat as boundary
            ids.get(i + 1)
                .and_then(|id| tokenizer.id_to_token(*id))
                .is_none_or(|s| !s.starts_with("##"))
        }) {
            next_boundary
        } else {
            next_limit
        };

        let token_len = end_limit.saturating_sub(start);
        // Allow tail chunk to be smaller than min_tokens: ensure full range coverage
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

        // Overlap: calculate new start token index (ensure strict progress, avoid infinite loop)
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

        // Clamp to token_range
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
                start_line: start_line0,
                end_line: end_line0,
                text: src[start_byte..end_byte].to_string(),
            },
        )
        .collect()
}

/// Expand retrieved IndexChunk hits into ContextChunk (function/class-level context).
///
/// - Parse scope graph
/// - Find "minimum enclosing top-level scope covering hit", output corresponding range
/// - Fallback to line window near hit line if not found
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
        // Priority 1: find minimum enclosing top-level scope
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
                start_line: r.start.line,
                end_line: r.end.line,
                reason: "refill from enclosing top-level scope".to_string(),
            }
        } else {
            // Fallback: line window near hit
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
                start_line: start0,
                end_line: end0,
                reason: "refill fallback window".to_string(),
            }
        };

        out.push(chunk);
    }

    // Normalize alias to be sequential
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
    // Trailing sentinel: facilitate line -> byte range conversion
    starts.push(src.len());
    starts
}

fn byte_range_for_lines(
    line_starts: &[usize],
    start_line0: usize,
    end_line0: usize,
) -> (usize, usize) {
    let start = *line_starts.get(start_line0).unwrap_or(&0);
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

        // Root scope has no ScopeToScope outgoing edges
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
