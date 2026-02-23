use anyhow::Result;
use core::code_chunk::{ContextChunk, IndexChunk, IndexChunkOptions, RefillOptions};
use core::symbol::Symbol;
use intelligence::{ALL_LANGUAGES, TreeSitterFile};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::{fs, str};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolName {
    ReadFile,
    SearchCode,
    RefillChunks,
    ListSymbols,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTrace {
    pub tool: ToolName,
    pub summary: String,
}

// Minimal context pack for LLM/frontend (Context Engine input/output carrier)
pub struct ContextPack {
    pub query: String,
    // Retrieval hit protocol, to be replaced with vector/hybrid retrieval
    pub hits: Vec<IndexChunk>,
    // Readable context entering prompt (refilled function/class-level context)
    pub context: Vec<ContextChunk>,
    // Tool call trace (for debugging/explainability)
    pub trace: Vec<ToolTrace>,
}

// Context Engine: fixed N ContextChunks + budget trimming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineOptions {
    pub max_chunks: usize,
    // 0 means not limit
    pub max_total_tokens: usize,
    pub merge_gap_lines: usize,
}

impl Default for ContextEngineOptions {
    fn default() -> Self {
        Self {
            max_chunks: 8,
            max_total_tokens: 2_000,
            merge_gap_lines: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCodeOptions {
    pub max_files: usize,
    pub max_hits: usize,
    pub max_file_bytes: usize,
    pub ignore_dirs: Vec<String>,
}

impl Default for SearchCodeOptions {
    fn default() -> Self {
        Self {
            max_files: 8_000,
            max_hits: 64,
            max_file_bytes: 500 * 1_000,
            ignore_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "dist".to_string(),
                "build".to_string(),
            ],
        }
    }
}

// Read file
pub fn read_file(path: &Path, range: Option<(usize, usize)>) -> Result<String> {
    let s = fs::read_to_string(path)?;
    if let Some((start, end)) = range {
        if start > end {
            return Ok(String::new());
        }
        let mut out = String::new();
        for (i, line) in s.lines().enumerate() {
            if i < start {
                continue;
            }
            if i > end {
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
        Ok(out)
    } else {
        Ok(s)
    }
}

fn read_file_by_lines(
    repo_root: &Path,
    rel_path: &str,
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let full = repo_root.join(rel_path);
    read_file(&full, Some((start_line, end_line)))
}

// Infer language id from file extension (for tree-sitter parsing)
pub fn detect_lang_id(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    ALL_LANGUAGES
        .iter()
        .copied()
        .find(|cfg| cfg.file_extensions.iter().any(|e| e.to_lowercase() == ext))
        .and_then(|cfg| cfg.language_ids.first().copied())
}

// Keyword placeholder for search_code: scan repo files, normalize hits using IndexChunk protocol
// Returns: IndexChunk hits (each chunk's text contains query)
pub fn search_code_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &Tokenizer,
    idx_opt: IndexChunkOptions,
    opt: SearchCodeOptions,
) -> Result<(Vec<IndexChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();

    let q = query.trim();
    if q.is_empty() {
        return Ok((Vec::new(), trace));
    }
    let q_lower = q.to_lowercase();

    let mut hits = Vec::new();
    let mut scanned_files = 0usize;

    let mut stack = vec![repo_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if scanned_files >= opt.max_files || hits.len() >= opt.max_hits {
            break;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            if scanned_files >= opt.max_files || hits.len() >= opt.max_hits {
                break;
            }
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if let Some(name) = path.file_name().map(|s| s.to_string_lossy().to_string()) {
                    if opt.ignore_dirs.iter().any(|d| d == &name) {
                        continue;
                    }
                }
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.len() as usize > opt.max_file_bytes {
                continue;
            }

            scanned_files += 1;

            let Some(lang_id) = detect_lang_id(&path) else {
                continue;
            };
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Coarse filter: does file content contain query
            let content = String::from_utf8_lossy(&bytes);
            if !content.to_lowercase().contains(&q_lower) {
                continue;
            }

            let rel = path.strip_prefix(repo_root).unwrap_or(&path);
            let rel_str = rel.to_string_lossy().to_string();

            // Normalize using IndexChunk protocol (scope -> token budget)
            let idx_chunks =
                index::index_chunks("", &rel_str, &bytes, lang_id, tokenizer, idx_opt.clone());
            for c in idx_chunks {
                if hits.len() >= opt.max_hits {
                    break;
                }
                if c.text.to_lowercase().contains(&q_lower) {
                    hits.push(c);
                }
            }
        }
    }

    trace.push(ToolTrace {
        tool: ToolName::SearchCode,
        summary: format!(
            "keyword search scanned_files={} hits={} query={:?}",
            scanned_files,
            hits.len(),
            q
        ),
    });
    Ok((hits, trace))
}

// Refill IndexChunk hits into ContextChunk (group by file, parse each file only once)
pub fn refill_hits(
    repo_root: &Path,
    hits: &[IndexChunk],
    opt: RefillOptions,
) -> Result<(Vec<ContextChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();

    let mut hits_by_path: BTreeMap<String, Vec<IndexChunk>> = BTreeMap::new();
    for h in hits {
        hits_by_path
            .entry(h.path.clone())
            .or_default()
            .push(h.clone());
    }
    let file_count = hits_by_path.len();

    let mut out = Vec::new();
    for (rel_path, mut hs) in hits_by_path {
        hs.sort_by_key(|h| h.start_byte);
        let full_path = repo_root.join(&rel_path);
        let bytes = match fs::read(&full_path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let Some(lang_id) = detect_lang_id(&full_path) else {
            continue;
        };
        let ctx = index::refill_chunks(&rel_path, &bytes, lang_id, &hs, opt.clone())?;
        out.extend(ctx);
    }

    // Dedup: multiple IndexChunk hits may fall in the same enclosing scope, refill produces duplicate ContextChunks
    // Dedup by (path, start_line, end_line) for stable CLI/subsequent Context Engine injection
    let raw_count = out.len();
    let mut uniq: BTreeMap<(String, usize, usize), ContextChunk> = BTreeMap::new();
    for c in out {
        let key = (c.path.clone(), c.start_line, c.end_line);
        uniq.entry(key)
            .and_modify(|existing| {
                if existing.reason != c.reason {
                    if !existing.reason.is_empty() {
                        existing.reason.push_str("; ");
                    }
                    existing.reason.push_str(&c.reason);
                }
            })
            .or_insert(c);
    }
    let mut out = uniq.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| {
        (a.path.as_str(), a.start_line, a.end_line).cmp(&(
            b.path.as_str(),
            b.start_line,
            b.end_line,
        ))
    });
    for (i, c) in out.iter_mut().enumerate() {
        c.alias = i;
    }

    trace.push(ToolTrace {
        tool: ToolName::RefillChunks,
        summary: format!(
            "refill files={} context_chunks={} (raw={})",
            file_count,
            out.len(),
            raw_count
        ),
    });
    Ok((out, trace))
}

/// List symbols in file (function/class/variable definitions, etc.)
pub fn list_symbols(path: &Path) -> Result<Vec<Symbol>> {
    let bytes = fs::read(path)?;
    let Some(lang_id) = detect_lang_id(path) else {
        return Ok(Vec::new());
    };

    let ts = TreeSitterFile::try_build(&bytes, lang_id)
        .map_err(|e| anyhow::anyhow!("failed to parse {:?}: {e:?}", path))?;
    let graph = ts
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("failed to  build scope graph {:?}: {e:?}", path))?;

    Ok(graph.symbols())
}

/// search -> refill -> pack
pub fn build_context_pack_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &Tokenizer,
    search_opt: SearchCodeOptions,
    idx_opt: IndexChunkOptions,
    refill_opt: RefillOptions,
) -> Result<ContextPack> {
    let (hits, mut trace) = search_code_keyword(repo_root, query, tokenizer, idx_opt, search_opt)?;
    let (context, mut trace2) = refill_hits(repo_root, &hits, refill_opt)?;
    trace.append(&mut trace2);

    Ok(ContextPack {
        query: query.to_string(),
        hits,
        context,
        trace,
    })
}

// Render ContextPack into prompt context paragraphs ready for LLM injection
pub fn render_prompt_context(
    repo_root: &Path,
    pack: &ContextPack,
    tokenizer: &Tokenizer,
    opt: ContextEngineOptions,
) -> Result<String> {
    let selected =
        select_context_chunks(repo_root, &pack.hits, &pack.context, tokenizer, opt.clone())?;
    let mut out = String::new();
    out.push_str("# Retrieved Context\n\n");
    out.push_str(&format!("Query: {}\n\n", pack.query));
    out.push_str(&format!("Chunks: {}\n\n", selected.len()));

    for (i, c) in selected.iter().enumerate() {
        let start1 = c.start_line + 1;
        let end1 = c.end_line + 1;
        out.push_str(&format!("## [{i:02}] {}:{}..={}\n", c.path, start1, end1));
        if !c.reason.is_empty() {
            out.push_str(&format!("reason: {}\n", c.reason));
        }
        out.push_str("```\n");
        for (ln0, line) in c.snippet.lines().enumerate() {
            out.push_str(&format!("{:>5} {}\n", start1 + ln0, line));
        }
        out.push_str("```\n\n");
    }
    Ok(out)
}

fn select_context_chunks(
    repo_root: &Path,
    hits: &[IndexChunk],
    context: &[ContextChunk],
    tokenizer: &Tokenizer,
    opt: ContextEngineOptions,
) -> Result<Vec<ContextChunk>> {
    // 1) Merge by (path, start_line, end_line)
    let mut by_path: BTreeMap<String, Vec<ContextChunk>> = BTreeMap::new();
    for c in context {
        by_path.entry(c.path.clone()).or_default().push(c.clone());
    }

    let mut merged_all = Vec::new();
    for (path, mut chunks) in by_path {
        chunks.sort_by_key(|c| (c.start_line, c.end_line));
        let mut merged: Vec<(usize, usize, String)> = Vec::new();
        for c in chunks {
            let (s, e) = (c.start_line, c.end_line);
            if let Some((_ms, me, reason)) = merged.last_mut() {
                let gap_ok = s <= me.saturating_add(opt.merge_gap_lines + 1);
                if gap_ok {
                    *me = (*me).max(e);
                    if !c.reason.is_empty() && !reason.contains(&c.reason) {
                        if !reason.is_empty() {
                            reason.push_str("; ");
                        }
                        reason.push_str(&c.reason);
                    }
                    continue;
                }
            }
            merged.push((s, e, c.reason));
        }
        for (s, e, reason) in merged {
            let snippet = read_file_by_lines(repo_root, &path, s, e)?;
            merged_all.push(ContextChunk {
                path: path.clone(),
                alias: 0,
                snippet,
                start_line: s,
                end_line: e,
                reason,
            });
        }
    }
    // 2) Calculate hit count for each ContextChunk, used for ranking
    let mut scored = merged_all
        .into_iter()
        .map(|c| {
            let mut cnt = 0usize;
            for h in hits {
                if h.path == c.path && h.start_line >= c.start_line && h.end_line <= c.end_line {
                    cnt += 1;
                }
            }
            (cnt, c)
        })
        .collect::<Vec<_>>();
    // Prioritize more hits; with same hits, prefer shorter; then sort by path
    scored.sort_by(|(ac, a), (bc, b)| {
        bc.cmp(ac)
            .then_with(|| {
                let asz = a.end_line.saturating_sub(a.start_line);
                let bsz = b.end_line.saturating_sub(b.start_line);
                asz.cmp(&bsz)
            })
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });

    let mut selected = scored
        .into_iter()
        .map(|(_, c)| c)
        .take(opt.max_chunks.max(1))
        .collect::<Vec<_>>();

    // 3) Token budget trimming: drop from end (low priority first)
    if opt.max_total_tokens > 0 {
        let mut total = 0usize;
        let mut keep = Vec::new();
        for c in &selected {
            let s = format!("{}\n{}", c.path, c.snippet);
            let t = tokenizer.encode(s, true).map(|e| e.len()).unwrap_or(0);
            if total + t <= opt.max_total_tokens {
                total += t;
                keep.push(c.clone());
            }
        }
        selected = keep;
    }
    // Normalize alias + stable sort (for downstream referencing)
    selected.sort_by(|a, b| (a.path.as_str(), a.start_line).cmp(&(b.path.as_str(), b.start_line)));
    for (i, c) in selected.iter_mut().enumerate() {
        c.alias = i;
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_by_range_in_refill_hits_keeps_unique_ranges() {
        let mut uniq: BTreeMap<(String, usize, usize), ContextChunk> = BTreeMap::new();
        let mk = |s, e| ContextChunk {
            path: "a.rs".to_string(),
            alias: 0,
            snippet: "x".to_string(),
            start_line: s,
            end_line: e,
            reason: "r".to_string(),
        };
        for c in [mk(1, 10), mk(1, 10), mk(5, 20)] {
            let key = (c.path.clone(), c.start_line, c.end_line);
            uniq.entry(key).or_insert(c);
        }
        assert_eq!(uniq.len(), 2);
    }
}
