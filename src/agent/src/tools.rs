use crate::types::{ContextPack, ToolName, ToolTrace};
use anyhow::Result;
use core::code_chunk::{ContextChunk, IndexChunk, IndexChunkOptions, RefillOptions};
use core::symbol::Symbol;
use intelligence::{ALL_LANGUAGES, TreeSitterFile};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

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

pub fn read_file_by_lines(
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

    let terms = q
        .split_whitespace()
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Ok((Vec::new(), trace));
    }

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
            let content_lower = content.to_lowercase();
            if !terms.iter().any(|t| content_lower.contains(t)) {
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
                let text_lower = c.text.to_lowercase();
                if terms.iter().any(|t| text_lower.contains(t)) {
                    hits.push(c);
                }
            }
        }
    }

    trace.push(ToolTrace {
        tool: ToolName::SearchCode,
        summary: format!(
            "keyword search scanned_files={} hits={} terms={:?}",
            scanned_files,
            hits.len(),
            terms,
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

    // Dedup: multiple IndexChunk hits may fall in the same enclosing scope
    let raw_count = out.len();
    let out = core::code_chunk::dedup_context_chunks(out);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_map_by_range_works() {
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
