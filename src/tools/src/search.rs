//! Code search operations for agents

use crate::detect_lang_id;
use crate::{ToolResult, ToolTrace};
use core::code_chunk::{ContextChunk, IndexChunk, IndexChunkOptions, RefillOptions};
use index;
use intelligence::TreeSitterFile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

// ============================================================================
// Search Backend Abstraction
// ============================================================================

/// Search backend abstraction: decouples placeholder keyword search from future vector/hybrid search.
///
/// Constraint: Regardless of backend changes, must produce a unified `IndexChunk` hit protocol
/// for use by Refill/ContextEngine.
pub trait SearchBackend: Send + Sync {
    fn search(
        &self,
        repo_root: &Path,
        query: &str,
        tokenizer: &Tokenizer,
        idx_opt: IndexChunkOptions,
        opt: SearchCodeOptions,
    ) -> ToolResult<(Vec<IndexChunk>, Vec<ToolTrace>)>;
}

/// Keyword placeholder search backend: scans repo files, matches query terms, and normalizes hits using `IndexChunk`.
#[derive(Debug, Clone, Default)]
pub struct KeywordSearchBackend;

impl SearchBackend for KeywordSearchBackend {
    fn search(
        &self,
        repo_root: &Path,
        query: &str,
        tokenizer: &Tokenizer,
        idx_opt: IndexChunkOptions,
        opt: SearchCodeOptions,
    ) -> ToolResult<(Vec<IndexChunk>, Vec<ToolTrace>)> {
        let mut trace = Vec::new();
        let q = query.trim();

        if q.is_empty() {
            return Ok((Vec::new(), trace));
        }

        let terms: Vec<&str> = q
            .split_whitespace()
            .filter(|t| !t.trim().is_empty())
            .collect();

        if terms.is_empty() {
            return Ok((Vec::new(), trace));
        }

        // Single-term fast path: exact match
        let is_single_term = terms.len() == 1;

        let mut hits = Vec::new();
        let mut files_scanned = 0usize;

        for entry in walkdir::WalkDir::new(repo_root)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                let path = e.path();

                // Skip ignored directories (but still traverse into them to find files)
                if path.is_dir() {
                    return !opt.ignore_dirs.iter().any(|d| name == *d);
                }

                // Only process files
                if !path.is_file() {
                    return false;
                }

                // Check file extension
                detect_lang_id(path).is_some()
            })
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();

            if files_scanned >= opt.max_files {
                break;
            }
            files_scanned += 1;

            // Read file
            let metadata = fs::metadata(path)?;
            if metadata.len() > opt.max_file_bytes as u64 {
                continue;
            }

            let src = fs::read(path)?;

            // Check if file contains query terms
            let src_str = String::from_utf8_lossy(&src);
            let matches = if is_single_term {
                src_str.contains(terms[0])
            } else {
                terms.iter().all(|t| src_str.contains(t))
            };

            if !matches {
                continue;
            }

            // Generate IndexChunks
            let lang_id = detect_lang_id(path).unwrap_or("");
            let chunks = index::index_chunks(
                "",
                &path.to_string_lossy(),
                &src,
                lang_id,
                tokenizer,
                idx_opt.clone(),
            );

            for chunk in chunks {
                let chunk_matches = if is_single_term {
                    chunk.text.contains(terms[0])
                } else {
                    terms.iter().all(|t| chunk.text.contains(t))
                };

                if chunk_matches {
                    hits.push(chunk);
                }
            }
        }

        // Deduplicate hits by (path, start_byte, end_byte)
        let mut uniq: BTreeMap<(String, usize, usize), IndexChunk> = BTreeMap::new();
        for h in hits {
            let key = (h.path.clone(), h.start_byte, h.end_byte);
            uniq.entry(key).or_insert(h);
        }

        let hits: Vec<_> = uniq.into_values().take(opt.max_hits).collect();

        trace.push(ToolTrace {
            tool: "search_code".to_string(),
            summary: format!(
                "backend=keyword scanned={} files, found={} hits",
                files_scanned,
                hits.len()
            ),
        });

        Ok((hits, trace))
    }
}

// ============================================================================
// Search Options
// ============================================================================

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

// ============================================================================
// Keyword Search
// ============================================================================

/// Keyword placeholder for search_code: scan repo files, normalize hits using IndexChunk protocol
///
/// Returns: IndexChunk hits (each chunk's text contains query)
pub fn search_code_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &Tokenizer,
    idx_opt: IndexChunkOptions,
    opt: SearchCodeOptions,
) -> ToolResult<(Vec<IndexChunk>, Vec<ToolTrace>)> {
    KeywordSearchBackend::default().search(repo_root, query, tokenizer, idx_opt, opt)
}

// ============================================================================
// Refill Hits
// ============================================================================

/// Refill IndexChunk hits into ContextChunks (function/class-level context)
pub fn refill_hits(
    repo_root: &Path,
    hits: &[IndexChunk],
    opt: RefillOptions,
) -> ToolResult<(Vec<ContextChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();
    let mut context = Vec::new();

    // Group hits by file
    let mut by_file: BTreeMap<String, Vec<IndexChunk>> = BTreeMap::new();
    for h in hits {
        by_file.entry(h.path.clone()).or_default().push(h.clone());
    }

    for (path, file_hits) in by_file {
        let full_path = repo_root.join(&path);

        // Read file
        let src = fs::read(&full_path)?;
        let lang_id = detect_lang_id(&full_path).unwrap_or("");

        // Refill using index module
        let mut file_context = index::refill_chunks(&path, &src, lang_id, &file_hits, opt.clone())
            .map_err(|e| crate::ToolError::search_failed(format!("refill failed for {}: {:?}", path, e)))?;

        context.append(&mut file_context);
    }

    // Deduplicate by (path, start_line, end_line)
    let mut uniq: BTreeMap<(String, usize, usize), ContextChunk> = BTreeMap::new();
    for c in context {
        let key = (c.path.clone(), c.start_line, c.end_line);
        uniq.entry(key).or_insert(c);
    }

    let context: Vec<_> = uniq.into_values().collect();

    trace.push(ToolTrace {
        tool: "refill_hits".to_string(),
        summary: format!(
            "refilled {} hits into {} context chunks",
            hits.len(),
            context.len()
        ),
    });

    Ok((context, trace))
}

// ============================================================================
// Symbol-based Search
// ============================================================================

/// Find definitions of a symbol name across the repository
pub fn find_symbol_definitions(
    repo_root: &Path,
    symbol_name: &str,
    max_results: usize,
) -> ToolResult<Vec<SymbolLocation>> {
    let mut results = Vec::new();

    for entry in walkdir::WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let path = e.path();
            if path.is_dir() {
                let name = e.file_name().to_string_lossy();
                return !matches!(
                    name.as_ref(),
                    "target" | "node_modules" | ".git" | "dist" | "build"
                );
            }
            path.is_file() && detect_lang_id(path).is_some()
        })
    {
        let entry = entry.map_err(|e| crate::ToolError::search_failed(format!("walk error: {}", e)))?;
        let path = entry.path();

        if results.len() >= max_results {
            break;
        }

        let src = fs::read(path)?;
        let lang_id = detect_lang_id(path).unwrap_or("");

        let ts_file = match TreeSitterFile::try_build(&src, lang_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let scope_graph = match ts_file.scope_graph() {
            Ok(g) => g,
            Err(_) => continue,
        };

        let src_str = String::from_utf8_lossy(&src);

        for idx in scope_graph.graph.node_indices() {
            if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
                let name = String::from_utf8_lossy(def.name(src_str.as_bytes()));
                if name == symbol_name {
                    results.push(SymbolLocation {
                        path: path
                            .strip_prefix(repo_root)
                            .unwrap_or(path)
                            .to_string_lossy()
                            .to_string(),
                        start_line: def.range.start.line + 1,
                        end_line: def.range.end.line + 1,
                        kind: "definition".to_string(),
                    });
                }
            }
        }
    }

    Ok(results)
}

/// Symbol location for search results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolLocation {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: String,
}
