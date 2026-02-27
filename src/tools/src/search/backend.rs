use crate::{detect_lang_id, Result, ToolTrace};
use core::code_chunk::{IndexChunk, IndexChunkOptions};
use index;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

use super::options::SearchCodeOptions;

/// 搜索后端抽象：用于把"占位关键词检索"与未来的"向量/混合检索"解耦。
///
/// 约束：无论后端如何变化，都应产出统一的 `IndexChunk` 命中协议，供 Refill/ContextEngine 使用。
pub trait SearchBackend: Send + Sync {
    fn search(
        &self,
        repo_root: &Path,
        query: &str,
        tokenizer: &Tokenizer,
        idx_opt: IndexChunkOptions,
        opt: SearchCodeOptions,
    ) -> Result<(Vec<IndexChunk>, Vec<ToolTrace>)>;
}

/// 关键词占位检索后端：扫描仓库文件，匹配 query terms，并用 `IndexChunk` 规范化命中。
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
    ) -> Result<(Vec<IndexChunk>, Vec<ToolTrace>)> {
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
