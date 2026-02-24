//! Context Engine for ReAct Loop
//!
//! This module handles:
//! - Selecting and ranking context chunks
//! - Rendering context for LLM consumption
//! - Token budget management

use crate::ContextPack;
use anyhow::Result;
use core::code_chunk::ContextChunk;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

use tools::fs::read_file_by_lines;

// ============================================================================
// Context Engine Options
// ============================================================================

/// Options for context rendering and selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineOptions {
    /// Maximum number of chunks to include
    pub max_chunks: usize,

    /// Maximum total tokens (0 = no limit)
    pub max_total_tokens: usize,

    /// Merge chunks within this many lines
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

// ============================================================================
// Context Rendering
// ============================================================================

/// Render ContextPack into prompt context paragraphs ready for LLM injection
pub fn render_prompt_context(
    repo_root: &Path,
    pack: &ContextPack,
    tokenizer: &Tokenizer,
    opt: ContextEngineOptions,
) -> Result<String> {
    let selected = select_context_chunks(
        repo_root,
        &pack.hits,
        &pack.context,
        tokenizer,
        opt.clone(),
    )?;

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

// ============================================================================
// Context Selection
// ============================================================================

/// Select and rank context chunks based on hits
fn select_context_chunks(
    repo_root: &Path,
    hits: &[core::code_chunk::IndexChunk],
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_engine_options_default() {
        let opt = ContextEngineOptions::default();
        assert_eq!(opt.max_chunks, 8);
        assert_eq!(opt.max_total_tokens, 2000);
        assert_eq!(opt.merge_gap_lines, 3);
    }
}
