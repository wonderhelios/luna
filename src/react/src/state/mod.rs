//! State summary and analysis for ReAct agent
//!
//! This module provides functionality to summarize the current state
//! of the codebase for the LLM planner, including:
//! - Basic state summary with definition extraction
//! - Enhanced state summary using ScopeGraph analysis

use core::code_chunk::{ContextChunk, IndexChunk};
use std::collections::BTreeMap;

// Re-export public types and functions
mod enhanced;
mod patterns;

pub use enhanced::{summarize_state_enhanced, SymbolInfo};
pub use patterns::{extract_definition_name, is_definition_chunk};

/// Maximum number of context chunks to show in state summaries
const STATE_CONTEXT_PREVIEW_MAX: usize = 6;

/// Summarize the current state for the LLM planner
///
/// This provides a concise view of:
/// - Number of hits and context chunks
/// - Whether definitions were found
/// - Preview of context chunks
pub fn summarize_state(hits: &[IndexChunk], context: &[ContextChunk]) -> String {
    let definition_names: Vec<String> =
        context.iter().filter_map(extract_definition_name).collect();

    let has_definition = !definition_names.is_empty();
    let mut s = String::new();
    s.push_str(&format!(
        "hits={} context_chunks={} has_definition={}",
        hits.len(),
        context.len(),
        has_definition
    ));

    if !definition_names.is_empty() {
        s.push_str(&format!(" definitions=[{}]", definition_names.join(", ")));
    }

    // Show specific functions if visible
    let has_list_dir = context
        .iter()
        .any(|c| c.snippet.contains("fn list_dir") || c.snippet.contains("pub fn list_dir"));
    let has_sort = context.iter().any(|c| {
        c.snippet.contains("entries.sort_by") || c.snippet.contains("entries.sort_by_key")
    });
    if has_list_dir {
        s.push_str(&format!(
            " visible_functions=[list_dir] has_sort={}",
            has_sort
        ));
    }
    s.push('\n');

    for c in context.iter().take(STATE_CONTEXT_PREVIEW_MAX) {
        let is_def = is_definition_chunk(c);
        let def_name = extract_definition_name(c);
        s.push_str(&format!(
            "- {}:{}..={}{}{} reason={}\n",
            c.path,
            c.start_line + 1,
            c.end_line + 1,
            if is_def { " [def]" } else { "" },
            if let Some(name) = def_name {
                format!(" ({})", name)
            } else {
                String::new()
            },
            c.reason,
        ));
    }
    if context.len() > STATE_CONTEXT_PREVIEW_MAX {
        s.push_str("- ...\n");
    }
    s
}

/// Merge hit lists, deduplicating by (path, start_byte, end_byte)
pub fn merge_hits(mut base: Vec<IndexChunk>, more: Vec<IndexChunk>) -> Vec<IndexChunk> {
    let mut uniq: BTreeMap<(String, usize, usize), IndexChunk> = BTreeMap::new();
    for h in base.drain(..).chain(more.into_iter()) {
        let key = (h.path.clone(), h.start_byte, h.end_byte);
        uniq.entry(key).or_insert(h);
    }
    uniq.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_state_basic() {
        let hits = vec![];
        let context = vec![ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "pub fn test() {}".to_string(),
            start_line: 0,
            end_line: 0,
            reason: "search".to_string(),
        }];

        let summary = summarize_state(&hits, &context);
        assert!(summary.contains("hits=0"));
        assert!(summary.contains("context_chunks=1"));
        assert!(summary.contains("has_definition=true"));
        assert!(summary.contains("definitions=[test]"));
    }

    #[test]
    fn test_merge_hits() {
        let hits1 = vec![IndexChunk {
            path: "a.rs".to_string(),
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            end_line: 5,
            text: "fn a()".to_string(),
        }];
        let hits2 = vec![IndexChunk {
            path: "a.rs".to_string(),
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            end_line: 5,
            text: "fn a()".to_string(),
        }];

        let merged = merge_hits(hits1, hits2);
        assert_eq!(merged.len(), 1);
    }
}
