//! Tools for Agent Operations
//!
//! This crate provides various tools that agents can use to interact with
//! the codebase: reading files, searching code, listing directories, etc.
//!
//! Tools are organized into modules:
//! - `fs`: File system operations (read, list, edit)
//! - `search`: Code search operations
//! - `terminal`: Terminal command execution

pub mod fs;
pub mod search;
pub mod terminal;

// Re-export error type
pub use error::LunaError;

// Re-export commonly used types
pub use fs::{edit_file, list_dir, read_file, DirEntry, EditOp, EditResult};
pub use search::{
    find_symbol_definitions, refill_hits, search_code_keyword, SearchCodeOptions, SymbolLocation,
};
pub use terminal::{run_terminal, TerminalResult};

use core::code_chunk::{ContextChunk, IndexChunk, IndexChunkOptions};
use intelligence::ALL_LANGUAGES;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Result type alias for tools operations
pub type Result<T> = error::Result<T>;

// ============================================================================
// Common Types
// ============================================================================

/// Simple trace record for tool executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTrace {
    pub tool: String,
    pub summary: String,
}

/// Minimal context pack for LLM/frontend (Context Engine input/output carrier)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub query: String,
    /// Retrieval hit protocol, to be replaced with vector/hybrid retrieval
    pub hits: Vec<IndexChunk>,
    /// Readable context entering prompt (refilled function/class-level context)
    pub context: Vec<ContextChunk>,
    /// Tool call trace (for debugging/explainability)
    pub trace: Vec<ToolTrace>,
}

// ============================================================================
// Language Detection
// ============================================================================

/// Infer language ID from file extension (for tree-sitter parsing)
pub fn detect_lang_id(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    ALL_LANGUAGES
        .iter()
        .copied()
        .find(|cfg| cfg.file_extensions.iter().any(|e| e.to_lowercase() == ext))
        .and_then(|cfg| cfg.language_ids.first().copied())
}

// ============================================================================
// Query Processing
// ============================================================================

/// Extract code identifiers from a natural language query.
///
/// Extracts:
/// - snake_case identifiers (e.g., `context_chunks`, `my_function`)
/// - camelCase identifiers (e.g., `contextChunks`, `myFunction`)
/// - PascalCase identifiers (e.g., `ContextChunk`, `MyClass`)
pub fn extract_code_identifiers(query: &str) -> Vec<String> {
    // Regex 编译开销不低，这里用 Lazy 缓存编译结果，保证：
    // - 只编译一次
    // - 避免每次调用重复分配/编译
    static SNAKE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]{2,}").expect("internal regex must be valid")
    });
    static CAMEL_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\b[a-zA-Z][a-zA-Z0-9]*?[A-Z][a-zA-Z0-9]*\b")
            .expect("internal regex must be valid")
    });

    let mut identifiers = Vec::new();

    for cap in SNAKE_RE.find_iter(query) {
        let s = cap.as_str();
        // Only add if it looks like code (contains underscore or is reasonably long)
        if s.contains('_') || s.len() >= 3 {
            identifiers.push(s.to_string());
        }
    }

    for cap in CAMEL_RE.find_iter(query) {
        let s = cap.as_str();
        if !identifiers.iter().any(|existing| existing == s) {
            identifiers.push(s.to_string());
        }
    }

    identifiers
}

// ============================================================================
// Context Pack Builder
// ============================================================================

/// Build context pack using keyword search + refill
pub fn build_context_pack_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &tokenizers::Tokenizer,
    search_opt: SearchCodeOptions,
    index_opt: IndexChunkOptions,
    refill_opt: core::code_chunk::RefillOptions,
) -> Result<ContextPack> {
    // Determine if this is a natural language query (contains non-ASCII chars or spaces)
    let is_natural_language = query.chars().any(|c| c.is_alphabetic() && !c.is_ascii());

    let search_queries = if is_natural_language || (query.contains(' ') && query.len() > 20) {
        // Query looks like natural language - extract identifiers
        let ids = extract_code_identifiers(query);
        if ids.is_empty() {
            vec![query.to_string()]
        } else {
            ids
        }
    } else {
        vec![query.to_string()]
    };

    let mut all_hits = Vec::new();
    let mut all_trace = Vec::new();

    for search_query in search_queries {
        let (hits, trace) = search_code_keyword(
            repo_root,
            &search_query,
            tokenizer,
            index_opt.clone(),
            search_opt.clone(),
        )?;
        all_hits.extend(hits);
        all_trace.extend(trace);
    }

    // Deduplicate hits
    let mut uniq_hits: BTreeMap<(String, usize, usize), IndexChunk> = BTreeMap::new();
    for h in all_hits {
        let key = (h.path.clone(), h.start_byte, h.end_byte);
        uniq_hits.entry(key).or_insert(h);
    }
    let hits: Vec<_> = uniq_hits.into_values().collect();

    let (context, trace2) = refill_hits(repo_root, &hits, refill_opt)?;
    all_trace.extend(trace2);

    Ok(ContextPack {
        query: query.to_string(),
        hits,
        context,
        trace: all_trace,
    })
}
