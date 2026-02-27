//! Code search operations for agents

mod backend;
mod keyword;
mod options;
mod refill;
mod symbol;

pub use backend::{KeywordSearchBackend, SearchBackend};
pub use options::SearchCodeOptions;
pub use refill::refill_hits;
pub use symbol::{find_symbol_definitions, SymbolLocation};

use crate::{Result, ToolTrace};
use core::code_chunk::{IndexChunk, IndexChunkOptions};
use tokenizers::Tokenizer;

/// Keyword placeholder for search_code: scan repo files, normalize hits using IndexChunk protocol
///
/// Returns: IndexChunk hits (each chunk's text contains query)
pub fn search_code_keyword(
    repo_root: &std::path::Path,
    query: &str,
    tokenizer: &Tokenizer,
    idx_opt: IndexChunkOptions,
    opt: SearchCodeOptions,
) -> Result<(Vec<IndexChunk>, Vec<ToolTrace>)> {
    KeywordSearchBackend::default().search(repo_root, query, tokenizer, idx_opt, opt)
}
