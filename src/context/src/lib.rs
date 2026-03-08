//! Context Pipeline: IndexChunk → RefillPipeline → ContextChunk
//!
//! Provides context management for the TPAR runtime:
//! - Retrieve relevant code context from repository
//! - Refine and optimize context for LLM consumption
//! - Cache and manage token budgets

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub mod cache;
pub mod context_chunk;
pub mod index_chunk;
pub mod query;
pub mod refill;

pub use cache::ContextCache;
pub use context_chunk::{ContextChunk, ContextType};
pub use index_chunk::{IndexChunk, IndexChunkType};
pub use query::ContextQuery;
pub use refill::RefillPipeline;

/// Unique identifier for chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub uuid::Uuid);

impl ChunkId {
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ChunkId {
    fn default() -> Self {
        Self::new()
    }
}

/// Source location for a chunk
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub repo_root: PathBuf,
    pub rel_path: PathBuf,
    pub range: TextRange,
}

impl SourceLocation {
    #[must_use]
    pub fn abs_path(&self) -> PathBuf {
        self.repo_root.join(&self.rel_path)
    }
}

/// Text range (line/column based)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextRange {
    pub start_line: usize,
    pub end_line: usize,
    pub start_col: usize,
    pub end_col: usize,
}

impl TextRange {
    #[must_use]
    pub fn new(start_line: usize, end_line: usize) -> Self {
        Self {
            start_line,
            end_line,
            start_col: 0,
            end_col: 0,
        }
    }

    #[must_use]
    pub fn with_cols(
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
    ) -> Self {
        Self {
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    #[must_use]
    pub fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// Language identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanguageId {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Ruby,
    Php,
    R,
    Proto,
    Unknown,
}

impl LanguageId {
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" => Self::Python,
            "js" => Self::JavaScript,
            "ts" => Self::TypeScript,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "hpp" => Self::Cpp,
            "cs" => Self::CSharp,
            "rb" => Self::Ruby,
            "php" => Self::Php,
            "r" => Self::R,
            "proto" => Self::Proto,
            _ => Self::Unknown,
        }
    }
}

/// Symbol identifier (wrapper around intelligence namespace)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolId {
    pub name: String,
    pub namespace: String, // e.g., "crate::module::function"
}

impl SymbolId {
    #[must_use]
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
        }
    }

    #[must_use]
    pub fn full_name(&self) -> String {
        if self.namespace.is_empty() {
            self.name.clone()
        } else {
            format!("{}::{}", self.namespace, self.name)
        }
    }
}

/// Timestamp in milliseconds
pub type TimestampMs = u64;

/// Token budget for context management
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    pub max_context_tokens: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_context_tokens: 4000,
        }
    }
}

impl TokenBudget {
    /// Rough token estimation: ~4 chars per token
    #[must_use]
    pub fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(LanguageId::from_extension("rs"), LanguageId::Rust);
        assert_eq!(LanguageId::from_extension("py"), LanguageId::Python);
        assert_eq!(LanguageId::from_extension("JS"), LanguageId::JavaScript);
        assert_eq!(LanguageId::from_extension("unknown"), LanguageId::Unknown);
    }

    #[test]
    fn test_text_range_line_count() {
        let range = TextRange::new(10, 15);
        assert_eq!(range.line_count(), 6);
    }

    #[test]
    fn test_symbol_id_full_name() {
        let sym = SymbolId::new("foo", "crate::bar");
        assert_eq!(sym.full_name(), "crate::bar::foo");

        let sym2 = SymbolId::new("foo", "");
        assert_eq!(sym2.full_name(), "foo");
    }

    #[test]
    fn test_token_estimation() {
        // 4 chars ~= 1 token
        assert_eq!(TokenBudget::estimate_tokens("abcd"), 1);
        assert_eq!(TokenBudget::estimate_tokens("abcdefghij"), 3); // 10 / 4 = 2.5 -> 3
    }
}
