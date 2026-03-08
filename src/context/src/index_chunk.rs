//! IndexChunk: Retrieval-phase code chunks
//!
//! Used for fast recall with high coverage. Can be coarse-grained.

use serde::{Deserialize, Serialize};

use crate::{ChunkId, LanguageId, SourceLocation, SymbolId, TimestampMs};

/// Index chunk type - categorizes what kind of content this chunk represents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexChunkType {
    /// File-level summary (imports, module doc, etc.)
    FileSummary,
    /// Symbol definition (function, struct, etc.)
    SymbolDefinition,
    /// Symbol reference (call site, usage)
    SymbolReference,
    /// Code block (function body, impl block)
    CodeBlock,
    /// Documentation comment
    Documentation,
}

/// IndexChunk: Raw content from repository for retrieval phase
///
/// This is the "coarse" representation - may contain extra content
/// but has high recall for relevant information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexChunk {
    pub id: ChunkId,
    /// Raw content (may be longer than final context)
    pub content: String,
    /// Source location in repository
    pub source: SourceLocation,
    /// Vector embedding for semantic search (optional, Phase 4.2)
    pub embedding: Option<Vec<f32>>,
    /// Associated symbols (from ScopeGraph)
    pub symbols: Vec<SymbolId>,
    /// Programming language
    pub language: LanguageId,
    /// Last modified timestamp
    pub modified_at: TimestampMs,
    /// Type of chunk
    pub chunk_type: IndexChunkType,
}

impl IndexChunk {
    /// Create a new index chunk
    #[must_use]
    pub fn new(
        content: impl Into<String>,
        source: SourceLocation,
        chunk_type: IndexChunkType,
    ) -> Self {
        Self {
            id: ChunkId::new(),
            content: content.into(),
            source,
            embedding: None,
            symbols: Vec::new(),
            language: LanguageId::Unknown,
            modified_at: 0,
            chunk_type,
        }
    }

    /// Create a symbol definition chunk
    #[must_use]
    pub fn symbol_definition(
        content: impl Into<String>,
        source: SourceLocation,
        symbol: SymbolId,
    ) -> Self {
        let mut chunk = Self::new(content, source, IndexChunkType::SymbolDefinition);
        chunk.symbols.push(symbol);
        chunk
    }

    /// Create a file summary chunk
    #[must_use]
    pub fn file_summary(
        content: impl Into<String>,
        source: SourceLocation,
        language: LanguageId,
    ) -> Self {
        let mut chunk = Self::new(content, source, IndexChunkType::FileSummary);
        chunk.language = language;
        chunk
    }

    /// Estimate token count for this chunk
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        crate::TokenBudget::estimate_tokens(&self.content)
    }

    /// Check if this chunk contains a specific symbol
    #[must_use]
    pub fn contains_symbol(&self, symbol: &SymbolId) -> bool {
        self.symbols.contains(symbol)
    }

    /// Get the primary symbol (first one) if any
    #[must_use]
    pub fn primary_symbol(&self) -> Option<&SymbolId> {
        self.symbols.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextRange;
    use std::path::PathBuf;

    fn test_source() -> SourceLocation {
        SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(1, 10),
        }
    }

    #[test]
    fn test_index_chunk_creation() {
        let chunk = IndexChunk::new(
            "fn main() {}",
            test_source(),
            IndexChunkType::SymbolDefinition,
        );

        assert_eq!(chunk.content, "fn main() {}");
        assert_eq!(chunk.chunk_type, IndexChunkType::SymbolDefinition);
        assert!(chunk.symbols.is_empty());
    }

    #[test]
    fn test_symbol_definition_chunk() {
        let symbol = SymbolId::new("main", "crate");
        let chunk = IndexChunk::symbol_definition("fn main() {}", test_source(), symbol.clone());

        assert_eq!(chunk.chunk_type, IndexChunkType::SymbolDefinition);
        assert_eq!(chunk.symbols.len(), 1);
        assert_eq!(chunk.symbols[0], symbol);
        assert!(chunk.contains_symbol(&symbol));
    }

    #[test]
    fn test_file_summary_chunk() {
        let chunk = IndexChunk::file_summary(
            "//! Module doc\nuse std::io;",
            test_source(),
            LanguageId::Rust,
        );

        assert_eq!(chunk.chunk_type, IndexChunkType::FileSummary);
        assert_eq!(chunk.language, LanguageId::Rust);
    }

    #[test]
    fn test_token_estimation() {
        let chunk = IndexChunk::new("abcd".repeat(100), test_source(), IndexChunkType::CodeBlock);
        // 400 chars / 4 = 100 tokens
        assert_eq!(chunk.estimated_tokens(), 100);
    }
}
