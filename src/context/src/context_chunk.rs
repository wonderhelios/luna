//! ContextChunk: Generation-phase refined chunks
//!
//! Optimized for LLM consumption - token-efficient, high relevance.

use serde::{Deserialize, Serialize};

use crate::{ChunkId, SourceLocation};

/// Context type - what role this chunk plays in the prompt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextType {
    /// Navigation result (goto definition)
    NavigationResult,
    /// Code snippet (extracted function body, etc.)
    CodeSnippet,
    /// File overview (module structure)
    FileOverview,
    /// Dependency graph (callers/callees)
    DependencyGraph,
    /// Related symbol (contextually relevant)
    RelatedSymbol,
    /// Documentation
    Documentation,
}

/// ContextChunk: Refined content ready for LLM consumption
///
/// This is the "fine" representation - token-optimized, high relevance,
/// with symbol signatures injected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    pub id: ChunkId,
    /// Refined content (may be truncated/optimized)
    pub content: String,
    /// Source location
    pub source: SourceLocation,
    /// Combined relevance score (0.0 - 1.0)
    pub relevance_score: f32,
    /// Actual token count
    pub token_count: usize,
    /// Symbol signatures injected into this chunk
    pub symbol_signatures: Vec<String>,
    /// Type of context
    pub context_type: ContextType,
}

impl ContextChunk {
    /// Create a new context chunk
    #[must_use]
    pub fn new(
        content: impl Into<String>,
        source: SourceLocation,
        context_type: ContextType,
    ) -> Self {
        let content = content.into();
        let token_count = crate::TokenBudget::estimate_tokens(&content);

        Self {
            id: ChunkId::new(),
            content,
            source,
            relevance_score: 0.0,
            token_count,
            symbol_signatures: Vec::new(),
            context_type,
        }
    }

    /// Create a navigation result chunk
    #[must_use]
    pub fn navigation_result(
        content: impl Into<String>,
        source: SourceLocation,
        symbol_signature: impl Into<String>,
    ) -> Self {
        let mut chunk = Self::new(content, source, ContextType::NavigationResult);
        chunk.symbol_signatures.push(symbol_signature.into());
        chunk.relevance_score = 1.0; // Navigation results are highly relevant
        chunk
    }

    /// Create a code snippet chunk
    #[must_use]
    pub fn code_snippet(
        content: impl Into<String>,
        source: SourceLocation,
        relevance_score: f32,
    ) -> Self {
        let mut chunk = Self::new(content, source, ContextType::CodeSnippet);
        chunk.relevance_score = relevance_score.clamp(0.0, 1.0);
        chunk
    }

    /// Add a symbol signature
    pub fn add_signature(&mut self, signature: impl Into<String>) {
        self.symbol_signatures.push(signature.into());
    }

    /// Set relevance score
    pub fn set_relevance(&mut self, score: f32) {
        self.relevance_score = score.clamp(0.0, 1.0);
    }

    /// Set relevance score (builder style)
    #[must_use]
    pub fn with_relevance(mut self, score: f32) -> Self {
        self.relevance_score = score.clamp(0.0, 1.0);
        self
    }

    /// Format this chunk for inclusion in a prompt
    #[must_use]
    pub fn format_for_prompt(&self) -> String {
        let mut output = String::new();

        // Header with location
        output.push_str(&format!(
            "// {}:{}-{}",
            self.source.rel_path.display(),
            self.source.range.start_line,
            self.source.range.end_line
        ));

        // Add signatures if present
        if !self.symbol_signatures.is_empty() {
            output.push_str(" (");
            for (i, sig) in self.symbol_signatures.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(sig);
            }
            output.push(')');
        }
        output.push('\n');

        // Content
        output.push_str(&self.content);
        output.push('\n');

        output
    }

    /// Truncate content to fit within token budget
    pub fn truncate_to_tokens(&mut self, max_tokens: usize) {
        if self.token_count <= max_tokens {
            return;
        }

        // Approximate: 4 chars per token
        let max_chars = max_tokens * 4;
        if self.content.len() > max_chars {
            let truncated = &self.content[..max_chars];
            // Try to truncate at a line boundary
            if let Some(last_newline) = truncated.rfind('\n') {
                self.content = truncated[..last_newline].to_string();
            } else {
                self.content = truncated.to_string();
            }
            self.token_count = crate::TokenBudget::estimate_tokens(&self.content);
        }
    }
}

/// Builder for constructing context chunks from index chunks
pub struct ContextChunkBuilder {
    content: String,
    source: SourceLocation,
    context_type: ContextType,
    signatures: Vec<String>,
    relevance_score: f32,
}

impl ContextChunkBuilder {
    #[must_use]
    pub fn new(source: SourceLocation, context_type: ContextType) -> Self {
        Self {
            content: String::new(),
            source,
            context_type,
            signatures: Vec::new(),
            relevance_score: 0.0,
        }
    }

    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    pub fn signature(mut self, signature: impl Into<String>) -> Self {
        self.signatures.push(signature.into());
        self
    }

    pub fn relevance(mut self, score: f32) -> Self {
        self.relevance_score = score;
        self
    }

    #[must_use]
    pub fn build(self) -> ContextChunk {
        let mut chunk = ContextChunk::new(self.content, self.source, self.context_type);
        chunk.symbol_signatures = self.signatures;
        chunk.relevance_score = self.relevance_score.clamp(0.0, 1.0);
        chunk
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
            range: TextRange::new(10, 15),
        }
    }

    #[test]
    fn test_context_chunk_creation() {
        let chunk = ContextChunk::new("fn foo() {}", test_source(), ContextType::CodeSnippet);

        assert_eq!(chunk.content, "fn foo() {}");
        assert_eq!(chunk.context_type, ContextType::CodeSnippet);
        assert_eq!(chunk.token_count, 3); // ~11 chars / 4
    }

    #[test]
    fn test_navigation_result_chunk() {
        let chunk = ContextChunk::navigation_result(
            "pub fn find_main() -> Vec<Location>",
            test_source(),
            "fn find_main() -> Vec<Location>",
        );

        assert_eq!(chunk.context_type, ContextType::NavigationResult);
        assert_eq!(chunk.relevance_score, 1.0);
        assert_eq!(chunk.symbol_signatures.len(), 1);
    }

    #[test]
    fn test_format_for_prompt() {
        let chunk = ContextChunk::navigation_result(
            "pub fn foo() {}",
            test_source(),
            "pub fn foo()",
        );

        let formatted = chunk.format_for_prompt();
        assert!(formatted.contains("src/lib.rs:10-15"));
        assert!(formatted.contains("pub fn foo()"));
        assert!(formatted.contains("pub fn foo() {}"));
    }

    #[test]
    fn test_truncate_to_tokens() {
        let mut chunk = ContextChunk::new(
            "line1\nline2\nline3\nline4\nline5",
            test_source(),
            ContextType::CodeSnippet,
        );

        // Truncate to 2 tokens (~8 chars)
        chunk.truncate_to_tokens(2);
        assert!(chunk.content.len() <= 8);
        assert!(!chunk.content.contains("line3"));
    }

    #[test]
    fn test_builder() {
        let chunk = ContextChunkBuilder::new(test_source(), ContextType::CodeSnippet)
            .content("fn bar() {}")
            .signature("fn bar()")
            .relevance(0.85)
            .build();

        assert_eq!(chunk.content, "fn bar() {}");
        assert_eq!(chunk.symbol_signatures, vec!["fn bar()".to_string()]);
        assert!((chunk.relevance_score - 0.85).abs() < 0.01);
    }
}
