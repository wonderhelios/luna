//! Semantic Retriever - Natural Language Code Search
//!
//! This module provides semantic code search capabilities by combining:
//! - Embedding models (local or remote)
//! - Vector indexing (HNSW)
//! - Smart code chunking
//! - Integration with existing Context Pipeline

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    chunking::ChunkType,
    index::{HnswIndex, VectorEntry, VectorId, VectorIndex},
    ChunkingStrategy, CodeChunk, CodeChunker, DistanceMetric, EmbeddingModel,
};
use crate::{ContextChunk, ContextType};

/// Options for semantic search
#[derive(Debug, Clone)]
pub struct SemanticSearchOptions {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum similarity score (0-1) for results
    pub min_score: f32,
    /// Whether to include file summaries
    pub include_summaries: bool,
    /// Whether to expand results with nearby chunks
    pub expand_context: bool,
}

impl Default for SemanticSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.5,
            include_summaries: true,
            expand_context: true,
        }
    }
}

/// Semantic retriever - main entry point for vector-based search
pub struct SemanticRetriever {
    /// Embedding model for encoding queries and documents
    embedding: Arc<dyn EmbeddingModel>,
    /// Vector index for efficient search
    index: HnswIndex,
    /// Code chunker for document processing
    chunker: CodeChunker,
    /// Mapping from VectorId to CodeChunk
    chunks: HashMap<VectorId, CodeChunk>,
    /// Repository root
    repo_root: PathBuf,
}

impl SemanticRetriever {
    /// Create a new semantic retriever
    pub fn new(
        embedding: Arc<dyn EmbeddingModel>,
        repo_root: PathBuf,
    ) -> Self {
        let dim = embedding.embedding_dim();
        Self {
            embedding,
            index: HnswIndex::default_with_dim(dim),
            chunker: CodeChunker::default_config(),
            chunks: HashMap::new(),
            repo_root,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        embedding: Arc<dyn EmbeddingModel>,
        repo_root: PathBuf,
        chunking_strategy: ChunkingStrategy,
        max_tokens: usize,
    ) -> Self {
        let dim = embedding.embedding_dim();
        Self {
            embedding,
            index: HnswIndex::default_with_dim(dim),
            chunker: CodeChunker::new(chunking_strategy, max_tokens, max_tokens / 8),
            chunks: HashMap::new(),
            repo_root,
        }
    }

    /// Index a single file
    pub fn index_file(&mut self, path: &Path) -> error::Result<usize> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| error::LunaError::io(Some(path.to_path_buf()), e))?;

        self.index_file_content(path, &content)
    }

    /// Index file content (for testing)
    pub fn index_file_content(&mut self, path: &Path, content: &str) -> error::Result<usize> {
        let chunks = self.chunker.chunk_file(content, path, &self.repo_root);
        let mut indexed = 0;

        for chunk in chunks {
            let vector_id = VectorId::new();
            let text = chunk.embedding_text();

            // Generate embedding
            match self.embedding.encode(&text) {
                Ok(embedding) => {
                    let entry = VectorEntry {
                        id: vector_id,
                        vector: embedding.vector,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("chunk_id".to_string(), chunk.id.clone());
                            m.insert("file".to_string(), chunk.source.rel_path.display().to_string());
                            m
                        },
                    };

                    self.index.insert(entry)?;
                    self.chunks.insert(vector_id, chunk);
                    indexed += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to encode chunk {}: {}", chunk.id, e);
                }
            }
        }

        tracing::info!("Indexed {} chunks from {}", indexed, path.display());
        Ok(indexed)
    }

    /// Index multiple files
    pub fn index_files(&mut self, paths: &[PathBuf]) -> error::Result<IndexStats> {
        let mut stats = IndexStats::default();

        for path in paths {
            match self.index_file(path) {
                Ok(count) => {
                    stats.files_indexed += 1;
                    stats.chunks_indexed += count;
                }
                Err(e) => {
                    tracing::warn!("Failed to index {}: {}", path.display(), e);
                    stats.files_failed += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Search for code using natural language query
    pub fn search(
        &self,
        query: &str,
        options: &SemanticSearchOptions,
    ) -> error::Result<Vec<SemanticResult>> {
        // Encode query
        let query_embedding = self.embedding.encode(query)?;

        // Search index
        let results = self.index.search(&query_embedding.vector, options.top_k * 2)?;

        // Convert to semantic results
        let mut semantic_results: Vec<SemanticResult> = results
            .into_iter()
            .filter(|r| r.score >= options.min_score)
            .filter_map(|r| {
                self.chunks.get(&r.id).map(|chunk| SemanticResult {
                    chunk: chunk.clone(),
                    score: r.score,
                    distance: r.distance,
                })
            })
            .collect();

        // Sort by score (descending)
        semantic_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        semantic_results.truncate(options.top_k);

        // Expand context if requested
        if options.expand_context {
            semantic_results = self.expand_results(semantic_results);
        }

        Ok(semantic_results)
    }

    /// Search and convert to ContextChunks for integration with RefillPipeline
    pub fn search_as_context(
        &self,
        query: &str,
        options: &SemanticSearchOptions,
    ) -> error::Result<Vec<ContextChunk>> {
        let results = self.search(query, options)?;

        Ok(results
            .into_iter()
            .map(|r| self.semantic_result_to_context(r))
            .collect())
    }

    /// Expand results with nearby chunks for better context
    fn expand_results(&self, results: Vec<SemanticResult>) -> Vec<SemanticResult> {
        // For now, just return as-is
        // Future: could fetch adjacent chunks from the same file
        results
    }

    /// Convert SemanticResult to ContextChunk
    fn semantic_result_to_context(&self, result: SemanticResult) -> ContextChunk {
        let chunk = result.chunk;

        let context_type = match chunk.chunk_type {
            ChunkType::Function { .. } => ContextType::NavigationResult,
            ChunkType::Struct { .. } => ContextType::RelatedSymbol,
            ChunkType::Trait { .. } => ContextType::RelatedSymbol,
            ChunkType::Impl { .. } => ContextType::CodeSnippet,
            ChunkType::Module { .. } => ContextType::FileOverview,
            ChunkType::Comment => ContextType::Documentation,
            ChunkType::Import => ContextType::FileOverview,
            ChunkType::FileSummary => ContextType::FileOverview,
            ChunkType::CodeBlock => ContextType::CodeSnippet,
        };

        // Build content with metadata
        let content = format!(
            "// File: {} (score: {:.2})\n{}\n// End of chunk",
            chunk.source.rel_path.display(),
            result.score,
            chunk.content
        );

        ContextChunk::new(content, chunk.source, context_type)
            .with_relevance(result.score)
    }

    /// Get the number of indexed chunks
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clear all indexed data
    pub fn clear(&mut self) {
        self.index.clear();
        self.chunks.clear();
    }

    /// Get repository root
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }
}

/// Result of a semantic search
#[derive(Debug, Clone)]
pub struct SemanticResult {
    /// The matched code chunk
    pub chunk: CodeChunk,
    /// Similarity score (0-1, higher is better)
    pub score: f32,
    /// Raw distance metric
    pub distance: f32,
}

/// Statistics from indexing operation
#[derive(Debug, Default)]
pub struct IndexStats {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub files_failed: usize,
}

/// Builder for SemanticRetriever with fluent API
pub struct SemanticRetrieverBuilder {
    embedding: Option<Arc<dyn EmbeddingModel>>,
    repo_root: Option<PathBuf>,
    chunking_strategy: ChunkingStrategy,
    max_chunk_tokens: usize,
    distance_metric: DistanceMetric,
}

impl Default for SemanticRetrieverBuilder {
    fn default() -> Self {
        Self {
            embedding: None,
            repo_root: None,
            chunking_strategy: ChunkingStrategy::Semantic,
            max_chunk_tokens: 256,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

impl SemanticRetrieverBuilder {
    /// Set embedding model
    pub fn embedding(mut self, embedding: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set repository root
    pub fn repo_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.repo_root = Some(path.into());
        self
    }

    /// Set chunking strategy
    pub fn chunking_strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.chunking_strategy = strategy;
        self
    }

    /// Set max chunk tokens
    pub fn max_chunk_tokens(mut self, tokens: usize) -> Self {
        self.max_chunk_tokens = tokens;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Build the retriever
    pub fn build(self) -> error::Result<SemanticRetriever> {
        let embedding = self.embedding
            .ok_or_else(|| error::LunaError::invalid_input("Embedding model is required"))?;

        let repo_root = self.repo_root
            .ok_or_else(|| error::LunaError::invalid_input("Repository root is required"))?;

        Ok(SemanticRetriever::with_config(
            embedding,
            repo_root,
            self.chunking_strategy,
            self.max_chunk_tokens,
        ))
    }
}

/// Extension trait to integrate with RefillPipeline
///
/// This trait can be implemented by RefillPipeline to add semantic search
/// as a fallback or complementary retrieval method.
pub trait SemanticSearchExtension {
    /// Search with both symbol-based and semantic methods
    fn search_hybrid(
        &self,
        query: &str,
        semantic_retriever: &SemanticRetriever,
        options: &SemanticSearchOptions,
    ) -> error::Result<Vec<ContextChunk>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::embedding::MockEmbedding;
    use crate::SourceLocation;
    use std::path::PathBuf;

    fn create_test_retriever() -> SemanticRetriever {
        let embedding = Arc::new(MockEmbedding::new(128));
        let temp_dir = tempfile::tempdir().unwrap();

        SemanticRetriever::new(embedding, temp_dir.path().to_path_buf())
    }

    #[test]
    fn test_index_and_search() {
        let mut retriever = create_test_retriever();

        // Index a file
        let code = r#"
pub fn hello() -> &'static str {
    "Hello, world!"
}

pub fn goodbye() -> &'static str {
    "Goodbye!"
}
"#;

        let indexed = retriever
            .index_file_content(Path::new("/repo/src/lib.rs"), code)
            .unwrap();

        assert!(indexed > 0);

        // Search
        let options = SemanticSearchOptions::default();
        let results = retriever.search("say hello", &options).unwrap();

        assert!(!results.is_empty());
        // Mock embedding is deterministic, so we can check results
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_semantic_result_to_context() {
        let chunk = CodeChunk::new(
            "test",
            "fn main() { println!(\"hello\"); }",
            SourceLocation {
                repo_root: PathBuf::from("/repo"),
                rel_path: PathBuf::from("src/main.rs"),
                range: crate::TextRange::new(1, 1),
            },
            crate::LanguageId::Rust,
            super::ChunkType::Function {
                name: "main".to_string(),
                signature: "fn main()".to_string(),
                is_async: false,
                is_unsafe: false,
            },
        );

        let result = SemanticResult {
            chunk,
            score: 0.85,
            distance: 0.15,
        };

        let retriever = create_test_retriever();
        let context = retriever.semantic_result_to_context(result);

        assert!(matches!(context.context_type, ContextType::NavigationResult));
        assert!(context.content.contains("score: 0.85"));
        assert!(context.content.contains("fn main()"));
    }

    #[test]
    fn test_builder() {
        let embedding = Arc::new(MockEmbedding::new(128));
        let temp_dir = tempfile::tempdir().unwrap();

        let retriever = SemanticRetrieverBuilder::default()
            .embedding(embedding)
            .repo_root(temp_dir.path())
            .chunking_strategy(ChunkingStrategy::FixedSize)
            .max_chunk_tokens(128)
            .distance_metric(DistanceMetric::Euclidean)
            .build()
            .unwrap();

        assert!(retriever.is_empty());
    }
}
