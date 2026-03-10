//! Vector-based Semantic Retrieval for Context Pipeline
//!
//! This module provides semantic code search capabilities using vector embeddings.
//! It enables natural language queries to find relevant code without exact symbol matches.
//!
//! ## Architecture
//!
//! ```text
//! User Query: "how does error handling work?"
//!     ↓
//! EmbeddingModel::encode("how does error handling work?")
//!     ↓
//! VectorIndex::search(embedding, top_k=10)
//!     ↓
//! Vec<CodeChunk> with similarity scores
//!     ↓
//! RefillPipeline::refine(chunks)
//!     ↓
//! ContextChunks for LLM prompt
//! ```
//!
//! ## Features
//!
//! - **Multiple Embedding Providers**: Ollama (local), OpenAI (remote)
//! - **Efficient Vector Index**: HNSW-based approximate nearest neighbor search
//! - **Smart Code Chunking**: Semantic-aware code splitting
//! - **Incremental Updates**: Add/remove chunks without full reindex

mod chunking;
mod embedding;
mod index;
mod semantic;

pub use chunking::{ChunkingStrategy, CodeChunk, CodeChunker};
pub use embedding::{Embedding, EmbeddingModel, MockEmbedding, OllamaEmbedding, OpenAIEmbedding};
pub use index::{HnswIndex, SearchResult, VectorIndex};
pub use semantic::{SemanticRetriever, SemanticRetrieverBuilder, SemanticSearchOptions};

use std::path::PathBuf;

/// Configuration for vector retrieval
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Embedding model to use
    pub embedding: EmbeddingConfig,
    /// Index configuration
    pub index: IndexConfig,
    /// Chunking strategy
    pub chunking: ChunkingConfig,
    /// Cache directory for persistence
    pub cache_dir: Option<PathBuf>,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig::default(),
            index: IndexConfig::default(),
            chunking: ChunkingConfig::default(),
            cache_dir: None,
        }
    }
}

/// Embedding provider configuration
#[derive(Debug, Clone)]
pub enum EmbeddingConfig {
    /// Ollama local embedding model
    Ollama {
        base_url: String,
        model: String,
        embedding_dim: usize,
    },
    /// OpenAI embedding API
    OpenAI {
        api_key: String,
        model: String,
        embedding_dim: usize,
    },
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            embedding_dim: 768,
        }
    }
}

/// Vector index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Maximum number of connections per node (HNSW M parameter)
    pub max_connections: usize,
    /// Expansion factor during construction
    pub ef_construction: usize,
    /// Expansion factor during search
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Distance metric for vector comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (normalized dot product)
    #[default]
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    /// Lower values = more similar
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                let dot = dot_product(a, b);
                let norm_a = dot_product(a, a).sqrt();
                let norm_b = dot_product(b, b).sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    return 1.0;
                }
                1.0 - (dot / (norm_a * norm_b))
            }
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::DotProduct => {
                let dot = dot_product(a, b);
                -dot // Negative because we want lower = more similar
            }
        }
    }
}

/// Code chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Maximum chunk size in tokens (approximate)
    pub max_chunk_tokens: usize,
    /// Overlap between chunks in tokens
    pub overlap_tokens: usize,
    /// Strategy for splitting code
    pub strategy: ChunkingStrategy,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_tokens: 256,
            overlap_tokens: 32,
            strategy: ChunkingStrategy::Semantic,
        }
    }
}

/// Calculate dot product of two vectors
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        let metric = DistanceMetric::Cosine;

        // Same vector = 0 distance
        assert!((metric.distance(&a, &a) - 0.0).abs() < 0.001);

        // Orthogonal vectors = 1 distance
        assert!((metric.distance(&a, &c) - 1.0).abs() < 0.001);

        // Same direction = 0 distance
        assert!((metric.distance(&a, &b) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        let metric = DistanceMetric::Euclidean;
        let dist = metric.distance(&a, &b);

        // 3-4-5 triangle
        assert!((dist - 5.0).abs() < 0.001);
    }
}
