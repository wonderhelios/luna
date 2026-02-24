//! Index Module
//!
//! This crate provides code indexing capabilities:
//! - Semantic chunking based on AST/scope graph
//! - IndexChunk generation for vector/keyword retrieval
//! - Refill pipeline to expand hits into ContextChunks
//!
//! Design Philosophy:
//! - IndexChunk: For retrieval (token-aware, normalized size)
//! - ContextChunk: For LLM consumption (semantically complete)
//! - Two-stage pipeline: Search IndexChunk → Refill → ContextChunk

pub mod chunk;
pub mod error;

pub use chunk::{
    chunk_source,
    index_chunks,
    refill_chunks,
};
pub use error::ChunkError;

// Re-export from core for convenience
pub use core::code_chunk::{
    ChunkOptions, CodeChunk, ContextChunk, IndexChunk, IndexChunkBuildError,
    IndexChunkOptions, RefillOptions,
};
