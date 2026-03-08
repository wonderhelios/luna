//! ContextCache: Session-level context caching
//!
//! Supports incremental Refill operations by caching retrieved chunks.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{ChunkId, ContextChunk, ContextQuery, IndexChunk, SourceLocation, SymbolId};

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    index_chunk: IndexChunk,
    /// When this was cached
    cached_at: std::time::Instant,
    /// Access count for LRU
    access_count: usize,
}

impl CacheEntry {
    fn new(chunk: IndexChunk) -> Self {
        Self {
            index_chunk: chunk,
            cached_at: std::time::Instant::now(),
            access_count: 1,
        }
    }

    fn touch(&mut self) {
        self.access_count += 1;
    }
}

/// Session-level context cache
///
/// Caches IndexChunks to support incremental Refill operations.
/// When LLM requests "see more code", we can fetch additional chunks
/// without re-scanning the repository.
pub struct ContextCache {
    /// Map from chunk ID to entry
    chunks: Mutex<HashMap<ChunkId, CacheEntry>>,
    /// Map from symbol to chunk IDs
    symbol_index: Mutex<HashMap<SymbolId, Vec<ChunkId>>>,
    /// Map from file path to chunk IDs
    file_index: Mutex<HashMap<SourceLocation, Vec<ChunkId>>>,
    /// Map from query hash to results
    query_cache: Mutex<HashMap<String, Vec<ChunkId>>>,
    /// Maximum cache size
    max_entries: usize,
}

impl ContextCache {
    /// Create a new cache with specified max size
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            chunks: Mutex::new(HashMap::new()),
            symbol_index: Mutex::new(HashMap::new()),
            file_index: Mutex::new(HashMap::new()),
            query_cache: Mutex::new(HashMap::new()),
            max_entries,
        }
    }

    /// Create cache with default size (1000 entries)
    #[must_use]
    pub fn with_default_size() -> Self {
        Self::new(1000)
    }
}

impl ContextCache {
    /// Store an index chunk in cache
    pub fn store(&self, chunk: IndexChunk) {
        let mut chunks = self.chunks.lock().unwrap();

        // Evict if at capacity (simple eviction: clear half)
        if chunks.len() >= self.max_entries {
            self.evict_entries(&mut chunks);
        }

        let chunk_id = chunk.id;

        // Update symbol index
        if !chunk.symbols.is_empty() {
            let mut symbol_index = self.symbol_index.lock().unwrap();
            for symbol in &chunk.symbols {
                symbol_index
                    .entry(symbol.clone())
                    .or_default()
                    .push(chunk_id);
            }
        }

        // Update file index
        let mut file_index = self.file_index.lock().unwrap();
        file_index
            .entry(chunk.source.clone())
            .or_default()
            .push(chunk_id);

        // Store chunk
        chunks.insert(chunk_id, CacheEntry::new(chunk));
    }

    /// Store multiple chunks
    pub fn store_batch(&self, chunks: Vec<IndexChunk>) {
        for chunk in chunks {
            self.store(chunk);
        }
    }

    /// Get a chunk by ID
    #[must_use]
    pub fn get(&self, id: ChunkId) -> Option<IndexChunk> {
        let mut chunks = self.chunks.lock().unwrap();
        chunks.get_mut(&id).map(|entry| {
            entry.touch();
            entry.index_chunk.clone()
        })
    }

    /// Find chunks by symbol
    #[must_use]
    pub fn find_by_symbol(&self, symbol: &SymbolId) -> Vec<IndexChunk> {
        let symbol_index = self.symbol_index.lock().unwrap();
        let chunk_ids = match symbol_index.get(symbol) {
            Some(ids) => ids,
            None => return Vec::new(),
        };

        let mut chunks = self.chunks.lock().unwrap();
        let mut result = Vec::new();

        for id in chunk_ids {
            if let Some(entry) = chunks.get_mut(id) {
                entry.touch();
                result.push(entry.index_chunk.clone());
            }
        }

        result
    }

    /// Find chunks by file path
    #[must_use]
    pub fn find_by_file(&self, source: &SourceLocation) -> Vec<IndexChunk> {
        let file_index = self.file_index.lock().unwrap();
        let chunk_ids = match file_index.get(source) {
            Some(ids) => ids,
            None => return Vec::new(),
        };

        let mut chunks = self.chunks.lock().unwrap();
        let mut result = Vec::new();

        for id in chunk_ids {
            if let Some(entry) = chunks.get_mut(id) {
                entry.touch();
                result.push(entry.index_chunk.clone());
            }
        }

        result
    }

    /// Cache query results
    pub fn cache_query_result(&self, query: &ContextQuery, chunk_ids: Vec<ChunkId>) {
        let query_key = format!("{:?}", query); // Simple serialization
        let mut query_cache = self.query_cache.lock().unwrap();
        query_cache.insert(query_key, chunk_ids);
    }

    /// Get cached query results
    #[must_use]
    pub fn get_cached_query(&self, query: &ContextQuery) -> Option<Vec<IndexChunk>> {
        let query_key = format!("{:?}", query);
        let query_cache = self.query_cache.lock().unwrap();
        let chunk_ids = query_cache.get(&query_key)?;

        let chunks = self.chunks.lock().unwrap();
        let mut result = Vec::new();

        for id in chunk_ids {
            if let Some(entry) = chunks.get(id) {
                result.push(entry.index_chunk.clone());
            }
        }

        if result.len() == chunk_ids.len() {
            Some(result)
        } else {
            None // Some chunks were evicted
        }
    }

    /// Check if a query is cached
    #[must_use]
    pub fn is_query_cached(&self, query: &ContextQuery) -> bool {
        let query_key = format!("{:?}", query);
        let query_cache = self.query_cache.lock().unwrap();
        query_cache.contains_key(&query_key)
    }

    /// Invalidate all entries for a file (when file changes)
    pub fn invalidate_file(&self, source: &SourceLocation) {
        let mut file_index = self.file_index.lock().unwrap();
        if let Some(chunk_ids) = file_index.remove(source) {
            drop(file_index);

            let mut chunks = self.chunks.lock().unwrap();
            let mut symbol_index = self.symbol_index.lock().unwrap();

            for id in chunk_ids {
                if let Some(entry) = chunks.remove(&id) {
                    // Remove from symbol index
                    for symbol in &entry.index_chunk.symbols {
                        if let Some(ids) = symbol_index.get_mut(symbol) {
                            ids.retain(|&x| x != id);
                        }
                    }
                }
            }
        }

        // Clear query cache (conservative)
        let mut query_cache = self.query_cache.lock().unwrap();
        query_cache.clear();
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let chunks = self.chunks.lock().unwrap();
        let symbol_index = self.symbol_index.lock().unwrap();
        let file_index = self.file_index.lock().unwrap();
        let query_cache = self.query_cache.lock().unwrap();

        CacheStats {
            total_chunks: chunks.len(),
            total_symbols: symbol_index.len(),
            total_files: file_index.len(),
            cached_queries: query_cache.len(),
        }
    }

    /// Clear all cached data
    pub fn clear(&self) {
        let mut chunks = self.chunks.lock().unwrap();
        let mut symbol_index = self.symbol_index.lock().unwrap();
        let mut file_index = self.file_index.lock().unwrap();
        let mut query_cache = self.query_cache.lock().unwrap();

        chunks.clear();
        symbol_index.clear();
        file_index.clear();
        query_cache.clear();
    }

    /// Evict entries when cache is full
    fn evict_entries(&self, chunks: &mut HashMap<ChunkId, CacheEntry>) {
        // Simple strategy: remove oldest 50%
        let target_size = self.max_entries / 2;

        if chunks.len() <= target_size {
            return;
        }

        // Collect and sort entries by score
        let mut scored_entries: Vec<(ChunkId, f64)> = chunks
            .iter()
            .map(|(id, entry)| {
                let score = entry.access_count as f64
                    / (entry.cached_at.elapsed().as_secs() as f64 + 1.0);
                (*id, score)
            })
            .collect();

        scored_entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove lowest scored entries
        let to_remove: Vec<ChunkId> = scored_entries
            .into_iter()
            .take(chunks.len() - target_size)
            .map(|(id, _)| id)
            .collect();

        for id in to_remove {
            chunks.remove(&id);
        }
    }
}

impl Default for ContextCache {
    fn default() -> Self {
        Self::with_default_size()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_chunks: usize,
    pub total_symbols: usize,
    pub total_files: usize,
    pub cached_queries: usize,
}

/// Conversion from IndexChunk to ContextChunk (for cached results)
impl ContextCache {
    /// Convert cached IndexChunks to ContextChunks
    #[must_use]
    pub fn to_context_chunks(&self, relevance_score: f32) -> Vec<ContextChunk> {
        let chunks = self.chunks.lock().unwrap();
        chunks
            .values()
            .map(|entry| index_to_context(&entry.index_chunk, relevance_score))
            .collect()
    }
}

fn index_to_context(index: &IndexChunk, relevance: f32) -> ContextChunk {
    use crate::{ContextChunk, ContextType};

    let context_type = match index.chunk_type {
        crate::IndexChunkType::FileSummary => ContextType::FileOverview,
        crate::IndexChunkType::SymbolDefinition => ContextType::NavigationResult,
        crate::IndexChunkType::SymbolReference => ContextType::RelatedSymbol,
        crate::IndexChunkType::CodeBlock => ContextType::CodeSnippet,
        crate::IndexChunkType::Documentation => ContextType::Documentation,
    };

    let mut chunk = ContextChunk::new(
        index.content.clone(),
        index.source.clone(),
        context_type,
    );
    chunk.set_relevance(relevance);

    // Add symbol signatures if available
    for symbol in &index.symbols {
        chunk.add_signature(symbol.full_name());
    }

    chunk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexChunkType, LanguageId, SourceLocation, SymbolId, TextRange};
    use std::path::PathBuf;

    fn test_chunk(name: &str) -> IndexChunk {
        IndexChunk::new(
            format!("fn {}() {{}}", name),
            SourceLocation {
                repo_root: PathBuf::from("/repo"),
                rel_path: PathBuf::from("src/lib.rs"),
                range: TextRange::new(1, 5),
            },
            IndexChunkType::SymbolDefinition,
        )
    }

    fn test_chunk_with_symbol(name: &str, symbol: SymbolId) -> IndexChunk {
        let mut chunk = test_chunk(name);
        chunk.symbols.push(symbol);
        chunk
    }

    #[test]
    fn test_store_and_get() {
        let cache = ContextCache::new(100);
        let chunk = test_chunk("foo");
        let id = chunk.id;

        cache.store(chunk);
        let retrieved = cache.get(id);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "fn foo() {}");
    }

    #[test]
    fn test_find_by_symbol() {
        let cache = ContextCache::new(100);
        let symbol = SymbolId::new("foo", "crate");
        let chunk = test_chunk_with_symbol("foo", symbol.clone());

        cache.store(chunk);
        let results = cache.find_by_symbol(&symbol);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "fn foo() {}");
    }

    #[test]
    fn test_query_caching() {
        let cache = ContextCache::new(100);
        let query = ContextQuery::symbol("foo");
        let chunk = test_chunk("foo");
        let id = chunk.id;

        cache.store(chunk);
        cache.cache_query_result(&query, vec![id]);

        assert!(cache.is_query_cached(&query));

        let results = cache.get_cached_query(&query);
        assert!(results.is_some());
        assert_eq!(results.unwrap().len(), 1);
    }

    #[test]
    fn test_invalidate_file() {
        let cache = ContextCache::new(100);
        let source = SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(1, 5),
        };
        let chunk = IndexChunk::new("content", source.clone(), IndexChunkType::CodeBlock);

        cache.store(chunk);
        assert_eq!(cache.stats().total_chunks, 1);

        cache.invalidate_file(&source);
        assert_eq!(cache.stats().total_chunks, 0);
    }

    #[test]
    fn test_stats() {
        let cache = ContextCache::new(100);
        let chunk = test_chunk_with_symbol("foo", SymbolId::new("foo", "crate"));

        cache.store(chunk);
        let stats = cache.stats();

        assert_eq!(stats.total_chunks, 1);
        assert_eq!(stats.total_symbols, 1);
        assert_eq!(stats.total_files, 1);
    }
}
