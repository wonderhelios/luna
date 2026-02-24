//! Simple in-memory caching layer for Luna
//!
//! This module provides a simple LRU cache for storing:
//! - ScopeGraph parsing results
//! - Tokenization results
//! - Search results
//!
//! Design Principles:
//! - Simple: No external cache dependencies
//! - Fast: In-memory storage
//! - Safe: Automatic size limiting

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

// ============================================================================
// Cache Entry
// ============================================================================

/// A cached value with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value
    pub value: V,
    /// When this entry was created (Unix timestamp)
    pub created_at: u64,
    /// Size of the entry in bytes (approximate)
    pub size_bytes: usize,
}

impl<V> CacheEntry<V> {
    /// Create a new cache entry
    pub fn new(value: V, size_bytes: usize) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            value,
            created_at,
            size_bytes,
        }
    }

    /// Check if the entry is older than the given age (seconds)
    pub fn is_older_than(&self, max_age_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now.saturating_sub(self.created_at) > max_age_secs
    }
}

// ============================================================================
// Simple LRU Cache
// ============================================================================

/// A simple LRU cache with size limiting
///
/// This cache evicts the oldest entries when the size limit is reached.
#[derive(Debug)]
pub struct LruCache<K, V> {
    entries: HashMap<K, CacheEntry<V>>,
    /// Maximum total size in bytes
    max_bytes: usize,
    /// Current total size in bytes
    current_bytes: usize,
    /// Access order for LRU eviction (most recent at end)
    access_order: Vec<K>,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    /// Create a new LRU cache with the given size limit
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            access_order: Vec::new(),
        }
    }

    /// Insert a value into the cache
    ///
    /// Returns the evicted entry if any
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize) -> Option<(K, V)> {
        // Update access order
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key.clone());

        // Check if key already exists
        if let Some(old_entry) = self.entries.get(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old_entry.size_bytes);
        }

        let entry = CacheEntry::new(value, size_bytes);
        self.current_bytes += size_bytes;

        // Evict if over capacity
        while self.current_bytes > self.max_bytes && !self.access_order.is_empty() {
            let old_key = self.access_order.remove(0);
            if let Some(old_entry) = self.entries.remove(&old_key) {
                self.current_bytes = self.current_bytes.saturating_sub(old_entry.size_bytes);
            }
        }

        // Insert the new entry
        self.entries.insert(key.clone(), entry);
        None
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(_) = self.entries.get(key) {
            // Update access order
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.clone());
        }

        self.entries.get(key).map(|e| &e.value)
    }

    /// Remove a value from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.size_bytes);
            self.access_order.retain(|k| k != key);
            Some(entry.value)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_bytes = 0;
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the current size in bytes
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Remove all entries older than the given age
    pub fn evict_older_than(&mut self, max_age_secs: u64) -> usize {
        let mut to_remove = Vec::new();

        for (key, entry) in &self.entries {
            if entry.is_older_than(max_age_secs) {
                to_remove.push(key.clone());
            }
        }

        let count = to_remove.len();
        for key in to_remove {
            self.remove(&key);
        }

        count
    }
}

// ============================================================================
// File Metadata Cache Key
// ============================================================================

/// A cache key based on file path and modification time
///
/// This ensures that cached data is invalidated when files change.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileCacheKey {
    pub path: PathBuf,
    pub modified_time: u64,
    pub file_size: u64,
}

impl FileCacheKey {
    /// Create a cache key from a file path
    ///
    /// Returns None if the file doesn't exist or metadata can't be read.
    pub fn from_path(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;

        let modified_time = metadata.modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs();

        let file_size = metadata.len();

        Some(Self {
            path: path.to_path_buf(),
            modified_time,
            file_size,
        })
    }

    /// Create a cache key from a file path with explicit content hash
    pub fn from_path_with_hash(path: &Path, _content_hash: &str) -> Self {
        Self {
            path: path.to_path_buf(),
            modified_time: 0,
            file_size: 0,
        }
    }
}

// ============================================================================
// Global Cache Manager
// ============================================================================

/// Global cache manager for Luna
///
/// Manages multiple cache types with different size limits.
pub struct CacheManager {
    /// Cache for ScopeGraph results (default: 100MB)
    pub scope_graph: LruCache<FileCacheKey, Vec<u8>>,

    /// Cache for tokenization results (default: 50MB)
    pub tokenization: LruCache<FileCacheKey, Vec<u32>>,
}

impl CacheManager {
    /// Create a new cache manager with default size limits
    pub fn new() -> Self {
        Self {
            scope_graph: LruCache::new(100 * 1024 * 1024), // 100MB
            tokenization: LruCache::new(50 * 1024 * 1024),  // 50MB
        }
    }

    /// Create a new cache manager with custom size limits
    pub fn with_limits(scope_graph_bytes: usize, tokenization_bytes: usize) -> Self {
        Self {
            scope_graph: LruCache::new(scope_graph_bytes),
            tokenization: LruCache::new(tokenization_bytes),
        }
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        self.scope_graph.clear();
        self.tokenization.clear();
    }

    /// Get total cache size in bytes
    pub fn total_bytes(&self) -> usize {
        self.scope_graph.current_bytes() + self.tokenization.current_bytes()
    }

    /// Evict old entries from all caches
    pub fn evict_old(&mut self, max_age_secs: u64) -> usize {
        let count = self.scope_graph.evict_older_than(max_age_secs);
        self.tokenization.evict_older_than(max_age_secs);
        count
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a hash key for caching
pub fn hash_key(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_insert_get() {
        let mut cache = LruCache::new(100);

        cache.insert("key1", "value1", 10);
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LruCache::new(20);

        cache.insert("key1", "value1", 10);
        cache.insert("key2", "value2", 8);
        cache.insert("key3", "value3", 5); // Total: 23 > 20

        // One entry should be evicted
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry::new("value", 10);
        assert!(!entry.is_older_than(1000)); // Not older than 1000 seconds
    }

    #[test]
    fn test_hash_key() {
        let hash1 = hash_key(b"hello");
        let hash2 = hash_key(b"hello");
        let hash3 = hash_key(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_file_cache_key() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        std::fs::write(&file_path, b"hello").unwrap();

        let key = FileCacheKey::from_path(&file_path);
        assert!(key.is_some());

        let key = key.unwrap();
        assert_eq!(key.path, file_path);
        assert!(key.file_size == 5);
    }

    #[test]
    fn test_cache_manager() {
        let manager = CacheManager::new();
        assert_eq!(manager.total_bytes(), 0);

        let mut manager = CacheManager::with_limits(100, 50);
        assert_eq!(manager.scope_graph.max_bytes, 100);
        assert_eq!(manager.tokenization.max_bytes, 50);
    }
}
