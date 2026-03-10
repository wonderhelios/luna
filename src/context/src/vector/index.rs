//! Vector Index using HNSW (Hierarchical Navigable Small World)
//!
//! HNSW is a graph-based approximate nearest neighbor search algorithm.
//! It builds a multi-layer graph where:
//! - Layer 0 contains all vectors with dense connections
//! - Higher layers have fewer vectors with longer connections
//! - Search starts from top layer and descends
//!
//! This provides O(log n) search complexity with high recall.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::DistanceMetric;

/// Unique identifier for indexed vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(pub Uuid);

impl VectorId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VectorId {
    fn default() -> Self {
        Self::new()
    }
}

/// A vector entry in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Search result from vector index
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub score: f32,
}

impl SearchResult {
    /// Convert distance to similarity score (0-1, higher is better)
    pub fn from_distance(id: VectorId, distance: f32, metric: DistanceMetric) -> Self {
        let score = match metric {
            DistanceMetric::Cosine => 1.0 - distance, // Cosine distance is already 0-1
            DistanceMetric::Euclidean => {
                // Convert euclidean to approximate similarity
                1.0 / (1.0 + distance)
            }
            DistanceMetric::DotProduct => -distance, // Dot product: negative = more similar
        }
        .clamp(0.0, 1.0);

        Self {
            id,
            distance,
            score,
        }
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap (we want smallest distance first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Trait for vector index implementations
pub trait VectorIndex: Send + Sync {
    /// Insert a vector into the index
    fn insert(&mut self, entry: VectorEntry) -> error::Result<()>;

    /// Remove a vector from the index
    fn remove(&mut self, id: VectorId) -> error::Result<bool>;

    /// Search for nearest neighbors
    fn search(&self, query: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a vector by ID
    fn get(&self, id: VectorId) -> Option<&VectorEntry>;

    /// Clear all vectors from the index
    fn clear(&mut self);
}

/// HNSW (Hierarchical Navigable Small World) Index
///
/// This is a simplified implementation of the HNSW algorithm.
/// For production use, consider using a specialized library like `hnsw` crate.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    /// Maximum number of connections per node (M parameter)
    max_connections: usize,
    /// Expansion factor during construction
    ef_construction: usize,
    /// Expansion factor during search
    ef_search: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// All vectors
    vectors: HashMap<VectorId, VectorEntry>,
    /// Layer 0 connections (dense)
    layer0: HashMap<VectorId, Vec<VectorId>>,
    /// Upper layers (simplified: just layer 1 for now)
    upper_layers: Vec<HashMap<VectorId, Vec<VectorId>>>,
    /// Entry point for search (top layer)
    entry_point: Option<VectorId>,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(
        max_connections: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            max_connections,
            ef_construction,
            ef_search,
            metric,
            vectors: HashMap::new(),
            layer0: HashMap::new(),
            upper_layers: Vec::new(),
            entry_point: None,
        }
    }

    /// Create with default configuration
    pub fn default_with_dim(_dim: usize) -> Self {
        Self::new(
            16,              // max_connections
            100,             // ef_construction
            50,              // ef_search
            DistanceMetric::Cosine,
        )
    }

    /// Calculate distance between two vectors using the configured metric
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    /// Get distance between two indexed vectors
    fn distance_between(&self, id1: VectorId, id2: VectorId) -> Option<f32> {
        let v1 = self.vectors.get(&id1)?;
        let v2 = self.vectors.get(&id2)?;
        Some(self.distance(&v1.vector, &v2.vector))
    }

    /// Greedy search in a layer starting from a given point
    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: &HashMap<VectorId, Vec<VectorId>>,
    ) -> Vec<SearchResult> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Start with entry point
        if let Some(entry_vec) = self.vectors.get(&entry) {
            let dist = self.distance(query, &entry_vec.vector);
            candidates.push(SearchResult::from_distance(entry, dist, self.metric));
            results.push(SearchResult::from_distance(entry, dist, self.metric));
            visited.insert(entry);
        }

        while let Some(current) = candidates.pop() {
            // Check if we can still improve results
            let worst_in_results = results.peek().map(|r: &SearchResult| r.distance).unwrap_or(f32::INFINITY);

            if current.distance > worst_in_results {
                break; // No need to continue
            }

            // Explore neighbors
            if let Some(neighbors) = layer.get(&current.id) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    if let Some(neighbor_vec) = self.vectors.get(&neighbor_id) {
                        let dist = self.distance(query, &neighbor_vec.vector);
                        let result = SearchResult::from_distance(neighbor_id, dist, self.metric);

                        candidates.push(result.clone());

                        if results.len() < ef {
                            results.push(result);
                        } else if let Some(worst) = results.peek() {
                            if dist < worst.distance {
                                results.pop();
                                results.push(result);
                            }
                        }
                    }
                }
            }
        }

        // Convert to vec and sort by distance (ascending)
        let mut result_vec: Vec<_> = results.into_vec();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Find nearest neighbors using brute force (for small indices)
    fn brute_force_search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<_> = self
            .vectors
            .values()
            .map(|entry| {
                let dist = self.distance(query, &entry.vector);
                SearchResult::from_distance(entry.id, dist, self.metric)
            })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Select neighbors using simple heuristic (M closest)
    fn select_neighbors(&self, candidates: Vec<SearchResult>, m: usize) -> Vec<VectorId> {
        candidates
            .into_iter()
            .take(m)
            .map(|r| r.id)
            .collect()
    }

    /// Determine which layer a new node should be in
    /// Uses probabilistic leveling: layer 0 with probability 1, layer 1 with 1/2, etc.
    fn random_level(&self) -> usize {
        let mut level = 0;
        let mut rng = fastrand::Rng::new();
        while rng.f64() < 0.5 && level < 8 {
            level += 1;
        }
        level
    }

    /// Prune connections to keep at most max_connections
    fn prune_connections(
        &self,
        connections: &mut Vec<VectorId>,
        _keep: &[VectorId],
    ) {
        // Simple pruning: just keep the first max_connections
        // A more sophisticated implementation would use distance
        if connections.len() > self.max_connections {
            connections.truncate(self.max_connections);
        }
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, entry: VectorEntry) -> error::Result<()> {
        let id = entry.id;
        let level = self.random_level();

        // Handle first insertion
        if self.vectors.is_empty() {
            self.vectors.insert(id, entry);
            self.layer0.insert(id, Vec::new());
            self.entry_point = Some(id);
            return Ok(());
        }

        // Ensure upper layers exist
        while self.upper_layers.len() < level {
            self.upper_layers.push(HashMap::new());
        }

        // Search for nearest neighbors at each level
        let mut current_entry = self.entry_point.unwrap();

        // Search from top level down to level+1
        for layer_idx in (level..self.upper_layers.len()).rev() {
            let layer = &self.upper_layers[layer_idx];
            let results = self.search_layer(&entry.vector, current_entry, 1, layer);
            if let Some(best) = results.first() {
                current_entry = best.id;
            }
        }

        // At level, do proper ef search and connect
        if level > 0 {
            for layer_idx in (0..level).rev() {
                // Clone the layer to avoid borrow issues
                let layer_clone = if layer_idx == 0 {
                    self.layer0.clone()
                } else {
                    self.upper_layers[layer_idx - 1].clone()
                };

                let results = self.search_layer(&entry.vector, current_entry, self.ef_construction, &layer_clone);
                let neighbors = self.select_neighbors(results, self.max_connections);

                // Store connections
                if layer_idx == 0 {
                    self.layer0.insert(id, neighbors.clone());
                } else {
                    self.upper_layers[layer_idx - 1].insert(id, neighbors.clone());
                }

                // Bidirectional connections
                for neighbor_id in neighbors {
                    let neighbor_conns = if layer_idx == 0 {
                        self.layer0.entry(neighbor_id).or_default()
                    } else {
                        self.upper_layers[layer_idx - 1].entry(neighbor_id).or_default()
                    };

                    if !neighbor_conns.contains(&id) {
                        neighbor_conns.push(id);
                    }

                    // Prune if too many connections
                    if neighbor_conns.len() > self.max_connections {
                        neighbor_conns.truncate(self.max_connections);
                    }
                }

                // Update entry for next level
                let results = self.search_layer(&entry.vector, current_entry, 1, &layer_clone);
                if let Some(best) = results.first() {
                    current_entry = best.id;
                }
            }
        }

        // Layer 0: always add with ef search
        let results = self.search_layer(&entry.vector, current_entry, self.ef_construction, &self.layer0);
        let neighbors = self.select_neighbors(results, self.max_connections * 2);
        self.layer0.insert(id, neighbors.clone());

        // Bidirectional connections at layer 0
        for neighbor_id in neighbors {
            let neighbor_conns = self.layer0.entry(neighbor_id).or_default();
            if !neighbor_conns.contains(&id) {
                neighbor_conns.push(id);
            }
        }

        // Store the vector
        self.vectors.insert(id, entry);

        // Update entry point if needed
        if level >= self.upper_layers.len() && self.entry_point != Some(id) {
            self.entry_point = Some(id);
        }

        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> error::Result<bool> {
        if !self.vectors.contains_key(&id) {
            return Ok(false);
        }

        // Remove from layer 0
        self.layer0.remove(&id);
        for conns in self.layer0.values_mut() {
            conns.retain(|&conn_id| conn_id != id);
        }

        // Remove from upper layers
        for layer in &mut self.upper_layers {
            layer.remove(&id);
            for conns in layer.values_mut() {
                conns.retain(|&conn_id| conn_id != id);
            }
        }

        // Remove the vector
        self.vectors.remove(&id);

        // Update entry point if needed
        if self.entry_point == Some(id) {
            self.entry_point = self.vectors.keys().next().copied();
        }

        Ok(true)
    }

    fn search(&self, query: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // For small indices, use brute force
        if self.vectors.len() < 100 {
            return Ok(self.brute_force_search(query, top_k));
        }

        let ef = self.ef_search.max(top_k);
        let mut current_entry = self.entry_point.unwrap();

        // Search from top layer down
        for layer in self.upper_layers.iter().rev() {
            let results = self.search_layer(query, current_entry, 1, layer);
            if let Some(best) = results.first() {
                current_entry = best.id;
            }
        }

        // Final search at layer 0
        let results = self.search_layer(query, current_entry, ef, &self.layer0);

        Ok(results.into_iter().take(top_k).collect())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn get(&self, id: VectorId) -> Option<&VectorEntry> {
        self.vectors.get(&id)
    }

    fn clear(&mut self) {
        self.vectors.clear();
        self.layer0.clear();
        self.upper_layers.clear();
        self.entry_point = None;
    }
}

/// Simple flat index for small datasets
/// Uses brute force search, good for < 1000 vectors
#[derive(Debug, Default)]
pub struct FlatIndex {
    vectors: HashMap<VectorId, VectorEntry>,
    metric: DistanceMetric,
}

impl FlatIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            vectors: HashMap::new(),
            metric,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn insert(&mut self, entry: VectorEntry) -> error::Result<()> {
        self.vectors.insert(entry.id, entry);
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> error::Result<bool> {
        Ok(self.vectors.remove(&id).is_some())
    }

    fn search(&self, query: &[f32], top_k: usize) -> error::Result<Vec<SearchResult>> {
        let mut results: Vec<_> = self
            .vectors
            .values()
            .map(|entry| {
                let dist = self.metric.distance(query, &entry.vector);
                SearchResult::from_distance(entry.id, dist, self.metric)
            })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn get(&self, id: VectorId) -> Option<&VectorEntry> {
        self.vectors.get(&id)
    }

    fn clear(&mut self) {
        self.vectors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    #[test]
    fn test_flat_index_basic() {
        let mut index = FlatIndex::new(DistanceMetric::Euclidean);

        // Insert vectors
        for i in 0..10 {
            let entry = VectorEntry {
                id: VectorId::new(),
                vector: create_test_vector(4, i as f32),
                metadata: HashMap::new(),
            };
            index.insert(entry).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Search
        let query = create_test_vector(4, 5.0);
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // Closest should be 5.0
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn test_hnsw_index_basic() {
        let mut index = HnswIndex::default_with_dim(4);

        // Insert vectors
        let mut ids = Vec::new();
        for i in 0..50 {
            let id = VectorId::new();
            ids.push(id);
            let entry = VectorEntry {
                id,
                vector: vec![
                    i as f32,
                    (i * 2) as f32,
                    (i * 3) as f32,
                    (i * 4) as f32,
                ],
                metadata: HashMap::new(),
            };
            index.insert(entry).unwrap();
        }

        assert_eq!(index.len(), 50);

        // Search
        let query = vec![25.0, 50.0, 75.0, 100.0]; // Should be closest to i=25
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn test_cosine_similarity_search() {
        let mut index = FlatIndex::new(DistanceMetric::Cosine);

        // Insert orthogonal vectors
        let v1 = VectorEntry {
            id: VectorId::new(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        };
        let v2 = VectorEntry {
            id: VectorId::new(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        };
        let v3 = VectorEntry {
            id: VectorId::new(),
            vector: vec![0.5, 0.5, 0.0], // 45 degrees between v1 and v2
            metadata: HashMap::new(),
        };

        index.insert(v1.clone()).unwrap();
        index.insert(v2.clone()).unwrap();
        index.insert(v3.clone()).unwrap();

        // Search with v1 direction
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        // v1 should be closest (distance 0)
        assert_eq!(results[0].id, v1.id);
        assert!(results[0].distance < 0.001);

        // v3 should be closer than v2 (45° vs 90°)
        assert!(results[1].distance < results[2].distance);
    }

    #[test]
    fn test_remove() {
        let mut index = FlatIndex::new(DistanceMetric::Euclidean);

        let id = VectorId::new();
        let entry = VectorEntry {
            id,
            vector: vec![1.0, 2.0, 3.0],
            metadata: HashMap::new(),
        };
        index.insert(entry).unwrap();

        assert_eq!(index.len(), 1);
        assert!(index.get(id).is_some());

        assert!(index.remove(id).unwrap());
        assert_eq!(index.len(), 0);
        assert!(index.get(id).is_none());

        // Removing again should return false
        assert!(!index.remove(id).unwrap());
    }

    #[test]
    fn test_search_result_ordering() {
        let r1 = SearchResult::from_distance(VectorId::new(), 0.1, DistanceMetric::Cosine);
        let r2 = SearchResult::from_distance(VectorId::new(), 0.5, DistanceMetric::Cosine);
        let r3 = SearchResult::from_distance(VectorId::new(), 0.9, DistanceMetric::Cosine);

        // For min-heap, smaller distances should have higher priority (be "greater" for BinaryHeap)
        // So r1 > r2 > r3 in terms of ordering
        assert!(r1 > r2);
        assert!(r2 > r3);
        assert!(r1 > r3);
    }
}
