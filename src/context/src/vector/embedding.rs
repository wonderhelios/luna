//! Embedding Model Abstraction
//!
//! Provides a unified interface for different embedding providers.
//! Supports both synchronous and asynchronous encoding.

use std::sync::Arc;

/// A vector embedding
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Dimension of the embedding
    pub dim: usize,
    /// Model used to generate the embedding
    pub model: String,
}

impl Embedding {
    /// Create a new embedding
    pub fn new(vector: Vec<f32>, model: impl Into<String>) -> Self {
        let dim = vector.len();
        Self {
            vector,
            dim,
            model: model.into(),
        }
    }

    /// Normalize the embedding to unit length (L2 norm)
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Get normalized copy
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

/// Trait for embedding models
///
/// Implement this trait to add support for new embedding providers.
pub trait EmbeddingModel: Send + Sync {
    /// Encode a single text into an embedding vector
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// The embedding vector or an error
    fn encode(&self, text: &str) -> error::Result<Embedding>;

    /// Encode multiple texts in a batch
    ///
    /// Default implementation encodes sequentially.
    fn encode_batch(&self, texts: &[String]) -> error::Result<Vec<Embedding>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Check if the model is available
    fn is_available(&self) -> bool {
        true
    }
}

impl EmbeddingModel for Arc<dyn EmbeddingModel> {
    fn encode(&self, text: &str) -> error::Result<Embedding> {
        (**self).encode(text)
    }

    fn encode_batch(&self, texts: &[String]) -> error::Result<Vec<Embedding>> {
        (**self).encode_batch(texts)
    }

    fn embedding_dim(&self) -> usize {
        (**self).embedding_dim()
    }

    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn is_available(&self) -> bool {
        (**self).is_available()
    }
}

// ============================================================================
// Ollama Local Embedding
// ============================================================================

/// Ollama-based local embedding model
///
/// Uses Ollama's embedding API to generate embeddings locally.
/// No API key required, runs entirely on your machine.
#[derive(Debug, Clone)]
pub struct OllamaEmbedding {
    base_url: String,
    model: String,
    embedding_dim: usize,
}

impl OllamaEmbedding {
    /// Create a new Ollama embedding client
    ///
    /// # Arguments
    /// * `base_url` - Ollama server URL (e.g., "http://localhost:11434")
    /// * `model` - Model name (e.g., "nomic-embed-text", "mxbai-embed-large")
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            embedding_dim,
        }
    }

    /// Create with default localhost configuration
    pub fn localhost(model: impl Into<String>, embedding_dim: usize) -> Self {
        Self::new("http://localhost:11434", model, embedding_dim)
    }

    /// Default nomic-embed-text configuration
    /// 768 dimensions, fast and good quality
    pub fn nomic_embed_text() -> Self {
        Self::localhost("nomic-embed-text", 768)
    }

    /// mxbai-embed-large configuration
    /// 1024 dimensions, higher quality but slower
    pub fn mxbai_embed_large() -> Self {
        Self::localhost("mxbai-embed-large", 1024)
    }

    /// Check if Ollama server is reachable
    pub fn is_server_ready(&self) -> bool {
        // Try to fetch the tags endpoint
        let url = format!("{}/api/tags", self.base_url);
        match reqwest::blocking::get(&url) {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Ensure the model is pulled
    pub fn ensure_model(&self) -> error::Result<()> {
        if !self.is_server_ready() {
            return Err(error::LunaError::network(
                "Ollama server not available. Start it with: ollama serve",
            ));
        }

        // Check if model exists
        let url = format!("{}/api/tags", self.base_url);
        let resp = reqwest::blocking::get(&url)
            .map_err(|e| error::LunaError::network(format!("Failed to connect to Ollama: {}", e)))?;

        let tags: serde_json::Value = resp.json().map_err(|e| {
            error::LunaError::invalid_input(format!("Failed to parse Ollama response: {}", e))
        })?;

        let models: Vec<String> = tags
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if !models.iter().any(|m| m.starts_with(&self.model)) {
            // Model not found, try to pull it
            tracing::info!("Pulling Ollama model: {}", self.model);
            self.pull_model()?;
        }

        Ok(())
    }

    fn pull_model(&self) -> error::Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .json(&serde_json::json!({
                "name": self.model,
            }))
            .send()
            .map_err(|e| error::LunaError::network(format!("Failed to pull model: {}", e)))?;

        if !resp.status().is_success() {
            return Err(error::LunaError::invalid_input(format!(
                "Failed to pull model: {}",
                resp.status()
            )));
        }

        Ok(())
    }
}

impl EmbeddingModel for OllamaEmbedding {
    fn encode(&self, text: &str) -> error::Result<Embedding> {
        let url = format!("{}/api/embeddings", self.base_url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| error::LunaError::internal(format!("Failed to create HTTP client: {}", e)))?;

        let resp = client
            .post(&url)
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": text,
            }))
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    error::LunaError::timeout("Ollama embedding request timed out".to_string())
                } else {
                    error::LunaError::network(format!("Ollama request failed: {}", e))
                }
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(error::LunaError::invalid_input(format!(
                "Ollama API error ({}): {}",
                status, body
            )));
        }

        let result: serde_json::Value = resp.json().map_err(|e| {
            error::LunaError::invalid_input(format!("Failed to parse Ollama response: {}", e))
        })?;

        let embedding = result
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| error::LunaError::invalid_input("Missing embedding in response"))?;

        let vector: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if vector.len() != self.embedding_dim {
            return Err(error::LunaError::invalid_input(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                vector.len()
            )));
        }

        Ok(Embedding::new(vector, self.model.clone()))
    }

    fn encode_batch(&self, texts: &[String]) -> error::Result<Vec<Embedding>> {
        // Ollama doesn't support batch encoding, encode sequentially
        texts.iter().map(|t| self.encode(t)).collect()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn is_available(&self) -> bool {
        self.is_server_ready()
    }
}

// ============================================================================
// OpenAI Embedding
// ============================================================================

/// OpenAI embedding API client
///
/// Uses OpenAI's embedding API. Requires an API key.
#[derive(Debug, Clone)]
pub struct OpenAIEmbedding {
    api_key: String,
    base_url: String,
    model: String,
    embedding_dim: usize,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding client
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key
    /// * `model` - Model name (default: "text-embedding-3-small")
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: model.into(),
            embedding_dim,
        }
    }

    /// Create with custom base URL (e.g., for OpenRouter or Azure)
    pub fn with_base_url(
        mut self,
        base_url: impl Into<String>,
    ) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Default text-embedding-3-small configuration
    /// 1536 dimensions, good balance of quality and cost
    pub fn text_embedding_3_small(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-small", 1536)
    }

    /// text-embedding-3-large configuration
    /// 3072 dimensions, highest quality
    pub fn text_embedding_3_large(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-large", 3072)
    }

    /// text-embedding-ada-002 configuration
    /// 1536 dimensions, legacy model
    pub fn ada_002(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-ada-002", 1536)
    }
}

impl EmbeddingModel for OpenAIEmbedding {
    fn encode(&self, text: &str) -> error::Result<Embedding> {
        let url = format!("{}/embeddings", self.base_url);

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
                "encoding_format": "float",
            }))
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    error::LunaError::timeout("OpenAI embedding request timed out".to_string())
                } else {
                    error::LunaError::network(format!("OpenAI request failed: {}", e))
                }
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(error::LunaError::invalid_input(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let result: serde_json::Value = resp.json().map_err(|e| {
            error::LunaError::invalid_input(format!("Failed to parse OpenAI response: {}", e))
        })?;

        let embedding = result
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|e| e.as_array())
            .ok_or_else(|| error::LunaError::invalid_input("Missing embedding in response"))?;

        let vector: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if vector.len() != self.embedding_dim {
            return Err(error::LunaError::invalid_input(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                vector.len()
            )));
        }

        Ok(Embedding::new(vector, self.model.clone()))
    }

    fn encode_batch(&self, texts: &[String]) -> error::Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            }))
            .send()
            .map_err(|e| error::LunaError::network(format!("OpenAI request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(error::LunaError::invalid_input(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let result: serde_json::Value = resp.json().map_err(|e| {
            error::LunaError::invalid_input(format!("Failed to parse OpenAI response: {}", e))
        })?;

        let data = result
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| error::LunaError::invalid_input("Missing data in response"))?;

        let mut embeddings = Vec::with_capacity(texts.len());

        for item in data {
            let vector = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| error::LunaError::invalid_input("Missing embedding in item"))?;

            let vec_f32: Vec<f32> = vector
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            embeddings.push(Embedding::new(vec_f32, self.model.clone()));
        }

        Ok(embeddings)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ============================================================================
// Mock Embedding for Testing
// ============================================================================

/// Mock embedding model for testing
pub struct MockEmbedding {
    dim: usize,
    model: String,
}

impl MockEmbedding {
    /// Create a new mock embedding model
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            model: "mock".to_string(),
        }
    }

    /// Generate deterministic embedding from text
    fn generate(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut vector = Vec::with_capacity(self.dim);
        let mut state = seed;

        for _ in 0..self.dim {
            // Simple pseudo-random number generator
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((state >> 33) as f32) / (u32::MAX as f32);
            vector.push(value * 2.0 - 1.0); // Scale to [-1, 1]
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }
}

impl EmbeddingModel for MockEmbedding {
    fn encode(&self, text: &str) -> error::Result<Embedding> {
        let vector = self.generate(text);
        Ok(Embedding::new(vector, self.model.clone()))
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0, 0.0], "test");
        emb.normalize();

        // 3-4-5 triangle normalized
        assert!((emb.vector[0] - 0.6).abs() < 0.001);
        assert!((emb.vector[1] - 0.8).abs() < 0.001);
        assert!(emb.vector[2].abs() < 0.001);
    }

    #[test]
    fn test_mock_embedding() {
        let mock = MockEmbedding::new(128);
        let emb1 = mock.encode("hello").unwrap();
        let emb2 = mock.encode("hello").unwrap();
        let emb3 = mock.encode("world").unwrap();

        // Same text should produce same embedding
        assert_eq!(emb1.vector, emb2.vector);

        // Different text should produce different embeddings
        assert_ne!(emb1.vector, emb3.vector);

        // Should be normalized
        let norm: f32 = emb1.vector.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
