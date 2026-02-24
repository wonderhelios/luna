//! Centralized configuration for Luna
//!
//! This module provides a single source of truth for configuration values
//! that were previously hardcoded throughout the codebase.
//!
//! Configuration priority:
//! 1. Environment variables
//! 2. Config file (luna.toml in project root or ~/.config/luna/config.toml)
//! 3. Default values

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Search Configuration
// ============================================================================

/// Configuration for code search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum number of files to scan during search
    pub max_files: usize,

    /// Maximum number of hits to return
    pub max_hits: usize,

    /// Maximum file size in bytes to consider
    pub max_file_bytes: usize,

    /// Directories to ignore during search
    pub ignore_dirs: Vec<String>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_files: 8_000,
            max_hits: 64,
            max_file_bytes: 500 * 1_000,
            ignore_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "dist".to_string(),
                "build".to_string(),
            ],
        }
    }
}

// ============================================================================
// Cache Configuration
// ============================================================================

/// Configuration for caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size for ScopeGraph in bytes
    pub scope_graph_max_bytes: usize,

    /// Maximum cache size for tokenization in bytes
    pub tokenization_max_bytes: usize,

    /// Maximum age for cache entries in seconds
    pub max_age_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            scope_graph_max_bytes: 100 * 1024 * 1024, // 100MB
            tokenization_max_bytes: 50 * 1024 * 1024,  // 50MB
            max_age_secs: 3600, // 1 hour
        }
    }
}

// ============================================================================
// ReAct Agent Configuration
// ============================================================================

/// Configuration for the ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactConfig {
    /// Maximum number of steps in the ReAct loop
    pub max_steps: usize,

    /// Maximum context chunks to include
    pub max_context_chunks: usize,

    /// Maximum context tokens
    pub max_context_tokens: usize,

    /// Search hits for initial query
    pub initial_search_hits: usize,

    /// Search hits for follow-up queries
    pub followup_search_hits: usize,
}

impl Default for ReactConfig {
    fn default() -> Self {
        Self {
            max_steps: 3,
            max_context_chunks: 8,
            max_context_tokens: 4_000,
            initial_search_hits: 200,
            followup_search_hits: 50,
        }
    }
}

// ============================================================================
// Chunk Configuration
// ============================================================================

/// Configuration for code chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum chunk size in tokens
    pub max_chunk_tokens: usize,

    /// Maximum chunk size in lines
    pub max_chunk_lines: usize,

    /// Overlap between chunks in lines
    pub overlap_lines: usize,

    /// Maximum chunk size in bytes
    pub max_chunk_bytes: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_tokens: 512,
            max_chunk_lines: 200,
            overlap_lines: 20,
            max_chunk_bytes: 20_000,
        }
    }
}

// ============================================================================
// LLM Configuration
// ============================================================================

/// Configuration for LLM API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API base URL
    pub api_base: String,

    /// API key (loaded from environment by default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Model name
    pub model: String,

    /// Sampling temperature
    pub temperature: f32,

    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_base: "https://open.bigmodel.cn/api/paas/v4/".to_string(),
            api_key: None,
            model: "glm-4-flash".to_string(),
            temperature: 0.2,
            timeout_secs: 120,
        }
    }
}

// ============================================================================
// Main Configuration
// ============================================================================

/// Main configuration structure for Luna
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Search configuration
    #[serde(default)]
    pub search: SearchConfig,

    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,

    /// ReAct agent configuration
    #[serde(default)]
    pub react: ReactConfig,

    /// Chunk configuration
    #[serde(default)]
    pub chunk: ChunkConfig,

    /// LLM configuration
    #[serde(default)]
    pub llm: LlmConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            search: SearchConfig::default(),
            cache: CacheConfig::default(),
            react: ReactConfig::default(),
            chunk: ChunkConfig::default(),
            llm: LlmConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a file
    ///
    /// Looks for `luna.toml` in the current directory or
    /// `~/.config/luna/config.toml`
    pub fn load() -> anyhow::Result<Self> {
        // First, try current directory
        if let Ok(content) = std::fs::read_to_string("luna.toml") {
            return Ok(toml::from_str(&content)?);
        }

        // Then try XDG config directory
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("luna").join("config.toml");
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                return Ok(toml::from_str(&content)?);
            }
        }

        // Return default if no config file found
        Ok(Self::default())
    }

    /// Load configuration with overrides from environment variables
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut config = Self::load()?;

        // Override LLM API key from environment
        if let Ok(key) = std::env::var("LLM_API_KEY") {
            config.llm.api_key = Some(key);
        }

        if let Ok(base) = std::env::var("LLM_API_BASE") {
            if !base.trim().is_empty() {
                config.llm.api_base = base;
            }
        }

        if let Ok(model) = std::env::var("LLM_MODEL") {
            if !model.trim().is_empty() {
                config.llm.model = model;
            }
        }

        Ok(config)
    }

    /// Get the project root directory
    pub fn project_root() -> PathBuf {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    /// Resolve a path relative to the project root
    pub fn resolve_path(&self, path: impl AsRef<std::path::Path>) -> PathBuf {
        Self::project_root().join(path)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.search.max_files, 8_000);
        assert_eq!(config.search.max_hits, 64);
        assert_eq!(config.react.max_steps, 3);
        assert_eq!(config.llm.model, "glm-4-flash");
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert!(config.ignore_dirs.contains(&".git".to_string()));
        assert!(config.ignore_dirs.contains(&"target".to_string()));
    }

    #[test]
    fn test_react_config_default() {
        let config = ReactConfig::default();
        assert_eq!(config.max_steps, 3);
        assert_eq!(config.max_context_chunks, 8);
    }

    #[test]
    fn test_config_load_returns_default() {
        // Should return default config when no file exists
        let config = Config::load().unwrap();
        assert_eq!(config.search.max_files, 8_000);
    }
}
