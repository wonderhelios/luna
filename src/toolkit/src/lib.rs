//! Toolkit: Tool Abstraction Layer for Agents
//!
//! This crate provides a trait-based abstraction for agent tools,
//! decoupling the react agent from concrete tool implementations.
//!
//! Design Principles:
//! - Trait-based: All tools implement the `Tool` trait
//! - Self-documenting: Each tool provides its own schema
//! - Composable: Tools can be chained or combined

mod registry;
mod tools;

pub use registry::ToolRegistry;

// Re-export common tool implementations
pub use tools::{
    ReadFileTool,
    EditFileTool,
    ListDirTool,
    RunTerminalTool,
};

use core::code_chunk::{IndexChunk, ContextChunk};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Common Tool Types
// ============================================================================

/// Input to a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// Tool arguments (JSON)
    pub args: serde_json::Value,

    /// Repository root path
    pub repo_root: PathBuf,
}

/// Output from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Whether the tool execution succeeded
    pub success: bool,

    /// Output data
    pub data: serde_json::Value,

    /// Error message if failed
    pub error: Option<String>,

    /// Trace information for debugging
    pub trace: String,

    /// Additional context chunks generated
    pub context_chunks: Vec<ContextChunk>,

    /// Search hits generated
    pub hits: Vec<IndexChunk>,
}

impl ToolOutput {
    /// Create a successful output with data
    pub fn success(data: serde_json::Value) -> Self {
        Self {
            success: true,
            data,
            error: None,
            trace: String::new(),
            context_chunks: Vec::new(),
            hits: Vec::new(),
        }
    }

    /// Create a failed output with error
    pub fn error<E: Into<String>>(error: E) -> Self {
        Self {
            success: false,
            data: serde_json::Value::Null,
            error: Some(error.into()),
            trace: String::new(),
            context_chunks: Vec::new(),
            hits: Vec::new(),
        }
    }

    /// Add context chunks to the output
    pub fn with_context(mut self, chunks: Vec<ContextChunk>) -> Self {
        self.context_chunks = chunks;
        self
    }

    /// Add search hits to the output
    pub fn with_hits(mut self, hits: Vec<IndexChunk>) -> Self {
        self.hits = hits;
        self
    }

    /// Add trace information
    pub fn with_trace<S: Into<String>>(mut self, trace: S) -> Self {
        self.trace = trace.into();
        self
    }
}

/// Schema describing a tool's capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name (identifier)
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Input schema (JSON Schema)
    pub input_schema: serde_json::Value,

    /// Output schema (JSON Schema)
    pub output_schema: serde_json::Value,
}

// ============================================================================
// Tool Trait
// ============================================================================

/// Abstract interface for agent tools
///
/// All tools must implement this trait to be usable by agents.
/// This decouples the react agent from concrete tool implementations.
pub trait Tool: Send + Sync {
    /// Get the tool's unique identifier
    fn name(&self) -> &str;

    /// Get the tool's schema
    fn schema(&self) -> ToolSchema;

    /// Execute the tool with given input
    fn execute(&self, input: &ToolInput) -> ToolOutput;

    /// Validate input before execution (optional)
    fn validate(&self, input: &ToolInput) -> Result<(), anyhow::Error> {
        let _ = input;
        Ok(())
    }

    /// Check if this tool can handle a given action
    fn can_handle(&self, action: &str) -> bool {
        self.name() == action
    }
}

// ============================================================================
// Helper Functions for Tool Implementations
// ============================================================================

/// Helper function to parse file path from JSON args
pub fn parse_path(args: &serde_json::Value, key: &str) -> Result<PathBuf, anyhow::Error> {
    let path = args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing field: {}", key))?;
    Ok(PathBuf::from(path))
}

/// Helper function to parse string from JSON args
pub fn parse_string(args: &serde_json::Value, key: &str) -> Result<String, anyhow::Error> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing field: {}", key))
}

/// Helper function to parse usize from JSON args
pub fn parse_usize(args: &serde_json::Value, key: &str) -> Result<usize, anyhow::Error> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("missing or invalid field: {}", key))
}

/// Helper function to parse bool from JSON args
pub fn parse_bool(args: &serde_json::Value, key: &str) -> Result<bool, anyhow::Error> {
    args.get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| anyhow::anyhow!("missing or invalid field: {}", key))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_output_success() {
        let output = ToolOutput::success(serde_json::json!({"result": "ok"}));
        assert!(output.success);
        assert!(output.error.is_none());
    }

    #[test]
    fn test_tool_output_error() {
        let output = ToolOutput::error("something went wrong");
        assert!(!output.success);
        assert_eq!(output.error, Some("something went wrong".to_string()));
    }

    #[test]
    fn test_tool_output_error_anyhow() {
        let err = anyhow::anyhow!("test error");
        let output = ToolOutput::error(format!("{}", err));
        assert!(!output.success);
        assert_eq!(output.error, Some("test error".to_string()));
    }

    #[test]
    fn test_parse_string() {
        let args = serde_json::json!({"name": "test"});
        assert_eq!(parse_string(&args, "name").unwrap(), "test");
        assert!(parse_string(&args, "missing").is_err());
    }

    #[test]
    fn test_parse_usize() {
        let args = serde_json::json!({"count": 42});
        assert_eq!(parse_usize(&args, "count").unwrap(), 42);
        assert!(parse_usize(&args, "missing").is_err());
    }
}
