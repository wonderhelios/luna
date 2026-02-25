//! Tool Registry for managing available tools

use crate::{Tool, ToolInput, ToolOutput, ToolSchema};
use std::collections::HashMap;

/// Registry for managing available tools
///
/// The registry allows:
/// - Dynamic tool registration
/// - Tool discovery
/// - Tool execution by name
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Register multiple tools
    pub fn register_all(&mut self, tools: Vec<Box<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Check if a tool is registered
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// List all registered tool names
    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Execute a tool by name
    pub fn execute(&self, name: &str, input: &ToolInput) -> ToolOutput {
        if let Some(tool) = self.get(name) {
            // Validate input first
            if let Err(e) = tool.validate(input) {
                return ToolOutput::error(format!("validation failed: {}", e));
            }
            tool.execute(input)
        } else {
            ToolOutput::error(format!("tool not found: {}", name))
        }
    }

    /// Get all tool schemas
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool {
        name: String,
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.clone(),
                description: "mock tool".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
            }
        }

        fn execute(&self, _input: &ToolInput) -> ToolOutput {
            ToolOutput::success(serde_json::json!({"result": "ok"}))
        }
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(MockTool {
            name: "test".to_string(),
        }));
        assert_eq!(registry.len(), 1);
        assert!(registry.has("test"));
    }

    #[test]
    fn test_registry_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool {
            name: "test".to_string(),
        }));

        let input = ToolInput {
            args: serde_json::json!({}),
            repo_root: std::path::PathBuf::from("."),
            policy: None,
        };

        let output = registry.execute("test", &input);
        assert!(output.success);
    }

    #[test]
    fn test_registry_execute_unknown() {
        let registry = ToolRegistry::new();

        let input = ToolInput {
            args: serde_json::json!({}),
            repo_root: std::path::PathBuf::from("."),
            policy: None,
        };

        let output = registry.execute("unknown", &input);
        assert!(!output.success);
    }
}
