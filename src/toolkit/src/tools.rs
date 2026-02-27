//! Concrete tool implementations
//!
//! These tools wrap the existing functions from the `tools` crate
//! to implement the `Tool` trait.

use crate::{
    parse_bool, parse_path, parse_string, parse_usize, Tool, ToolInput, ToolOutput, ToolSchema,
};
use serde_json::json;

// Import functions from the tools crate
use tools::{edit_file, find_symbol_definitions, list_dir, read_file, run_terminal, EditOp};

fn policy_of(input: &ToolInput) -> crate::ExecutionPolicy {
    input.policy.clone().unwrap_or_default()
}
// ============================================================================
// Read File Tool
// ============================================================================

/// Tool for reading file contents
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: "Read file contents, optionally with line range".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to repo_root)"
                    },
                    "start_line": {
                        "type": "number",
                        "description": "Start line (0-based, optional)"
                    },
                    "end_line": {
                        "type": "number",
                        "description": "End line (0-based, optional)"
                    }
                }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "File contents"
                    }
                }
            }),
        }
    }

    fn execute(&self, input: &ToolInput) -> ToolOutput {
        let args = &input.args;

        let path = match parse_path(args, "path") {
            Ok(p) => p,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let full_path = input.repo_root.join(&path);
        let range = match (
            parse_usize(args, "start_line").ok(),
            parse_usize(args, "end_line").ok(),
        ) {
            (Some(s), Some(e)) => Some((s, e)),
            (Some(s), None) => Some((s, s)),
            _ => None,
        };

        match read_file(&full_path, range) {
            Ok(content) => ToolOutput::success(json!({ "content": content }))
                .with_trace(format!("read {} bytes", content.len())),
            Err(e) => ToolOutput::error(format!("failed to read file: {}", e)),
        }
    }
}

// ============================================================================
// Edit File Tool
// ============================================================================

/// Tool for editing file contents
pub struct EditFileTool;

impl EditFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: "Edit file contents by replacing lines".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["path", "start_line", "end_line", "new_content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to repo_root)"
                    },
                    "start_line": {
                        "type": "number",
                        "description": "Start line (0-based, inclusive)"
                    },
                    "end_line": {
                        "type": "number",
                        "description": "End line (0-based, inclusive)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Create backup before editing",
                        "default": false
                    },
                    "confirm": {
                        "type":"boolean",
                        "description": "Explicit confirmation for potentially destructive actions",
                        "default": false
                    }
                }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "lines_changed": {"type": "number"},
                    "backup_path": {"type": "string"}
                }
            }),
        }
    }

    fn execute(&self, input: &ToolInput) -> ToolOutput {
        let policy = policy_of(input);
        if !policy.allow_edit_file {
            return ToolOutput::error("edit file is disabled by policy")
                .with_trace("policy_blocked".to_string());
        }
        let args = &input.args;

        if policy.require_confirm_edit_file {
            let confirmed = parse_bool(args, "confirm").unwrap_or(false);
            if !confirmed {
                return ToolOutput::error(
                    "edit_file requires explicit confirmation: set confirm=true in args",
                )
                .with_trace("confirmation_required".to_string());
            }
        }

        let path = match parse_path(args, "path") {
            Ok(p) => p,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let full_path = input.repo_root.join(&path);

        let start_line = match parse_usize(args, "start_line") {
            Ok(v) => v,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let end_line = match parse_usize(args, "end_line") {
            Ok(v) => v,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let new_content = match parse_string(args, "new_content") {
            Ok(v) => v,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let create_backup = parse_bool(args, "create_backup").unwrap_or(false);

        let op = EditOp::ReplaceLines {
            start_line,
            end_line,
            new_content,
        };

        match edit_file(&full_path, &op, create_backup) {
            Ok(result) => {
                if result.success {
                    ToolOutput::success(json!({
                        "success": true,
                        "lines_changed": result.lines_changed.unwrap_or(0),
                        "backup_path": result.backup_path,
                    }))
                    .with_trace(format!("edited {} lines", end_line - start_line + 1))
                } else {
                    ToolOutput::error(format!("edit failed: {}", result.error.unwrap_or_default()))
                }
            }
            Err(e) => ToolOutput::error(format!("edit error: {}", e)),
        }
    }
}

// ============================================================================
// List Directory Tool
// ============================================================================

/// Tool for listing directory contents
pub struct ListDirTool;

impl ListDirTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ListDirTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: "List directory contents".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to repo_root)"
                    }
                }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "is_dir": {"type": "boolean"},
                                "is_file": {"type": "boolean"},
                                "size": {"type": "number"}
                            }
                        }
                    }
                }
            }),
        }
    }

    fn execute(&self, input: &ToolInput) -> ToolOutput {
        let args = &input.args;

        let path = match parse_path(args, "path") {
            Ok(p) => p,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let full_path = input.repo_root.join(&path);

        match list_dir(&full_path) {
            Ok(entries) => {
                let entries_json: Vec<serde_json::Value> = entries
                    .into_iter()
                    .map(|e| {
                        json!({
                            "name": e.name,
                            "is_dir": e.is_dir,
                            "is_file": e.is_file,
                            "size": e.size,
                        })
                    })
                    .collect();

                ToolOutput::success(json!({ "entries": entries_json }))
                    .with_trace(format!("listed {} entries", entries_json.len()))
            }
            Err(e) => ToolOutput::error(format!("failed to list directory: {}", e)),
        }
    }
}

// ============================================================================
// Run Terminal Tool
// ============================================================================

/// Tool for running terminal commands
pub struct RunTerminalTool;

impl RunTerminalTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RunTerminalTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for RunTerminalTool {
    fn name(&self) -> &str {
        "run_terminal"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: "Run terminal commands with safety checks".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "allow_dangerous": {
                        "type": "boolean",
                        "description": "Allow potentially dangerous commands",
                        "default": false
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Explicit confirmation for command execution",
                        "default": false
                    }
                }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "exit_code": {"type": "number"},
                    "error": {"type": "string"}
                }
            }),
        }
    }

    fn execute(&self, input: &ToolInput) -> ToolOutput {
        let policy = policy_of(input);
        if !policy.allow_run_terminal {
            return ToolOutput::error("run_terminal is disabled by policy")
                .with_trace("policy_blocked".to_string());
        }

        let args = &input.args;

        let command = match parse_string(args, "command") {
            Ok(c) => c,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let allow_dangerous = parse_bool(args, "allow_dangerous").unwrap_or(false);

        if policy.require_confirm_run_terminal {
            let confirmed = parse_bool(args, "confirm").unwrap_or(false);
            if !confirmed {
                return ToolOutput::error(
                    "run_terminal requires explicit confirmation: set confirm=true in args",
                )
                .with_trace("confirmation_required".to_string());
            }
        }

        match run_terminal(&command, Some(&input.repo_root), allow_dangerous) {
            Ok(result) => {
                if result.success {
                    ToolOutput::success(json!({
                        "success": true,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                    }))
                    .with_trace(format!("command exited with {:?}", result.exit_code))
                } else {
                    ToolOutput::error(format!(
                        "command failed: {}",
                        result.error.unwrap_or_else(|| "unknown".to_string())
                    ))
                }
            }
            Err(e) => ToolOutput::error(format!("terminal error: {}", e)),
        }
    }
}

// ============================================================================
// Goto Definition Tool
// ============================================================================

/// Tool for finding symbol definitions (go-to-definition)
pub struct GotoDefinitionTool;

impl GotoDefinitionTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GotoDefinitionTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GotoDefinitionTool {
    fn name(&self) -> &str {
        "goto_definition"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: "Find the definition location of a symbol (function, struct, etc.) across the repository".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["symbol_name"],
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find (e.g., 'list_dir', 'ContextChunk')"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "definitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start_line": {"type": "number"},
                                "end_line": {"type": "number"},
                                "kind": {"type": "string"}
                            }
                        }
                    }
                }
            }),
        }
    }

    fn execute(&self, input: &ToolInput) -> ToolOutput {
        let args = &input.args;

        let symbol_name = match parse_string(args, "symbol_name") {
            Ok(s) => s,
            Err(e) => return ToolOutput::error(format!("{}", e)),
        };

        let max_results = parse_usize(args, "max_results").unwrap_or(5);

        match find_symbol_definitions(&input.repo_root, &symbol_name, max_results) {
            Ok(definitions) => {
                if definitions.is_empty() {
                    ToolOutput::success(json!({
                        "definitions": [],
                        "message": format!("No definitions found for '{}'", symbol_name)
                    }))
                    .with_trace(format!("no definitions found for '{}'", symbol_name))
                } else {
                    let defs_json: Vec<serde_json::Value> = definitions
                        .iter()
                        .map(|d| json!({
                            "path": d.path,
                            "start_line": d.start_line,
                            "end_line": d.end_line,
                            "kind": d.kind
                        }))
                        .collect();
                    ToolOutput::success(json!({ "definitions": defs_json }))
                        .with_trace(format!("found {} definitions for '{}'", definitions.len(), symbol_name))
                }
            }
            Err(e) => ToolOutput::error(format!("search error: {}", e)),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_tool_schema() {
        let tool = ReadFileTool::new();
        let schema = tool.schema();
        assert_eq!(schema.name, "read_file");
        assert!(schema.input_schema.is_object());
    }

    #[test]
    fn test_edit_file_tool_schema() {
        let tool = EditFileTool::new();
        let schema = tool.schema();
        assert_eq!(schema.name, "edit_file");
    }

    #[test]
    fn test_list_dir_tool_schema() {
        let tool = ListDirTool::new();
        let schema = tool.schema();
        assert_eq!(schema.name, "list_dir");
    }

    #[test]
    fn test_run_terminal_tool_schema() {
        let tool = RunTerminalTool::new();
        let schema = tool.schema();
        assert_eq!(schema.name, "run_terminal");
    }

    #[test]
    fn test_goto_definition_tool_schema() {
        let tool = GotoDefinitionTool::new();
        let schema = tool.schema();
        assert_eq!(schema.name, "goto_definition");
        assert!(schema.description.contains("definition"));
    }
}
