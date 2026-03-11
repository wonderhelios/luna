//! TPAR - Task Plan Act Review
//!
//! Minimal TPAR implementation inspired by OpenCode:
//! 1. Task: User input -> LLM decides tools
//! 2. Plan & Act: Execute tools sequentially
//! 3. Review: LLM synthesizes final response

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use error::{LunaError, Result};
use crate::llm::{CompletionRequest, LLMClient};
use intelligence::{
    repo_scan::{FsRepoFileProvider, RepoScanOptions},
    navigation::{TreeSitterNavigator, Navigator},
};

/// A tool call from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// Result of tool execution
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
}

/// TPAR Executor
pub struct TparExecutor {
    llm: Arc<dyn LLMClient>,
    tools: ToolRegistry,
}

impl TparExecutor {
    /// Create new executor
    pub fn new(llm: Arc<dyn LLMClient>) -> Self {
        Self {
            llm,
            tools: ToolRegistry::new(),
        }
    }

    /// Run a single turn
    pub fn run_turn(&self, input: &str, ctx: &TurnContext) -> Result<String> {
        // 1. Plan: Ask LLM what to do
        let tool_calls = self.plan(input, ctx)?;

        if tool_calls.is_empty() {
            // No tools needed, direct response
            return self.ask_llm_direct(input);
        }

        // 2 & 3. Act: Execute tools
        let mut tool_outputs = Vec::new();
        for call in tool_calls {
            let result = self.execute_tool(&call, ctx)?;
            tool_outputs.push((call.name, result));
        }

        // 4. Review: Synthesize results
        self.synthesize(input, &tool_outputs)
    }

    /// Plan: Ask LLM which tools to call
    fn plan(&self, input: &str, _ctx: &TurnContext) -> Result<Vec<ToolCall>> {
        let full_prompt = format!(
            "You are Luna, an expert code assistant. Analyze the user request and decide what tools to use.\n\n\
            USER REQUEST: {}\n\n\
            === AVAILABLE TOOLS ===\n\
            1. list_dir - List directory contents\n\
               Use: Exploring project structure, finding files\n\
               Args: {{\"path\": \"relative/path\"}}\n\n\
            2. read_file - Read file contents\n\
               Use: Reading source code, configs, documentation\n\
               Args: {{\"path\": \"relative/path\"}}\n\n\
            3. search_code - Search for symbol definitions\n\
               Use: Finding functions, structs, traits, classes by name\n\
               Args: {{\"query\": \"symbol_name\"}}\n\
               ⚠️ CRITICAL: Use this FIRST when user asks about a specific symbol!\n\n\
            4. edit_file - Edit a file\n\
               Use: Modifying code\n\
               Args: {{\"path\": \"...\", \"old_string\": \"...\", \"new_string\": \"...\"}}\n\n\


            5. run_terminal - Run a terminal command

               Use: Running tests, checking git status, building project

               Args: {{\"cmd\": \"command string\"}}

               ⚠️ Note: Dangerous commands are blocked (rm -rf /, mkfs, etc.)



            === DECISION RULES ===
            RULE 1: If user asks about a SPECIFIC SYMBOL (function, struct, trait, class, variable):
                    → FIRST use search_code to find its location\n\
                    → THEN use read_file to view the content\n\
                    Example: \"Where is parse_tool_calls?\" → search_code {{\"query\": \"parse_tool_calls\"}}\n\n\
            RULE 2: If user asks about OVERALL PROJECT STRUCTURE:\n\
                    → Use list_dir('.') to see top-level\n\
                    → Use read_file on README.md, Cargo.toml, package.json\n\
                    → Then read main entry files\n\n\
            RULE 3: If user asks about a SPECIFIC FILE:\n\
                    → Use read_file directly\n\n\
            === RESPONSE FORMAT ===\n\
            Respond with JSON:\n\
            {{\n\
              \"thought\": \"Brief analysis of what the user wants\",\n\
              \"tool_calls\": [\n\
                {{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n\
              ]\n\
            }}\n\n\
            === EXAMPLES ===\n\
            Example 1 - User: \"Find parse_tool_calls function\"\n\
            {{\n\
              \"thought\": \"User wants to find a specific function, I should search for it first\",\n\
              \"tool_calls\": [{{\"name\": \"search_code\", \"arguments\": {{\"query\": \"parse_tool_calls\"}}}}]\n\
            }}\n\n\
            Example 2 - User: \"Analyze this project\"\n\
            {{\n\
              \"thought\": \"User wants project overview, I should explore structure and key files\",\n\
              \"tool_calls\": [\n\
                {{\"name\": \"list_dir\", \"arguments\": {{\"path\": \".\"}}}},\n\
                {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"README.md\"}}}},\n\
                {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"Cargo.toml\"}}}}\n\
              ]\n\
            }}\n\n\
            Example 3 - User: \"Show me src/main.rs\"\n\
            {{\n\
              \"thought\": \"User wants to see a specific file\",\n\
              \"tool_calls\": [{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"src/main.rs\"}}}}]\n\
            }}\n\n\
            Now analyze the user request and respond with JSON:",
            input
        );

        let response = self.llm.complete(CompletionRequest { prompt: full_prompt })?;
        parse_tool_calls(&response.content)
    }

    /// Execute a tool
    fn execute_tool(&self, call: &ToolCall, ctx: &TurnContext) -> Result<ToolResult> {
        self.tools.execute(call, ctx)
    }

    /// Synthesize final response
    fn synthesize(
        &self,
        original_input: &str,
        tool_outputs: &[(String, ToolResult)],
    ) -> Result<String> {
        let mut context = format!(
            "User request: {}\n\nTool execution results:\n",
            original_input
        );

        for (name, result) in tool_outputs {
            context.push_str(&format!(
                "\n{}: {}\n{}",
                name,
                if result.success { "✓" } else { "✗" },
                result.output
            ));
        }

        // Check if we actually have file contents or just directory listings
        let has_file_content = tool_outputs.iter().any(|(_, r)| {
            r.success && (r.output.contains("File:") || r.output.contains("// File:"))
        });

        let additional_instructions = if !has_file_content {
            "\n\n⚠️ WARNING: You only have directory listings but no actual file contents! \
            Your response should say: 'I need to read the actual file contents to provide a detailed analysis. \
            Please let me read the key files first.'"
        } else {
            ""
        };

        let prompt = format!(
            "You are Luna, an expert code assistant. Analyze the information gathered and provide a comprehensive, insightful response.\n\n\
            INSTRUCTIONS:\n\
            1. Analyze the tool results carefully\n\
            2. Identify the project's purpose, architecture, and key components\n\
            3. Highlight interesting patterns, design decisions, or notable features\n\
            4. Structure your answer with clear sections\n\
            5. Be specific - cite file names, functions, or code patterns you observed\n\
            6. If analyzing a project, cover: what it does, how it's structured, key technologies\n\
            7. ONLY make claims based on actual file contents you see in the context\n\
            8. If the context lacks specific details, admit it and suggest what files to read next{}\n\n\
            Context from tool execution:\n{}\n\n\
            Provide a detailed, well-structured response in the same language as the user's request.\n\
            Format your response with markdown-style headers and bullet points where appropriate.",
            additional_instructions, context
        );

        let response = self.llm.complete(CompletionRequest { prompt })?;
        Ok(response.content)
    }

    /// Direct LLM query (when no tools needed)
    fn ask_llm_direct(&self, input: &str) -> Result<String> {
        let prompt = format!(
            "You are Luna, an expert code assistant.\n\n\
             User question: {}\n\n\
             Provide a helpful, accurate response. If this question requires exploring code or files, \
             suggest that the user asks in a way that would trigger file analysis (e.g., '查看文件内容' or '分析这个项目').\n\n\
             Assistant:",
            input
        );
        let response = self.llm.complete(CompletionRequest { prompt })?;
        Ok(response.content)
    }
}

/// Context for a turn
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub session_id: String,
    pub repo_root: Option<PathBuf>,
    pub cwd: Option<PathBuf>,
}

impl TurnContext {
    #[must_use]
    pub fn new(repo_root: Option<PathBuf>) -> Self {
        Self {
            session_id: "default".to_owned(),
            repo_root: repo_root.clone(),
            cwd: repo_root,
        }
    }
}

/// Safety guard trait
pub trait SafetyGuard: Send + Sync {
    fn check(&self, tool_name: &str, args: &serde_json::Map<String, serde_json::Value>) -> Result<()>;
}

/// Default safety guard implementation
pub struct DefaultSafetyGuard;

impl DefaultSafetyGuard {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultSafetyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyGuard for DefaultSafetyGuard {
    fn check(&self, tool_name: &str, args: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        if tool_name == "run_terminal" {
            if let Some(cmd) = args.get("cmd").and_then(|v| v.as_str()) {
                let dangerous_patterns = [
                    "rm -rf /",
                    "rm -rf /*",
                    "rm -rf ~",
                    "mkfs",
                    "dd if=/dev/zero",
                    ":(){ :|:& };:", // fork bomb
                    "> /dev/sda",
                    "curl", // could be pipe to shell
                    "wget", // could download malicious script
                ];

                for pattern in &dangerous_patterns {
                    if cmd.contains(pattern) {
                        return Err(LunaError::invalid_input(
                            format!("Command blocked by safety guard: contains dangerous pattern '{}'", pattern)
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Tool registry with safety guard
pub struct ToolRegistry {
    handlers: HashMap<String, Box<dyn ToolHandler>>,
    safety: Box<dyn SafetyGuard>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            handlers: HashMap::new(),
            safety: Box::new(DefaultSafetyGuard::new()),
        };
        registry.register_default_tools();
        registry
    }

    pub fn with_safety(guard: Box<dyn SafetyGuard>) -> Self {
        let mut registry = Self {
            handlers: HashMap::new(),
            safety: guard,
        };
        registry.register_default_tools();
        registry
    }

    pub fn register(&mut self, name: &str, handler: Box<dyn ToolHandler>) {
        self.handlers.insert(name.to_owned(), handler);
    }

    pub fn execute(&self, call: &ToolCall, ctx: &TurnContext) -> Result<ToolResult> {
        // Safety check first
        self.safety.check(&call.name, &call.arguments)?;

        let handler = self.handlers.get(&call.name).ok_or_else(|| {
            LunaError::invalid_input(format!("Unknown tool: {}", call.name))
        })?;
        handler.execute(&call.arguments, ctx)
    }

    fn register_default_tools(&mut self) {
        self.register("read_file", Box::new(ReadFileTool));
        self.register("edit_file", Box::new(EditFileTool));
        self.register("list_dir", Box::new(ListDirTool));
        self.register("search_code", Box::new(SearchCodeTool));
        self.register("run_terminal", Box::new(RunTerminalTool));
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool handler trait
trait ToolHandler: Send + Sync {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult>;
}

/// Read file tool
struct ReadFileTool;

impl ToolHandler for ReadFileTool {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult> {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'path' argument. Please provide a file path to read.".to_string(),
                });
            }
        };

        let full_path = resolve_path(path, ctx);

        match std::fs::read_to_string(&full_path) {
            Ok(content) => Ok(ToolResult {
                success: true,
                output: format!("// File: {} ({} lines)\n{}", path, content.lines().count(), content),
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Error reading file '{}': {}", path, e),
            }),
        }
    }
}

/// Edit file tool - replaces old_string with new_string
struct EditFileTool;

impl ToolHandler for EditFileTool {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult> {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'path' argument. Please provide a file path to edit.".to_string(),
                });
            }
        };

        let old_string = match args.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'old_string' argument. Please provide the text to replace.".to_string(),
                });
            }
        };

        let new_string = match args.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'new_string' argument. Please provide the replacement text.".to_string(),
                });
            }
        };

        let full_path = resolve_path(path, ctx);

        // Read current content
        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: format!("Error reading file '{}': {}", path, e),
                });
            }
        };

        // Check if old_string exists in content
        if !content.contains(old_string) {
            return Ok(ToolResult {
                success: false,
                output: format!(
                    "Error: Could not find the specified text in file '{}'. \
                    The 'old_string' must match exactly (including whitespace). \
                    Hint: Use read_file first to see the exact content.",
                    path
                ),
            });
        }

        // Count occurrences
        let count = content.matches(old_string).count();
        if count > 1 {
            return Ok(ToolResult {
                success: false,
                output: format!(
                    "Error: Found {} occurrences of the text in file '{}'. \
                    'old_string' must be unique. \
                    Hint: Include more context to make it unique (e.g., surrounding lines).",
                    count, path
                ),
            });
        }

        // Perform replacement
        let new_content = content.replacen(old_string, new_string, 1);

        // Write back
        match std::fs::write(&full_path, new_content) {
            Ok(_) => Ok(ToolResult {
                success: true,
                output: format!(
                    "✓ Successfully edited file: {}\n\
                     Replaced:\n\
                     ---\n{}\n\
                     ---\n\
                     With:\n\
                     ---\n{}\n\
                     ---",
                    path, old_string, new_string
                ),
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Error writing file '{}': {}", path, e),
            }),
        }
    }
}

/// List directory tool
struct ListDirTool;

impl ToolHandler for ListDirTool {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let full_path = resolve_path(path, ctx);

        match std::fs::read_dir(&full_path) {
            Ok(entries) => {
                let mut items: Vec<String> = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    items.push(format!("{}{}", name, if is_dir { "/" } else { "" }));
                }
                items.sort();
                Ok(ToolResult {
                    success: true,
                    output: format!("Directory: {}\n{}", path, items.join("\n")),
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Error listing directory '{}': {}", path, e),
            }),
        }
    }
}

/// Search code tool using ScopeGraph
struct SearchCodeTool;

impl ToolHandler for SearchCodeTool {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult> {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'query' argument. Please provide a symbol name to search.".to_string(),
                });
            }
        };

        // Get repo root from context
        let repo_root = match &ctx.cwd {
            Some(root) => root.clone(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: No working directory set. Cannot search for symbols.".to_string(),
                });
            }
        };

        // Create navigator with default options
        let provider = FsRepoFileProvider;
        let scan_opt = RepoScanOptions::default();
        let navigator = TreeSitterNavigator::new(provider, scan_opt);

        // Search for symbol
        match navigator.search_symbol(&repo_root, query) {
            Ok(result) => {
                if result.definitions.is_empty() {
                    Ok(ToolResult {
                        success: true,
                        output: format!("No definitions found for '{}'", query),
                    })
                } else {
                    let mut output = format!("Found definitions for '{}':\n\n", query);
                    for (i, loc) in result.definitions.iter().enumerate() {
                        output.push_str(&format!(
                            "{}. {} (line {}-{})\n",
                            i + 1,
                            loc.rel_path.display(),
                            loc.range.start.line,
                            loc.range.end.line
                        ));
                    }
                    output.push_str(&format!("\nUse 'read_file' to view the full content."));
                    Ok(ToolResult {
                        success: true,
                        output,
                    })
                }
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Error searching for '{}': {}", query, e),
            }),
        }
    }
}

/// Run terminal command tool
struct RunTerminalTool;

impl ToolHandler for RunTerminalTool {
    fn execute(
        &self,
        args: &serde_json::Map<String, serde_json::Value>,
        ctx: &TurnContext,
    ) -> Result<ToolResult> {
        let cmd = match args.get("cmd").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: "Error: Missing 'cmd' argument. Please provide a command to execute.".to_string(),
                });
            }
        };

        // Get working directory
        let cwd = ctx.cwd.clone().unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        // Execute command
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&cwd)
            .output();

        match output {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                let success = result.status.success();

                let mut output = String::new();
                if !stdout.is_empty() {
                    output.push_str(&format!("STDOUT:\n{}\n", stdout));
                }
                if !stderr.is_empty() {
                    output.push_str(&format!("STDERR:\n{}\n", stderr));
                }
                if stdout.is_empty() && stderr.is_empty() {
                    output.push_str("(no output)");
                }

                Ok(ToolResult {
                    success,
                    output: output.trim().to_string(),
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: format!("Error executing command: {}", e),
            }),
        }
    }
}

/// Resolve relative path to absolute
fn resolve_path(path: &str, ctx: &TurnContext) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else if let Some(cwd) = &ctx.cwd {
        cwd.join(p)
    } else {
        p.to_path_buf()
    }
}

/// Parse LLM response for tool calls
fn parse_tool_calls(content: &str) -> Result<Vec<ToolCall>> {
    let json_str = extract_json(content);

    #[derive(Deserialize)]
    struct Response {
        #[allow(dead_code)]
        thought: String,
        tool_calls: Vec<ToolCall>,
    }

    match serde_json::from_str::<Response>(&json_str) {
        Ok(resp) => Ok(resp.tool_calls),
        Err(_) => {
            // If parsing fails, treat as direct response (no tools)
            Ok(Vec::new())
        }
    }
}

/// Extract JSON from markdown or raw string
fn extract_json(content: &str) -> String {
    let trimmed = content.trim();

    // Try markdown code blocks
    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed[start + 7..].find("```") {
            return trimmed[start + 7..start + 7 + end].trim().to_owned();
        }
    }

    // Try plain code blocks
    if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed[start + 3..].find("```") {
            let inner = trimmed[start + 3..start + 3 + end].trim();
            if let Some(json_start) = inner.find('{') {
                return inner[json_start..].to_owned();
            }
        }
    }

    // Find JSON object directly
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_owned();
        }
    }

    trimmed.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json() {
        let content = r#"```json
{"thought": "test", "tool_calls": []}
```"#;
        assert_eq!(
            extract_json(content),
            r#"{"thought": "test", "tool_calls": []}"#
        );
    }

    #[test]
    fn test_extract_json_raw() {
        let content = r#"{"thought": "test", "tool_calls": []}"#;
        assert_eq!(
            extract_json(content),
            r#"{"thought": "test", "tool_calls": []}"#
        );
    }

    #[test]
    fn test_extract_json_with_newlines() {
        let content = r#"```json
{
  "thought": "test",
  "tool_calls": []
}
```"#;
        assert_eq!(
            extract_json(content),
            r#"{
  "thought": "test",
  "tool_calls": []
}"#
        );
    }

    #[test]
    fn test_extract_json_no_markdown() {
        // LLM might respond without markdown
        let content = r#"Sure! Let me help you.

{"thought": "User wants to read a file", "tool_calls": [{"name": "read_file", "arguments": {"path": "src/main.rs"}}]}"#;
        let result = extract_json(content);
        assert!(result.contains("tool_calls"));
    }

    #[test]
    fn test_parse_tool_calls_valid() {
        let content = r#"{"thought": "test", "tool_calls": [{"name": "read_file", "arguments": {"path": "src/main.rs"}}]}"#;
        let calls = parse_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(
            calls[0].arguments.get("path").unwrap().as_str().unwrap(),
            "src/main.rs"
        );
    }

    #[test]
    fn test_parse_tool_calls_empty() {
        let content = r#"{"thought": "direct response", "tool_calls": []}"#;
        let calls = parse_tool_calls(content).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_tool_calls_invalid_json() {
        // Invalid JSON should return empty vec (fallback to direct response)
        let content = r#"This is not JSON"#;
        let calls = parse_tool_calls(content).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_tool_calls_multiple_tools() {
        let content = r#"{"thought": "analyze project", "tool_calls": [
            {"name": "list_dir", "arguments": {"path": "."}},
            {"name": "read_file", "arguments": {"path": "Cargo.toml"}}
        ]}"#;
        let calls = parse_tool_calls(content).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "list_dir");
        assert_eq!(calls[1].name, "read_file");
    }

    #[test]
    fn test_resolve_path_absolute() {
        let ctx = TurnContext::new(Some(std::path::PathBuf::from("/home/user")));
        let path = resolve_path("/absolute/path", &ctx);
        assert_eq!(path, std::path::PathBuf::from("/absolute/path"));
    }

    #[test]
    fn test_resolve_path_relative() {
        let ctx = TurnContext::new(Some(std::path::PathBuf::from("/home/user")));
        let path = resolve_path("src/main.rs", &ctx);
        assert_eq!(path, std::path::PathBuf::from("/home/user/src/main.rs"));
    }
}
