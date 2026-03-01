//! MCP Tools Implementation
//!
//! Exposes Luna capabilities as MCP (Model Context Protocol) tools:
//! - search_symbol: Find symbol definitions across the codebase
//! - read_context: Read file content with semantic context
//! - fix_compile_error: Automated compile error fixing (M3)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use std::str::FromStr;

use react::{run_fix_loop, LunaRuntime};
use llm::LLMConfig;
use tools::{read_file, CargoErrorParser, ErrorParserRegistry, SearchCodeOptions};
use core::code_chunk::IndexChunkOptions;

/// MCP Tool definition (for tools/list)
#[derive(Debug, Clone, Serialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// MCP TextContent for tool responses
#[derive(Debug, Clone, Serialize)]
pub struct McpTextContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// MCP Tool response
#[derive(Debug, Clone, Serialize)]
pub struct McpToolResponse {
    pub content: Vec<McpTextContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl McpToolResponse {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpTextContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpTextContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: Some(true),
        }
    }
}

/// List all available MCP tools
pub fn list_tools() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "search_symbol".to_string(),
            description: "Search for symbol definitions in the codebase. Finds functions, structs, enums, etc. by name.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to search for (e.g., 'my_function', 'MyStruct')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["symbol_name"]
            }),
        },
        McpTool {
            name: "read_context".to_string(),
            description: "Read a file with semantic context including related symbol definitions.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to repo root)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-based, optional)",
                        "default": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (optional)",
                        "default": null
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "Whether to include related symbol definitions",
                        "default": true
                    }
                },
                "required": ["file_path"]
            }),
        },
        McpTool {
            name: "fix_compile_error".to_string(),
            description: "Automatically fix compile errors by running build, analyzing errors, and applying fixes.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "build_command": {
                        "type": "string",
                        "description": "Build command to run (e.g., 'cargo build', 'npm run build')",
                        "default": "cargo build"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum fix iterations",
                        "default": 5
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, only analyze without applying fixes",
                        "default": false
                    }
                },
                "required": []
            }),
        },
        McpTool {
            name: "analyze_build_errors".to_string(),
            description: "Parse build output and return structured error information.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "build_command": {
                        "type": "string",
                        "description": "Build command to run",
                        "default": "cargo build"
                    }
                },
                "required": []
            }),
        },
    ]
}

/// Search symbol tool input
#[derive(Debug, Deserialize)]
pub struct SearchSymbolInput {
    pub symbol_name: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_max_results() -> usize {
    10
}

/// Execute search_symbol tool
pub fn execute_search_symbol(
    runtime: &LunaRuntime,
    repo_root: &Path,
    input: SearchSymbolInput,
) -> Result<McpToolResponse> {
    // Use the runtime's search capability
    let (hits, _trace) = runtime.search_code_keyword(
        repo_root,
        &input.symbol_name,
        IndexChunkOptions::default(),
        SearchCodeOptions {
            max_hits: input.max_results,
            ..Default::default()
        },
    )?;

    if hits.is_empty() {
        return Ok(McpToolResponse::text(format!(
            "No symbols found matching '{}'",
            input.symbol_name
        )));
    }

    let mut results = vec![format!(
        "Found {} symbol(s) matching '{}':\n",
        hits.len(),
        input.symbol_name
    )];

    for (i, hit) in hits.iter().enumerate() {
        results.push(format!(
            "[{}] {} (lines {}-{})",
            i + 1,
            hit.path,
            hit.start_line + 1,
            hit.end_line + 1
        ));
    }

    Ok(McpToolResponse::text(results.join("\n")))
}

/// Read context tool input
#[derive(Debug, Deserialize)]
pub struct ReadContextInput {
    pub file_path: String,
    #[serde(default = "default_start_line")]
    pub start_line: usize,
    #[serde(default)]
    pub end_line: Option<usize>,
    #[serde(default = "default_include_related")]
    pub include_related: bool,
}

fn default_start_line() -> usize {
    1
}

fn default_include_related() -> bool {
    true
}

/// Execute read_context tool
pub fn execute_read_context(
    _runtime: &LunaRuntime,
    repo_root: &Path,
    input: ReadContextInput,
) -> Result<McpToolResponse> {
    let full_path = repo_root.join(&input.file_path);

    // Read the file
    let content = if let Some(end) = input.end_line {
        read_file(&full_path,
            Some((input.start_line.saturating_sub(1), end.saturating_sub(1))),
        )?
    } else {
        read_file(&full_path, Some((input.start_line.saturating_sub(1), usize::MAX)))?
    };

    let mut response = format!(
        "File: {} (lines {}-{})\n```\n{}\n```",
        input.file_path,
        input.start_line,
        input.end_line.map(|e| e.to_string()).unwrap_or_else(|| "end".to_string()),
        content
    );

    // If include_related, search for symbols in this file
    if input.include_related {
        // Extract potential symbols from the content (simplified)
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|w| w.len() > 3 && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
            .collect();

        if !words.is_empty() {
            response.push_str("\n\nRelated symbols found (for full context, use search_symbol):\n");
            for word in words.iter().take(5) {
                response.push_str(&format!("- {}\n", word.trim_matches(|c: char| !c.is_alphanumeric())));
            }
        }
    }

    Ok(McpToolResponse::text(response))
}

/// Fix compile error tool input
#[derive(Debug, Deserialize)]
pub struct FixCompileErrorInput {
    #[serde(default = "default_build_command")]
    pub build_command: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default)]
    pub dry_run: bool,
}

fn default_build_command() -> String {
    "cargo build".to_string()
}

fn default_max_iterations() -> usize {
    5
}

/// Execute fix_compile_error tool
pub fn execute_fix_compile_error(
    _runtime: &LunaRuntime,
    repo_root: &Path,
    input: FixCompileErrorInput,
) -> Result<McpToolResponse> {
    if input.dry_run {
        return Ok(McpToolResponse::text(format!(
            "Dry run mode: Would attempt to fix compile errors using:\n\
             - Build command: {}\n\
             - Max iterations: {}\n\
             Use dry_run: false to actually apply fixes.",
            input.build_command,
            input.max_iterations
        )));
    }

    // Try to get LLM configuration from environment
    let llm_config = match LLMConfig::from_env() {
        Ok(cfg) => cfg,
        Err(e) => {
            return Ok(McpToolResponse::error(format!(
                "LLM configuration error: {}\n\n\
                 Please set the following environment variables:\n\
                 - LLM_API_KEY: Your API key (required)\n\
                 - LLM_API_BASE: API base URL (optional, default: https://open.bigmodel.cn/api/paas/v4/)\n\
                 - LLM_MODEL: Model name (optional, default: glm-4-flash)\n\n\
                 Or use dry_run: true to see what would be fixed.",
                e
            )));
        }
    };

    // Run the actual fix loop
    use tokenizers::Tokenizer;

    // TODO: Use a real tokenizer or make it optional
    let tokenizer = Tokenizer::from_str(
        r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null, "model": {"type": "BPE", "vocab": {}, "merges": []}}"#,
    ).map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))?;

    match run_fix_loop(repo_root, &input.build_command, &tokenizer, &llm_config) {
        Ok(result) => {
            let status = if result.converged {
                "✅ Converged successfully"
            } else {
                "⚠️ Did not converge"
            };

            let mut response = format!(
                "{status}\n\n\
                 Final error count: {}\n\
                 Iterations: {}\n\
                 Modified files: {}\n",
                result.final_error_count,
                result.iterations.len(),
                result.modified_files.len()
            );

            if !result.modified_files.is_empty() {
                response.push_str("\nModified files:\n");
                for file in &result.modified_files {
                    response.push_str(&format!("  - {}\n", file));
                }
            }

            Ok(McpToolResponse::text(response))
        }
        Err(e) => Ok(McpToolResponse::error(format!(
            "FixLoop execution failed: {}\n\n\
             Note: This is an MVP implementation. Full error recovery may require manual intervention.",
            e
        ))),
    }
}

/// Analyze build errors tool input
#[derive(Debug, Deserialize)]
pub struct AnalyzeBuildErrorsInput {
    #[serde(default = "default_build_command")]
    pub build_command: String,
}

/// Execute analyze_build_errors tool
pub fn execute_analyze_build_errors(
    _runtime: &LunaRuntime,
    repo_root: &Path,
    input: AnalyzeBuildErrorsInput,
) -> Result<McpToolResponse> {
    use tools::run_terminal;

    // Run the build command
    let result = run_terminal(
        &input.build_command,
        Some(repo_root),
        false,
    )?;

    if result.success {
        return Ok(McpToolResponse::text(
            "Build succeeded! No errors found.".to_string()
        ));
    }

    // Parse errors - register CargoErrorParser for Rust projects
    let mut parser = ErrorParserRegistry::new();
    parser.register("cargo build", CargoErrorParser::new());
    parser.register("cargo test", CargoErrorParser::new());
    parser.register("cargo check", CargoErrorParser::new());

    let output = if result.stderr.is_empty() {
        &result.stdout
    } else {
        &result.stderr
    };

    let errors = parser.parse(&input.build_command,
        output,
        result.exit_code,
    );

    if errors.is_empty() {
        return Ok(McpToolResponse::text(
            "Build failed but no parseable errors found. Raw output:\n".to_string() + output
        ));
    }

    let mut response = format!("Found {} error(s):\n\n", errors.len());

    for (i, err) in errors.iter().enumerate() {
        response.push_str(&format!(
            "[{}] {}: {}\n",
            i + 1,
            err.error_code.as_deref().unwrap_or("ERROR"),
            err.message
        ));

        if let Some(loc) = err.locations.first() {
            response.push_str(&format!(
                "    at {}:{}\n",
                loc.path,
                loc.line
            ));
        }

        if let Some(suggestion) = &err.suggestion {
            response.push_str(&format!("    suggestion: {}\n", suggestion));
        }

        response.push('\n');
    }

    Ok(McpToolResponse::text(response))
}

/// Dispatch MCP tool calls
pub fn dispatch_tool(
    tool_name: &str,
    params: &serde_json::Value,
    runtime: &LunaRuntime,
    repo_root: &Path,
) -> Result<McpToolResponse> {
    match tool_name {
        "search_symbol" => {
            let input: SearchSymbolInput = serde_json::from_value(params.clone())?;
            execute_search_symbol(runtime, repo_root, input)
        }
        "read_context" => {
            let input: ReadContextInput = serde_json::from_value(params.clone())?;
            execute_read_context(runtime, repo_root, input)
        }
        "fix_compile_error" => {
            let input: FixCompileErrorInput = serde_json::from_value(params.clone())?;
            execute_fix_compile_error(runtime, repo_root, input)
        }
        "analyze_build_errors" => {
            let input: AnalyzeBuildErrorsInput = serde_json::from_value(params.clone())?;
            execute_analyze_build_errors(runtime, repo_root, input)
        }
        _ => Ok(McpToolResponse::error(format!(
            "Unknown tool: {}. Available tools: search_symbol, read_context, fix_compile_error, analyze_build_errors",
            tool_name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_tools() {
        let tools = list_tools();
        assert_eq!(tools.len(), 4);

        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_symbol"));
        assert!(names.contains(&"read_context"));
        assert!(names.contains(&"fix_compile_error"));
        assert!(names.contains(&"analyze_build_errors"));
    }

    #[test]
    fn test_mcp_tool_response() {
        let resp = McpToolResponse::text("Hello");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].text, "Hello");
        assert!(resp.is_error.is_none());

        let err = McpToolResponse::error("Failed");
        assert_eq!(err.is_error, Some(true));
    }
}
