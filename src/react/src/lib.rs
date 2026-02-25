//! ReAct Loop Implementation
//!
//! This crate implements the ReAct (Reasoning + Acting) loop pattern for agent execution.
//!
//! The ReAct loop follows the pattern:
//! 1. Think: LLM analyzes current state and decides next action
//! 2. Act: Execute the chosen tool/action
//! 3. Observe: Collect results and update state
//! 4. Repeat until completion
//!
//! Design Principles:
//! - Clear separation between planning and execution
//! - Tool-agnostic: easy to add new tools
//! - State tracking for explainability
//! - Configurable loop behavior

pub mod agent;
pub mod context;
pub mod planner;

pub use agent::{react_ask, ReactAgent, ReactOptions};
pub use context::{render_prompt_context, ContextEngineOptions};
pub use planner::{ReActAction, ReActStepTrace};

// Re-export common types
pub use llm::LLMConfig;
pub use tools::ContextPack;

use tokenizers::Tokenizer;
use toolkit::{
    EditFileTool, ExecutionPolicy, ListDirTool, ReadFileTool, RunTerminalTool, ToolInput,
};
use toolkit::{ToolOutput, ToolRegistry, ToolSchema};
use tools::SearchCodeOptions;

use core::code_chunk::{ContextChunk, IndexChunk};
use std::collections::BTreeMap;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of context chunks to show in state summaries
const STATE_CONTEXT_PREVIEW_MAX: usize = 6;

// ============================================================================
// State Summary
// ============================================================================

/// Summarize the current state for the LLM planner
///
/// This provides a concise view of:
/// - Number of hits and context chunks
/// - Whether definitions were found
/// - Preview of context chunks
pub fn summarize_state(hits: &[IndexChunk], context: &[ContextChunk]) -> String {
    let definition_names: Vec<String> =
        context.iter().filter_map(extract_definition_name).collect();

    let has_definition = !definition_names.is_empty();
    let mut s = String::new();
    s.push_str(&format!(
        "hits={} context_chunks={} has_definition={}",
        hits.len(),
        context.len(),
        has_definition
    ));

    if !definition_names.is_empty() {
        s.push_str(&format!(" definitions=[{}]", definition_names.join(", ")));
    }

    // Show specific functions if visible
    let has_list_dir = context
        .iter()
        .any(|c| c.snippet.contains("fn list_dir") || c.snippet.contains("pub fn list_dir"));
    let has_sort = context.iter().any(|c| {
        c.snippet.contains("entries.sort_by") || c.snippet.contains("entries.sort_by_key")
    });
    if has_list_dir {
        s.push_str(&format!(
            " visible_functions=[list_dir] has_sort={}",
            has_sort
        ));
    }
    s.push('\n');

    for c in context.iter().take(STATE_CONTEXT_PREVIEW_MAX) {
        let is_def = is_definition_chunk(c);
        let def_name = extract_definition_name(c);
        s.push_str(&format!(
            "- {}:{}..={}{}{} reason={}\n",
            c.path,
            c.start_line + 1,
            c.end_line + 1,
            if is_def { " [def]" } else { "" },
            if let Some(name) = def_name {
                format!(" ({})", name)
            } else {
                String::new()
            },
            c.reason,
        ));
    }
    if context.len() > STATE_CONTEXT_PREVIEW_MAX {
        s.push_str("- ...\n");
    }
    s
}

// ============================================================================
// Definition Extraction
// ============================================================================

/// Extract the name of a definition from a ContextChunk
fn extract_definition_name(chunk: &ContextChunk) -> Option<String> {
    let snippet = chunk.snippet.trim();
    let snippet_lower = snippet.to_lowercase();

    let patterns = [
        ("pub struct ", "struct "),
        ("pub enum ", "enum "),
        ("pub fn ", "fn "),
        ("trait ", "trait "),
        ("type ", "type "),
    ];

    for (full_pattern, short_pattern) in patterns {
        if let Some(pos) = snippet_lower.find(full_pattern) {
            let after = snippet[pos + full_pattern.len()..].trim();
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
        if let Some(pos) = snippet_lower.find(short_pattern) {
            let after = snippet[pos + short_pattern.len()..].trim();
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }

    None
}

/// Check if a ContextChunk appears to contain a type/function definition
fn is_definition_chunk(chunk: &ContextChunk) -> bool {
    let snippet_lower = chunk.snippet.to_lowercase();
    snippet_lower.contains("pub struct")
        || snippet_lower.contains("struct ")
        || snippet_lower.contains("pub enum")
        || snippet_lower.contains("enum ")
        || snippet_lower.contains("pub fn")
        || snippet_lower.contains("fn ")
        || snippet_lower.contains("impl ")
        || snippet_lower.contains("trait ")
        || snippet_lower.contains("type ")
}

// ============================================================================
// Hit Merging
// ============================================================================

/// Merge hit lists, deduplicating by (path, start_byte, end_byte)
pub fn merge_hits(mut base: Vec<IndexChunk>, more: Vec<IndexChunk>) -> Vec<IndexChunk> {
    let mut uniq: BTreeMap<(String, usize, usize), IndexChunk> = BTreeMap::new();
    for h in base.drain(..).chain(more.into_iter()) {
        let key = (h.path.clone(), h.start_byte, h.end_byte);
        uniq.entry(key).or_insert(h);
    }
    uniq.into_values().collect()
}

// ============================================================================
// Runtime Facade (External Entry Point)
// ============================================================================

/// Luna Runtime Facade: Unified external entry point aggregating tool registry,
/// tokenizer, LLM configuration, and ReAct options.
///
/// Design goals:
/// - Future `luna-server (MCP)` depends only on this layer, no need to assemble crates manually.
/// - Allows swapping retrieval backends/strategies without affecting the external API.
pub struct LunaRuntime {
    registry: ToolRegistry,
    policy: ExecutionPolicy,
    tokenizer: Tokenizer,
    llm_cfg: LLMConfig,
    react_opt: ReactOptions,
}

impl LunaRuntime {
    pub fn new(
        tokenizer: Tokenizer,
        llm_cfg: LLMConfig,
        policy: ExecutionPolicy,
        opt: ReactOptions,
    ) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ReadFileTool::new()));
        registry.register(Box::new(ListDirTool::new()));
        registry.register(Box::new(EditFileTool::new()));
        registry.register(Box::new(RunTerminalTool::new()));

        Self {
            registry,
            policy,
            tokenizer,
            llm_cfg,
            react_opt: opt,
        }
    }

    pub fn policy(&self) -> &ExecutionPolicy {
        &self.policy
    }

    pub fn tool_schemas(&self) -> Vec<ToolSchema> {
        self.registry.schemas()
    }

    pub fn execute_tool(
        &self,
        name: &str,
        repo_root: std::path::PathBuf,
        args: serde_json::Value,
    ) -> ToolOutput {
        let input = ToolInput {
            args,
            repo_root,
            policy: Some(self.policy.clone()),
        };
        self.registry.execute(name, &input)
    }

    /// Answer question using ReAct (internally runs search/refill/context/answer)
    pub fn ask_react(
        &self,
        repo_root: &std::path::Path,
        question: &str,
    ) -> anyhow::Result<(String, ContextPack, Vec<ReActStepTrace>)> {
        agent::react_ask(
            repo_root,
            question,
            &self.tokenizer,
            &self.llm_cfg,
            self.react_opt.clone(),
        )
    }

    /// Directly expose the placeholder search entry point for server/MCP layer to do finer-grained tool decomposition
    pub fn search_code_keyword(
        &self,
        repo_root: &std::path::Path,
        query: &str,
        idx_opt: core::code_chunk::IndexChunkOptions,
        opt: SearchCodeOptions,
    ) -> anyhow::Result<(Vec<core::code_chunk::IndexChunk>, Vec<tools::ToolTrace>)> {
        tools::search_code_keyword(repo_root, query, &self.tokenizer, idx_opt, opt)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn refill_hits(
        &self,
        repo_root: &std::path::Path,
        hits: &[core::code_chunk::IndexChunk],
        opt: core::code_chunk::RefillOptions,
    ) -> anyhow::Result<(Vec<core::code_chunk::ContextChunk>, Vec<tools::ToolTrace>)> {
        tools::refill_hits(repo_root, hits, opt)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_definition_name_struct() {
        let chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "pub struct MyStruct { x: i32 }".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert_eq!(
            extract_definition_name(&chunk),
            Some("MyStruct".to_string())
        );
    }

    #[test]
    fn test_extract_definition_name_fn() {
        let chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "fn my_function() -> i32 { 42 }".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert_eq!(
            extract_definition_name(&chunk),
            Some("my_function".to_string())
        );
    }

    #[test]
    fn test_is_definition_chunk() {
        let def_chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "pub struct Test {}".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert!(is_definition_chunk(&def_chunk));

        let non_def_chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "let x = 42;".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert!(!is_definition_chunk(&non_def_chunk));
    }
}
