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

/// Definition patterns for extracting names from code snippets
///
/// Each pattern contains:
/// - prefix: the keyword pattern to match (e.g., "pub fn ")
/// - skip_generic: whether to skip generic parameters after the name
struct DefPattern {
    prefixes: &'static [&'static str],
    skip_generic: bool,
}

static DEF_PATTERNS: &[DefPattern] = &[
    // Rust-style definitions
    DefPattern {
        prefixes: &["pub struct ", "struct "],
        skip_generic: true, // struct Foo<T> { ... }
    },
    DefPattern {
        prefixes: &["pub enum ", "enum "],
        skip_generic: true, // enum Foo<T> { ... }
    },
    DefPattern {
        prefixes: &["pub fn ", "fn ", "async fn ", "pub async fn "],
        skip_generic: true, // fn foo<T>() { ... }
    },
    DefPattern {
        prefixes: &["pub trait ", "trait "],
        skip_generic: true,
    },
    DefPattern {
        prefixes: &["pub type ", "type "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub const ", "const "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub static ", "static "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub impl ", "impl "],
        skip_generic: true, // impl<T> Foo<T> { ... }
    },
    // C-style definitions
    DefPattern {
        prefixes: &["class ", "public class ", "private class ", "protected class "],
        skip_generic: true,
    },
    DefPattern {
        prefixes: &["def "], // Python
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["function ", "export function "], // JavaScript/TypeScript
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["func "], // Go
        skip_generic: false,
    },
];

/// Extract the name of a definition from a ContextChunk
///
/// Uses AST-aware pattern matching to identify definition names
/// without requiring a full parse.
fn extract_definition_name(chunk: &ContextChunk) -> Option<String> {
    let snippet = chunk.snippet.trim_start();

    for pattern in DEF_PATTERNS {
        for &prefix in pattern.prefixes {
            if let Some(after_prefix) = snippet.strip_prefix(prefix) {
                return Some(extract_identifier(after_prefix, pattern.skip_generic));
            }
        }
    }

    None
}

/// Extract an identifier from the start of a string
///
/// Handles:
/// - Generic parameters: `Foo<T, U>` extracts `Foo`
/// - Method receivers: `fn foo(&self)` extracts `foo`
/// - Qualified names: `impl Foo for Bar` extracts `Foo`
fn extract_identifier(s: &str, skip_generic: bool) -> String {
    let s = s.trim_start();

    // Find the end of the identifier
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            end = i + c.len_utf8();
        } else if c == '<' && skip_generic {
            // Found generic parameter start, stop here
            break;
        } else if c == '(' || c == '{' || c == ':' || c == ' ' || c == '<' {
            // End of identifier
            break;
        } else {
            // Skip other characters (like & for self)
            break;
        }
    }

    s[..end].to_string()
}

/// Check if a ContextChunk appears to contain a type/function definition
///
/// Uses the same pattern set as `extract_definition_name` for consistency.
fn is_definition_chunk(chunk: &ContextChunk) -> bool {
    let snippet = chunk.snippet.trim_start();

    DEF_PATTERNS.iter().any(|pattern| {
        pattern
            .prefixes
            .iter()
            .any(|&prefix| snippet.starts_with(prefix))
    })
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

/// Luna 运行时门面：收敛对外入口，统一聚合工具注册表 + tokenizer + LLM 配置 + ReAct options。
///
/// 设计目标：
/// - 未来 `luna-server (MCP)` 只依赖这一层，不需要直接拼装各个 crate。
/// - 允许后续替换检索后端/策略边界，而不影响对外 API。
pub struct LunaRuntime {
    registry: ToolRegistry,
    policy: ExecutionPolicy,
    tokenizer: Tokenizer,
    llm_cfg: LLMConfig,
    react_opt: ReactOptions,
}

impl LunaRuntime {
    /// 创建默认 runtime，并注册基础工具。
    ///
    /// 说明：
    /// - `search_code/refill` 仍通过 `tools`/`react` 内部调用，不在这里暴露为 ToolRegistry 工具；
    ///   MCP 版本可以在 server 层决定如何把它们暴露为 tools。
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

    /// 以 ReAct 方式回答问题（内部会走 search/refill/context/answer）。
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

    /// 直接暴露“占位检索”的调用点，便于 server/MCP 层做更细粒度的工具拆分。
    pub fn search_code_keyword(
        &self,
        repo_root: &std::path::Path,
        query: &str,
        idx_opt: core::code_chunk::IndexChunkOptions,
        opt: tools::SearchCodeOptions,
    ) -> anyhow::Result<(Vec<core::code_chunk::IndexChunk>, Vec<tools::ToolTrace>)> {
        Ok(tools::search_code_keyword(
            repo_root,
            query,
            &self.tokenizer,
            idx_opt,
            opt,
        )?)
    }

    pub fn refill_hits(
        &self,
        repo_root: &std::path::Path,
        hits: &[core::code_chunk::IndexChunk],
        opt: core::code_chunk::RefillOptions,
    ) -> anyhow::Result<(Vec<core::code_chunk::ContextChunk>, Vec<tools::ToolTrace>)> {
        Ok(tools::refill_hits(repo_root, hits, opt)?)
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
