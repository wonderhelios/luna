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
pub mod state;

pub use agent::{react_ask, ReactAgent, ReactOptions};
pub use context::{render_prompt_context, ContextEngineOptions};
pub use planner::{ReActAction, ReActStepTrace};
pub use state::{summarize_state, summarize_state_enhanced, SymbolInfo};

// Re-export common types
pub use llm::LLMConfig;
pub use tools::ContextPack;

use tokenizers::Tokenizer;
use toolkit::{
    EditFileTool, ExecutionPolicy, ListDirTool, ReadFileTool, RunTerminalTool, ToolInput,
};
use toolkit::{ToolOutput, ToolRegistry, ToolSchema};

// ============================================================================
// Runtime Facade (External Entry Point)
// ============================================================================

/// Luna runtime facade: unified entry point aggregating tool registry + tokenizer + LLM config + ReAct options.
///
/// Design goals:
/// - Future `luna-server (MCP)` only depends on this layer, no need to directly assemble individual crates.
/// - Allows subsequent replacement of retrieval backend/strategy boundaries without affecting external API.
pub struct LunaRuntime {
    registry: ToolRegistry,
    policy: ExecutionPolicy,
    tokenizer: Tokenizer,
    llm_cfg: LLMConfig,
    react_opt: ReactOptions,
}

impl LunaRuntime {
    /// Create default runtime and register basic tools.
    ///
    /// Note:
    /// - `search_code/refill` are still called internally through `tools`/`react`, not exposed as ToolRegistry tools here;
    ///   MCP version can decide how to expose them as tools at the server layer.
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

    /// Answer question in ReAct way (internally goes through search/refill/context/answer).
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

    /// Directly expose the "placeholder retrieval" call point for server/MCP layer to do finer-grained tool splitting.
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
