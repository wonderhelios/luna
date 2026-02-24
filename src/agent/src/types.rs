use core::code_chunk::{ContextChunk, IndexChunk};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolName {
    ReadFile,
    SearchCode,
    RefillChunks,
    ListSymbols,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTrace {
    pub tool: ToolName,
    pub summary: String,
}

// Minimal context pack for LLM/frontend (Context Engine input/output carrier)
pub struct ContextPack {
    pub query: String,
    // Retrieval hit protocol, to be replaced with vector/hybrid retrieval
    pub hits: Vec<IndexChunk>,
    // Readable context entering prompt (refilled function/class-level context)
    pub context: Vec<ContextChunk>,
    // Tool call trace (for debugging/explainability)
    pub trace: Vec<ToolTrace>,
}

/// LLm Config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
}

/// LLM plan output (ReAct's Act)
/// 约定：LLM 必须输出一个 JSON object（允许被 ``` 包裹），结构如下：
/// - 搜索：{"action":"search","query":"index_chunks"}
/// - 直接回答：{"action":"answer"}
/// - 停止：{"action":"stop","reason":"..."}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum ReActAction {
    Search { query: String },
    Answer,
    Stop { reason: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStepTrace {
    pub step: usize,
    pub plan_raw: String,
    pub action: Option<ReActAction>,
    pub observation: String,
}

/// Context Engine: fixed N ContextChunks + budget trimming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineOptions {
    pub max_chunks: usize,
    // 0 means not limit
    pub max_total_tokens: usize,
    pub merge_gap_lines: usize,
}

impl Default for ContextEngineOptions {
    fn default() -> Self {
        Self {
            max_chunks: 8,
            max_total_tokens: 2_000,
            merge_gap_lines: 3,
        }
    }
}

pub struct ReActOptions {
    // max chat times
    pub max_steps: usize,
    // Context engine options
    pub context_engine: ContextEngineOptions,
}

impl Default for ReActOptions {
    fn default() -> Self {
        Self {
            max_steps: 3,
            context_engine: ContextEngineOptions::default(),
        }
    }
}
