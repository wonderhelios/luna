use core::str;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeChunk {
    pub path: String,
    #[serde(rename = "alias")]
    pub alias: usize,
    #[serde(rename = "snippet")]
    pub snippet: String,
    #[serde(rename = "start")]
    pub start_line: usize,
    #[serde(rename = "end")]
    pub end_line: usize,
}

impl CodeChunk {
    pub fn is_empty(&self) -> bool {
        self.snippet.trim().is_empty()
    }
}

impl fmt::Display for CodeChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}\n{}", self.alias, self.path, self.snippet)
    }
}

#[derive(Debug, Clone)]
pub struct ChunkOptions {
    // max bytes of one chunk
    pub max_chunk_bytes: usize,
    // split-windows' max lines of one chunk
    pub max_chunk_lines: usize,
    // split-windows' overlap lines
    pub overlap_lines: usize,
    // max lines for skipping global ast(for example not found top-level scope)
    pub fallback_max_lines: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        ChunkOptions {
            max_chunk_bytes: 8 * 1024,
            max_chunk_lines: 150,
            overlap_lines: 20,
            fallback_max_lines: 200,
        }
    }
}

// IndexChunk 用于 检索 阶段：尺寸可控，便于向量检索、便于去重
// Attention：不保证语义完整，因此在喂到 LLM 之前，通常需要经历 refill 变为 ContextChunk
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexChunk {
    pub path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

impl IndexChunk {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

// token 重叠策略
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlapStrategy {
    // 从窗口末尾回退 N 行
    ByLines(usize),
    // 0~1 之间的比例
    Partial(f64),
}

impl Default for OverlapStrategy {
    fn default() -> Self {
        Self::Partial(0.5)
    }
}

impl OverlapStrategy {
    pub fn next_subdivision(&self, max_tokens: usize) -> usize {
        (match self {
            OverlapStrategy::ByLines(_) => max_tokens.saturating_sub(1),
            OverlapStrategy::Partial(part) => ((max_tokens as f64) * (*part)).round() as usize,
        })
        .max(1)
    }
}
#[derive(Clone, Debug)]
pub struct IndexChunkOptions {
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub overlap: OverlapStrategy,
    pub fallback_lines: usize,
}

impl Default for IndexChunkOptions {
    fn default() -> Self {
        Self {
            min_chunk_tokens: 50,
            max_chunk_tokens: 256,
            overlap: OverlapStrategy::default(),
            fallback_lines: 120,
        }
    }
}

#[derive(Debug)]
pub enum IndexChunkBuildError {
    InvalidInput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextChunk {
    pub path: String,
    #[serde(rename = "alias")]
    pub alias: usize,
    #[serde(rename = "snippet")]
    pub snippet: String,
    #[serde(rename = "start")]
    pub start_line: usize,
    #[serde(rename = "end")]
    pub end_line: usize,
    #[serde(default)]
    pub reason: String,
}

impl ContextChunk {
    pub fn is_empty(&self) -> bool {
        self.snippet.trim().is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct RefillOptions {
    // 找不到 enclosing top-level scope时，围绕命中行做兜底窗口
    pub fallback_window_lines: usize,
}

impl Default for RefillOptions {
    fn default() -> Self {
        Self {
            fallback_window_lines: 120,
        }
    }
}
