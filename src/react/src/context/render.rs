//! Context Engine: 渲染 Retrieved Context

use crate::ContextPack;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

use super::select::select_context_chunks;

/// 上下文引擎配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineOptions {
    /// 最多包含多少个 chunk
    pub max_chunks: usize,
    /// token 上限（0 表示不限制）
    pub max_total_tokens: usize,
    /// 在多少行之内自动合并 chunk
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

/// 将 ContextPack 渲染为 LLM prompt 上下文文本
pub fn render_prompt_context(
    repo_root: &Path,
    pack: &ContextPack,
    tokenizer: &Tokenizer,
    opt: ContextEngineOptions,
) -> Result<String> {
    let selected = select_context_chunks(repo_root, &pack.hits, &pack.context, tokenizer, opt)?;

    let mut out = String::new();
    out.push_str("# Retrieved Context\n\n");
    out.push_str(&format!("Query: {}\n\n", pack.query));
    out.push_str(&format!("Chunks: {}\n\n", selected.len()));

    for (i, c) in selected.iter().enumerate() {
        let start1 = c.start_line + 1;
        let end1 = c.end_line + 1;
        out.push_str(&format!("## [{i:02}] {}:{}..={}\n", c.path, start1, end1));
        if !c.reason.is_empty() {
            out.push_str(&format!("reason: {}\n", c.reason));
        }
        out.push_str("```\n");
        for (ln0, line) in c.snippet.lines().enumerate() {
            out.push_str(&format!("{:>5} {}\n", start1 + ln0, line));
        }
        out.push_str("```\n\n");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_engine_options_default() {
        let opt = ContextEngineOptions::default();
        assert_eq!(opt.max_chunks, 8);
        assert_eq!(opt.max_total_tokens, 2000);
        assert_eq!(opt.merge_gap_lines, 3);
    }
}
