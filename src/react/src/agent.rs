//! ReAct Agent Implementation
//!
//! This module provides the main ReAct agent that orchestrates:
//! - LLM planning
//! - Tool execution
//! - State tracking
//! - Loop termination

use crate::context::{render_prompt_context, ContextEngineOptions};
use crate::planner::{
    expand_seed_terms, extract_first_json_object, plan_prompt, ReActAction, ReActStepTrace,
};
use crate::{merge_hits, summarize_state};
use anyhow::Result;
use core::code_chunk::{ContextChunk, IndexChunkOptions, RefillOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer;

use llm::{LLMClient, LLMConfig};
use toolkit::ExecutionPolicy;
use tools::{edit_file, read_file, refill_hits, search_code_keyword};
use tools::{ContextPack, EditOp, SearchCodeOptions};

// ============================================================================
// Agent Options
// ============================================================================

/// Options for configuring the ReAct agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactOptions {
    /// Maximum number of steps to execute
    pub max_steps: usize,

    /// Context engine options
    pub context_engine: ContextEngineOptions,

    /// Execution policy for potentially destructive actions
    pub policy: ExecutionPolicy,
}

impl Default for ReactOptions {
    fn default() -> Self {
        Self {
            max_steps: 3,
            context_engine: ContextEngineOptions::default(),
            policy: ExecutionPolicy::default(),
        }
    }
}

// ============================================================================
// ReAct Agent
// ============================================================================

/// A ReAct agent that can reason and act using tools
pub struct ReactAgent {
    config: LLMConfig,
    options: ReactOptions,
}

impl ReactAgent {
    /// Create a new ReAct agent
    pub fn new(config: LLMConfig, options: ReactOptions) -> Self {
        Self { config, options }
    }

    /// Run the agent on a question
    pub fn ask(
        &self,
        repo_root: &Path,
        question: &str,
        tokenizer: &Tokenizer,
    ) -> Result<(String, ContextPack, Vec<ReActStepTrace>)> {
        let mut step_traces = Vec::new();
        let client = LLMClient::new(self.config.clone());

        // Step0: Perform a "fallback search" using identifiers extracted from the question
        let seed_terms = expand_seed_terms(question);
        let seed_query = if seed_terms.is_empty() {
            question.trim().to_string()
        } else {
            seed_terms.join(" ")
        };

        let (mut hits, mut trace) = search_code_keyword(
            repo_root,
            &seed_query,
            tokenizer,
            IndexChunkOptions::default(),
            SearchCodeOptions {
                max_hits: 200,
                ..Default::default()
            },
        )?;
        let (mut context, mut trace2) = refill_hits(repo_root, &hits, RefillOptions::default())?;
        trace.append(&mut trace2);

        let mut pack = ContextPack {
            query: question.to_string(),
            hits: hits.clone(),
            context: context.clone(),
            trace,
        };

        let mut no_delta_searches = 0usize;
        let mut last_search_query: Option<String> = None;
        let mut last_edit: Option<(String, usize, usize)> = None;

        // ReAct loop
        for step in 0..self.options.max_steps.max(1) {
            // Auto-exit if no delta after multiple searches
            if no_delta_searches >= 2 && !context.is_empty() {
                pack.hits = hits.clone();
                pack.context = context.clone();

                let prompt_context = render_prompt_context(
                    repo_root,
                    &pack,
                    tokenizer,
                    self.options.context_engine.clone(),
                )?;
                let answer = self.answer(&client, question, &prompt_context)?;

                step_traces.push(ReActStepTrace {
                    step,
                    plan_raw: "{\"action\":\"answer\"}".to_string(),
                    action: Some(ReActAction::Answer),
                    observation: "auto-answer: previous search had no delta".to_string(),
                });
                return Ok((answer, pack, step_traces));
            }

            let state = summarize_state(&hits, &context);
            let (system, user) = plan_prompt(question, &state);

            let plan_raw = client.chat_system_user(&system, &user).unwrap_or_else(|e| {
                format!("{{\"action\":\"stop\",\"reason\":\"LLM call failed: {e}\"}}")
            });

            let action = extract_first_json_object(&plan_raw)
                .and_then(|j| serde_json::from_str::<ReActAction>(&j).ok());
            let mut observation = String::new();

            // Safety check: if state shows code exists and user asks to fix, force answer
            let question_lower = question.to_lowercase();
            let asks_to_fix = question_lower.contains("fix")
                || question_lower.contains("修复")
                || question_lower.contains("repair");
            let has_list_dir_visible = state.contains("visible_functions=[list_dir]");
            if asks_to_fix && has_list_dir_visible {
                observation
                    .push_str("Code already exists in context. Answering instead of editing.\n");
            }

            // Detect duplicate edits and force answer instead
            let action = match &action {
                Some(ReActAction::EditFile {
                    path,
                    start_line,
                    end_line,
                    ..
                }) => {
                    if asks_to_fix && has_list_dir_visible {
                        observation.push_str(
                            "Edit skipped: code exists in context, using answer instead.\n",
                        );
                        Some(ReActAction::Answer)
                    } else if let Some((last_path, last_start, last_end)) = &last_edit {
                        if path == last_path && start_line == last_start && end_line == last_end {
                            observation.push_str(&format!(
                                "duplicate edit skipped: already edited {}:{}..={} in previous step. Moving to answer.\n",
                                path, start_line, end_line
                            ));
                            Some(ReActAction::Answer)
                        } else {
                            action.clone()
                        }
                    } else {
                        action.clone()
                    }
                }
                _ => action.clone(),
            };

            match action.clone().unwrap_or(ReActAction::Stop {
                reason: Some("invalid plan".into()),
            }) {
                ReActAction::Search { query } => {
                    let q = query.trim();
                    if q.is_empty() {
                        observation.push_str("search skipped: empty query");
                    } else {
                        let before_hits = hits.len();
                        let before_ctx = context.len();
                        let repeated = last_search_query
                            .as_deref()
                            .is_some_and(|last| last.eq_ignore_ascii_case(q));

                        let (more, t) = search_code_keyword(
                            repo_root,
                            q,
                            tokenizer,
                            IndexChunkOptions::default(),
                            SearchCodeOptions {
                                max_hits: 200,
                                ..Default::default()
                            },
                        )?;
                        pack.trace.extend(t);
                        hits = merge_hits(hits, more);
                        let (ctx, t2) = refill_hits(repo_root, &hits, RefillOptions::default())?;
                        pack.trace.extend(t2);
                        context = ctx;
                        let after_hits = hits.len();
                        let after_ctx = context.len();
                        let no_delta = before_hits == after_hits && before_ctx == after_ctx;
                        if no_delta {
                            no_delta_searches += 1;
                        } else {
                            no_delta_searches = 0;
                        }
                        last_search_query = Some(q.to_string());

                        observation.push_str(&format!(
                            "search ok: hits={} context={}{}",
                            after_hits,
                            after_ctx,
                            if repeated && no_delta {
                                " (repeated, no delta)"
                            } else {
                                ""
                            }
                        ));
                    }
                }
                ReActAction::EditFile {
                    path,
                    start_line,
                    end_line,
                    new_content,
                    create_backup,
                    confirm,
                } => {
                    // Policy gate: allow/confirm edit_file
                    if !self.options.policy.allow_edit_file {
                        observation.push_str("edit blocked by policy: allow_edit_file=false");
                        continue;
                    }
                    if self.options.policy.require_confirm_edit_file
                        && confirm.unwrap_or(false) != true
                    {
                        observation.push_str(
                            "edit requires confirmation: set confirm=true (Human-in-the-loop)",
                        );
                        continue;
                    }
                    let file_path = repo_root.join(&path);
                    let op = EditOp::ReplaceLines {
                        start_line,
                        end_line,
                        new_content,
                    };

                    match edit_file(&file_path, &op, create_backup) {
                        Ok(result) => {
                            if result.success {
                                last_edit = Some((path.clone(), start_line, end_line));

                                let preview_start = start_line.saturating_sub(3);
                                let preview_end = start_line + 3;
                                let modified_content =
                                    match read_file(&file_path, Some((preview_start, preview_end)))
                                    {
                                        Ok(content) => content.trim().to_string(),
                                        Err(_) => "(unable to read modified content)".to_string(),
                                    };

                                observation.push_str(&format!(
                                    "edit ok: path={} lines_changed={} backup={}\nModified content preview:\n{}\n\nEDIT COMPLETE. File modified successfully. PROCEED TO ANSWER.",
                                    result.path,
                                    result.lines_changed.unwrap_or(0),
                                    result.backup_path.is_some(),
                                    modified_content
                                ));
                            } else {
                                observation.push_str(&format!(
                                    "edit failed: path={} error={}",
                                    result.path,
                                    result.error.unwrap_or_else(|| "unknown".to_string())
                                ));
                            }
                        }
                        Err(e) => {
                            observation.push_str(&format!("edit error: {}", e));
                        }
                    }

                    // After successful edit, refresh context
                    if observation.contains("EDIT COMPLETE") {
                        let preview_start = start_line.saturating_sub(10);
                        let preview_end = start_line + 20;

                        if let Ok(edited_snippet) =
                            read_file(&file_path, Some((preview_start, preview_end)))
                        {
                            let edited_chunk = ContextChunk {
                                path: path.clone(),
                                alias: 0,
                                snippet: edited_snippet,
                                start_line: preview_start,
                                end_line: preview_end,
                                reason: format!(
                                    "EDITED: lines {}..={} (modified content)",
                                    start_line + 1,
                                    end_line + 1
                                ),
                            };
                            context.insert(0, edited_chunk);
                        }

                        // Re-search to get related chunks
                        let file_keywords = path
                            .split('/')
                            .next_back()
                            .unwrap_or(&path)
                            .trim_end_matches(".rs")
                            .to_string();

                        if let Ok((new_hits, t)) = search_code_keyword(
                            repo_root,
                            &file_keywords,
                            tokenizer,
                            IndexChunkOptions::default(),
                            SearchCodeOptions {
                                max_hits: 50,
                                ..Default::default()
                            },
                        ) {
                            pack.trace.extend(t);
                            hits = merge_hits(hits, new_hits);
                            let (ctx, t2) =
                                refill_hits(repo_root, &hits, RefillOptions::default())?;
                            pack.trace.extend(t2);

                            let mut seen_paths = HashSet::new();
                            seen_paths.insert(path.clone());

                            for c in ctx {
                                if !seen_paths.contains(&c.path) || c.path != path {
                                    context.push(c);
                                }
                            }
                        }
                    }
                }
                ReActAction::Answer => {
                    observation.push_str("answer");
                    pack.hits = hits.clone();
                    pack.context = context.clone();

                    let mut prompt_context = render_prompt_context(
                        repo_root,
                        &pack,
                        tokenizer,
                        self.options.context_engine.clone(),
                    )?;

                    if let Some((edited_path, edited_start, edited_end)) = &last_edit {
                        let file_path = repo_root.join(edited_path);
                        let preview_start = edited_start.saturating_sub(10);
                        let preview_end = edited_end + 10;

                        if let Ok(edited_content) =
                            read_file(&file_path, Some((preview_start, preview_end)))
                        {
                            prompt_context.push_str("\n## Edited File Context\n");
                            prompt_context.push_str(&format!(
                                "{}:{}..={}\n",
                                edited_path,
                                preview_start + 1,
                                preview_end + 1
                            ));
                            prompt_context.push_str("reason: modified content (after edit)\n");
                            prompt_context.push_str("```\n");
                            prompt_context.push_str(&edited_content);
                            prompt_context.push_str("\n```\n");
                        }
                    }

                    let answer = self.answer(&client, question, &prompt_context)?;

                    step_traces.push(ReActStepTrace {
                        step,
                        plan_raw,
                        action: Some(ReActAction::Answer),
                        observation,
                    });
                    return Ok((answer, pack, step_traces));
                }
                ReActAction::Stop { reason } => {
                    observation.push_str(&format!("stop: {}", reason.unwrap_or_default()));
                    step_traces.push(ReActStepTrace {
                        step,
                        plan_raw,
                        action,
                        observation,
                    });
                    break;
                }
            }
            pack.hits = hits.clone();
            pack.context = context.clone();

            step_traces.push(ReActStepTrace {
                step,
                plan_raw,
                action,
                observation,
            });
        }

        // Fallback: directly answer using current context
        let prompt_context = render_prompt_context(
            repo_root,
            &pack,
            tokenizer,
            self.options.context_engine.clone(),
        )?;
        let answer = self.answer(&client, question, &prompt_context)?;

        Ok((answer, pack, step_traces))
    }

    /// Generate an answer using the LLM
    fn answer(&self, client: &LLMClient, question: &str, prompt_context: &str) -> Result<String> {
        const ANSWER_SYSTEM_PROMPT: &str = r###"You are a senior software engineer assistant. You can only answer based on the provided Retrieved Context.
- Do not fabricate non-existent files/functions/line numbers.
- Each conclusion must be cited in the format `path:start..end` (where start/end are line numbers, enclosed in backticks), and the citation must be enclosed in backticks.
- References can only come from the header line of the Retrieved Context (such as "## [00] path:start..=end"). Do not make up non-existent line numbers or reference files that have not appeared.
- If the context is insufficient to answer, please clearly state what information is missing and suggest using search/refill to retrieve it."###;

        let user = format!(
            "{}\n\n# User Question\n\n{}\n",
            prompt_context,
            question.trim()
        );

        let mut ans = client.chat_system_user(ANSWER_SYSTEM_PROMPT, &user)?;

        // Retry for citation compliance (simplified version)
        for _ in 0..1 {
            // Check if answer has citations (simple heuristic)
            let has_citations = ans.contains(':') && (ans.contains("..=") || ans.contains(".."));
            if has_citations {
                break;
            }

            let system = format!(
                "{}\n\nAdditional rule: Your answer must include citations in the format `path:start..end`.",
                ANSWER_SYSTEM_PROMPT
            );
            let user2 = format!(
                "{}\n\n# Previous Answer (missing citations)\n\n{}\n",
                user,
                ans.trim()
            );
            ans = client.chat_system_user(&system, &user2)?;
        }

        Ok(ans)
    }
}

// ============================================================================
// Convenience Function
// ============================================================================

/// Convenience function to run a ReAct query with default options
pub fn react_ask(
    repo_root: &Path,
    question: &str,
    tokenizer: &Tokenizer,
    llm_cfg: &LLMConfig,
    react_opt: ReactOptions,
) -> Result<(String, ContextPack, Vec<ReActStepTrace>)> {
    let agent = ReactAgent::new(llm_cfg.clone(), react_opt);
    agent.ask(repo_root, question, tokenizer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_options_default() {
        let opt = ReactOptions::default();
        assert_eq!(opt.max_steps, 3);
        assert_eq!(opt.context_engine.max_chunks, 8);
    }
}
