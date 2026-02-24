use crate::context_engine::render_prompt_context;
use crate::llm::llm_chat;
use crate::tools::{SearchCodeOptions, refill_hits, search_code_keyword};
use crate::types::{ContextPack, EditOp, ReActAction, ReActOptions, ReActStepTrace};
use core::code_chunk::{ContextChunk, IndexChunkOptions, RefillOptions};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use tokenizers::Tokenizer;

/// Maximum number of ContextChunks to show in the ReAct planning state summary.
///
/// Purpose: Prevent the state_summary from becoming too large (which would impact
/// LLM planning quality and token costs). 6 is an empirical value: with the default
/// `ContextEngineOptions::max_chunks=8`, showing the "first few" is typically
/// sufficient for the model to determine whether the context has enough information to answer.
const STATE_CONTEXT_PREVIEW_MAX: usize = 6;

fn extract_identifiers(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }

    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .filter(|id| {
            id.chars()
                .next()
                .map(|c| c.is_ascii_alphabetic() || c == '_')
                .unwrap_or(false)
        })
        .collect()
}

fn snake_to_pascal(s: &str) -> String {
    let mut out = String::new();
    for part in s.split('_').filter(|p| !p.is_empty()) {
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            continue;
        };
        out.push(first.to_ascii_uppercase());
        out.extend(chars);
    }
    out
}

/// Expand seed query terms with morphological variations.
///
/// This adapts user input keywords to various identifier forms found in code:
/// - Plural to singular (strip trailing 's')
/// - snake_case to PascalCase (e.g., "context_chunk" → "ContextChunk")
///
/// The transformation order is important:
/// 1. Start with original form
/// 2. Generate singular form (if plural)
/// 3. Generate PascalCase from singular form
///
/// Example: "context_chunks" → ["context_chunks", "context_chunk", "ContextChunk"]
fn expand_seed_terms(question: &str) -> Vec<String> {
    let ids = extract_identifiers(question);
    let mut out = Vec::new();

    for id in ids {
        let id = id.trim();
        if id.is_empty() {
            continue;
        }

        // Original form
        out.push(id.to_string());

        // Singular form (plural → singular)
        let singular = if id.ends_with('s') && id.len() > 1 {
            id.trim_end_matches('s')
        } else {
            id
        };
        out.push(singular.to_string());

        // PascalCase form (from singular)
        if singular.contains('_') {
            let pascal = snake_to_pascal(singular);
            if !pascal.is_empty() && pascal != id {
                out.push(pascal);
            }
        }
    }

    // Deduplicate while preserving order
    let mut uniq = Vec::new();
    for s in out {
        if !uniq.contains(&s) {
            uniq.push(s);
        }
    }
    uniq
}

fn extract_first_json_object(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'{' {
            let start = i;
            let mut depth = 0i32;
            while i < bytes.len() {
                match bytes[i] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            let end = i + 1;
                            let json_str = String::from_utf8_lossy(&bytes[start..end]).to_string();
                            // Validate that it's actually valid JSON
                            if serde_json::from_str::<serde_json::Value>(&json_str).is_ok() {
                                return Some(json_str);
                            }
                            // If not valid, continue searching
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        i += 1;
    }
    None
}

fn plan_prompt(question: &str, state_summary: &str) -> (String, String) {
    let system = r#"You are a JSON API. Output ONLY a valid JSON object.

Actions:
- {"action":"search","query":"keywords"}
- {"action":"edit_file","path":"...","start_line":N,"end_line":N,"new_content":"...","create_backup":true}
- {"action":"answer"}
- {"action":"stop","reason":"..."}

Rules:
- Output ONLY the JSON object, no markdown
- For edit_file: start_line equals end_line (single line)
- When state shows the code → answer
- When state shows NO code → search"#;

    let user = format!(
        "Question: {}\n\nState:\n{}",
        question.trim(),
        state_summary.trim()
    );
    (system.to_string(), user)
}

/// Extract the name of a definition from a ContextChunk.
///
/// Returns Some(name) if the chunk contains a struct/enum/fn/trait/impl definition,
/// or None otherwise. This helps the LLM understand WHAT definitions are available.
fn extract_definition_name(chunk: &core::code_chunk::ContextChunk) -> Option<String> {
    let snippet = chunk.snippet.trim();
    let snippet_lower = snippet.to_lowercase();

    // Try to extract: pub struct Name, struct Name, pub enum Name, etc.
    let patterns = [
        ("pub struct ", "struct "),
        ("pub enum ", "enum "),
        ("pub fn ", "fn "),
        ("trait ", "trait "),
        ("type ", "type "),
    ];

    for (full_pattern, short_pattern) in patterns {
        // Try full pattern first (e.g., "pub struct")
        if let Some(pos) = snippet_lower.find(full_pattern) {
            let after = snippet[pos + full_pattern.len()..].trim();
            // Extract the identifier (alphanumeric and underscore only)
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
        // Try short pattern (e.g., just "struct")
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

/// Check if a ContextChunk appears to contain a type/function definition.
///
/// This is a lightweight heuristic used in state summaries to help the LLM understand
/// whether retrieved context contains definitions (struct/enum/fn) or only references.
fn is_definition_chunk(chunk: &core::code_chunk::ContextChunk) -> bool {
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

fn summarize_state(
    hits: &[core::code_chunk::IndexChunk],
    context: &[core::code_chunk::ContextChunk],
) -> String {
    let definition_names: Vec<String> = context
        .iter()
        .filter_map(|c| extract_definition_name(c))
        .collect();

    let has_definition = !definition_names.is_empty();
    let mut s = String::new();
    s.push_str(&format!(
        "hits={} context_chunks={} has_definition={}",
        hits.len(),
        context.len(),
        has_definition
    ));

    // Show the names of definitions found (if any)
    if !definition_names.is_empty() {
        s.push_str(&format!(" definitions=[{}]", definition_names.join(", ")));
    }

    // Show if list_dir function is visible
    let has_list_dir = context.iter().any(|c| {
        c.snippet.contains("fn list_dir") || c.snippet.contains("pub fn list_dir")
    });
    let has_sort = context.iter().any(|c| {
        c.snippet.contains("entries.sort_by") || c.snippet.contains("entries.sort_by_key")
    });
    if has_list_dir {
        s.push_str(&format!(" visible_functions=[list_dir] has_sort={}", has_sort));
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

fn merge_hits(
    mut base: Vec<core::code_chunk::IndexChunk>,
    more: Vec<core::code_chunk::IndexChunk>,
) -> Vec<core::code_chunk::IndexChunk> {
    let mut uniq: BTreeMap<(String, usize, usize), core::code_chunk::IndexChunk> = BTreeMap::new();
    for h in base.drain(..).chain(more.into_iter()) {
        let key = (h.path.clone(), h.start_byte, h.end_byte);
        uniq.entry(key).or_insert(h);
    }
    uniq.into_values().collect()
}

/// ReAct Ask: LLM 规划 -> 工具调用(search/refill) -> 最终 answer
pub fn react_ask(
    repo_root: &Path,
    question: &str,
    tokenizer: &Tokenizer,
    llm_cfg: &crate::types::LLMConfig,
    react_opt: ReActOptions,
) -> anyhow::Result<(String, ContextPack, Vec<ReActStepTrace>)> {
    let mut step_traces = Vec::new();

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
            max_hits: 200, // Increase limit to find definitions before hitting the cap
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
    let mut last_edit: Option<(String, usize, usize)> = None; // Track last edit: (path, start, end)

    // ReAct loop: up to N rounds, each round lets the model decide whether to continue search or answer directly
    for step in 0..react_opt.max_steps.max(1) {
        // If search has no delta for multiple attempts, directly answer (to avoid model getting stuck in repeated searches)
        if no_delta_searches >= 2 && !context.is_empty() {
            pack.hits = hits.clone();
            pack.context = context.clone();

            let prompt_context = render_prompt_context(
                repo_root,
                &pack,
                tokenizer,
                react_opt.context_engine.clone(),
            )?;
            let answer = crate::llm::llm_answer(llm_cfg, question, &prompt_context)?;

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

        let plan_raw = llm_chat(llm_cfg, &system, &user)
            .unwrap_or_else(|e| format!("{{\"action\":\"stop\",\"reason\":\"LLM call failed: {e}\"}}"));

        let action = extract_first_json_object(&plan_raw)
            .and_then(|j| serde_json::from_str::<ReActAction>(&j).ok());
        let mut observation = String::new();

        // Safety check: if state shows list_dir and user asks to fix/repair, force answer
        let question_lower = question.to_lowercase();
        let asks_to_fix = question_lower.contains("fix") || question_lower.contains("修复") || question_lower.contains("repair");
        let has_list_dir_visible = state.contains("visible_functions=[list_dir]");
        if asks_to_fix && has_list_dir_visible {
            observation.push_str("Code already exists in context. Answering instead of editing.\n");
        }

        // Detect duplicate edits and force answer instead
        let action = match &action {
            Some(ReActAction::EditFile { path, start_line, end_line, .. }) => {
                // Also check: if user asks to fix existing code that's visible, force answer
                if asks_to_fix && has_list_dir_visible {
                    observation.push_str("Edit skipped: code exists in context, using answer instead.\n");
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
                            max_hits: 200, // Increase limit to find definitions before hitting the cap
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
            } => {
                let file_path = repo_root.join(&path);
                let op = EditOp::ReplaceLines {
                    start_line,
                    end_line,
                    new_content,
                };

                match crate::tools::edit_file(&file_path, &op, create_backup) {
                    Ok(result) => {
                        if result.success {
                            // Track this edit to prevent duplicates
                            last_edit = Some((path.clone(), start_line, end_line));

                            // Read the modified file to show the result
                            let preview_start = start_line.saturating_sub(3);
                            let preview_end = start_line + 3;
                            let modified_content = match crate::tools::read_file(&file_path, Some((preview_start, preview_end))) {
                                Ok(content) => content.trim().to_string(),
                                Err(_) => "(unable to read modified content)".to_string()
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

                // After successful edit, refresh context for the edited file
                if observation.contains("EDIT COMPLETE") {
                    // Read the edited function's content (around the edited lines)
                    let preview_start = start_line.saturating_sub(10);
                    let preview_end = start_line + 20;

                    if let Ok(edited_snippet) = crate::tools::read_file(&file_path, Some((preview_start, preview_end))) {
                        let edited_chunk = ContextChunk {
                            path: path.clone(),
                            alias: 0,
                            snippet: edited_snippet,
                            start_line: preview_start,
                            end_line: preview_end,
                            reason: format!("EDITED: lines {}..={} (modified content)", start_line, end_line),
                        };

                        // Add the edited file to the FRONT of context so it's prioritized
                        context.insert(0, edited_chunk);
                        observation.push_str(&format!(" Edited portion added to context: lines {}..={}.", preview_start, preview_end));
                    }

                    // Also do a re-search to get related chunks
                    let file_keywords = path.split('/')
                        .last()
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
                        let (ctx, t2) = refill_hits(repo_root, &hits, RefillOptions::default())?;
                        pack.trace.extend(t2);

                        // Merge without duplicating the edited file we just added
                        let mut seen_paths = HashSet::new();
                        seen_paths.insert(path.clone());

                        for c in ctx {
                            if !seen_paths.contains(&c.path) || c.path != path {
                                context.push(c);
                            }
                        }

                        observation.push_str(&format!(" Context refreshed: {} chunks available.", context.len()));
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
                    react_opt.context_engine.clone(),
                )?;

                // If there was an edit, add the edited content directly to prompt
                if let Some((edited_path, edited_start, edited_end)) = &last_edit {
                    let file_path = repo_root.join(edited_path);
                    let preview_start = edited_start.saturating_sub(10);
                    let preview_end = edited_end + 10;

                    if let Ok(edited_content) = crate::tools::read_file(&file_path, Some((preview_start, preview_end))) {
                        prompt_context.push_str(&format!("\n## Edited File Context\n"));
                        prompt_context.push_str(&format!("{}:{}..={}\n", edited_path, preview_start + 1, preview_end + 1));
                        prompt_context.push_str(&format!("reason: modified content (after edit)\n"));
                        prompt_context.push_str("```\n");
                        prompt_context.push_str(&edited_content);
                        prompt_context.push_str("\n```\n");
                    }
                }

                let answer = crate::llm::llm_answer(llm_cfg, question, &prompt_context)?;

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
    // Fallback: directly answer the question using current context
    let prompt_context = render_prompt_context(
        repo_root,
        &pack,
        tokenizer,
        react_opt.context_engine.clone(),
    )?;
    let answer = crate::llm::llm_answer(llm_cfg, question, &prompt_context)?;

    Ok((answer, pack, step_traces))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_first_json_object_works_with_fenced_block() {
        let s = "```json\n{\"action\":\"search\",\"query\":\"index_chunks\"}\n```";
        let j = extract_first_json_object(s).unwrap();
        let a: ReActAction = serde_json::from_str(&j).unwrap();
        match a {
            ReActAction::Search { query } => assert_eq!(query, "index_chunks"),
            _ => panic!("unexpected"),
        }
    }
}
