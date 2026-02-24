use crate::context_engine::render_prompt_context;
use crate::llm::llm_chat;
use crate::tools::{SearchCodeOptions, refill_hits, search_code_keyword};
use crate::types::{ContextPack, ReActAction, ReActOptions, ReActStepTrace};
use core::code_chunk::{IndexChunkOptions, RefillOptions};
use std::collections::BTreeMap;
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
                            return Some(String::from_utf8_lossy(&bytes[start..end]).to_string());
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            return None;
        }
        i += 1;
    }
    None
}

fn plan_prompt(question: &str, state_summary: &str) -> (String, String) {
    // Gather first, then answer
    let system = r#"You are the scheduler (ReAct) of a codebase agent. You can only choose one action from the following, and you must output a JSON object (do not output any extra text):
Optional actions:
1) {"action":"search","query":"..."} // The query must be ASCII code keywords (letters/numbers/underscores/spaces), no Chinese characters.
2) {"action":"answer"} // Indicates sufficient context to proceed to the final answer.
3) {"action":"stop","reason":"..."} // Indicates that you cannot continue (e.g., no keywords, empty context, and no search).
Rules:
- Prioritize search: when the context is empty or clearly insufficient to answer.
- Query selection for search: extract identifiers from the question, try singular/plural/snake_case/PascalCase variants.
- CRITICAL: If the question asks about a specific type/struct/enum/function (e.g., "what is ContextChunk?"), and the definitions list contains that name BUT the preview chunks don't show its actual definition body, you MUST search again with more specific query (e.g., "struct ContextChunk").
- Avoid duplicate searches: If the previous search did not result in an increase in hits/context_chunks, try a different query variant before giving up.
- If there is already a relevant defined block (e.g., fn/struct/enum definition) with its body content visible, then action=answer.
- Never output a natural language interpretation directly; only output JSON. "#;

    let user = format!(
        "# Question\n{}\n\n# State\n{}\n",
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
            .unwrap_or_else(|e| format!("{{\"action\":\"stop\",\"reason\":\"{e}\"}}"));

        let action = extract_first_json_object(&plan_raw)
            .and_then(|j| serde_json::from_str::<ReActAction>(&j).ok());
        let mut observation = String::new();

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
            ReActAction::Answer => {
                observation.push_str("answer");
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
