//! Planning and Action Selection for ReAct Loop
//!
//! This module handles:
//! - Prompt construction for LLM planning
//! - Action parsing from LLM responses
//! - Step trace tracking

use serde::{Deserialize, Serialize};

// ============================================================================
// Action Types
// ============================================================================

/// LLM plan output (ReAct's Act)
///
/// The LLM must output a JSON object with an "action" field:
/// - Search: {"action":"search","query":"keywords"}
/// - Edit: {"action":"edit_file","path":"...","start_line":N,"end_line":N,"new_content":"..."}
/// - Answer: {"action":"answer"}
/// - Stop: {"action":"stop","reason":"..."}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum ReActAction {
    Search {
        query: String,
    },
    #[serde(rename = "edit_file")]
    EditFile {
        path: String,
        start_line: usize,
        end_line: usize,
        new_content: String,
        create_backup: bool,
        #[serde(default)]
        confirm: Option<bool>,
    },
    Answer,
    Stop {
        reason: Option<String>,
    },
}

/// Trace of a single ReAct step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStepTrace {
    pub step: usize,
    pub plan_raw: String,
    pub action: Option<ReActAction>,
    pub observation: String,
}

// ============================================================================
// Prompt Construction
// ============================================================================

/// Build the planning prompt for the LLM
pub fn plan_prompt(question: &str, state_summary: &str) -> (String, String) {
    let system = r#"You are a JSON API. Output ONLY a valid JSON object.

Actions:
- {"action":"search","query":"keywords"}
- {"action":"edit_file","path":"...","start_line":N,"end_line":N,"new_content":"...","create_backup":true}
- {"action":"answer"}
- {"action":"stop","reason":"..."}

Rules:
- Output ONLY the JSON object, no markdown
- For edit_file: lines are 0-based, start_line equals end_line (single line)
- When state shows the code → answer
- When state shows NO code → search"#;

    let user = format!(
        "Question: {}\n\nState:\n{}",
        question.trim(),
        state_summary.trim()
    );
    (system.to_string(), user)
}

// ============================================================================
// JSON Extraction
// ============================================================================

/// Extract the first valid JSON object from a string
///
/// Handles cases where LLM wraps JSON in markdown code blocks
pub fn extract_first_json_object(s: &str) -> Option<String> {
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

// ============================================================================
// Seed Term Expansion
// ============================================================================

/// Extract identifiers from a string
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

/// Convert snake_case to PascalCase
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

/// Expand seed query terms with morphological variations
///
/// This adapts user input keywords to various identifier forms found in code:
/// - Plural to singular (strip trailing 's')
/// - snake_case to PascalCase (e.g., "context_chunk" → "ContextChunk")
///
/// Example: "context_chunks" → ["context_chunks", "context_chunk", "ContextChunk"]
pub fn expand_seed_terms(question: &str) -> Vec<String> {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_first_json_object_with_fence() {
        let s = "```json\n{\"action\":\"search\",\"query\":\"index_chunks\"}\n```";
        let j = extract_first_json_object(s).unwrap();
        let a: ReActAction = serde_json::from_str(&j).unwrap();
        match a {
            ReActAction::Search { query } => assert_eq!(query, "index_chunks"),
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn test_extract_first_json_object_without_fence() {
        let s = "{\"action\":\"answer\"}";
        let j = extract_first_json_object(s).unwrap();
        let a: ReActAction = serde_json::from_str(&j).unwrap();
        assert!(matches!(a, ReActAction::Answer));
    }

    #[test]
    fn test_expand_seed_terms() {
        let terms = expand_seed_terms("context_chunks");
        assert!(terms.contains(&"context_chunks".to_string()));
        assert!(terms.contains(&"context_chunk".to_string()));
        assert!(terms.contains(&"ContextChunk".to_string()));
    }

    #[test]
    fn test_snake_to_pascal() {
        assert_eq!(snake_to_pascal("context_chunk"), "ContextChunk");
        assert_eq!(snake_to_pascal("my_struct"), "MyStruct");
        assert_eq!(snake_to_pascal("already_pascal"), "AlreadyPascal");
    }
}
