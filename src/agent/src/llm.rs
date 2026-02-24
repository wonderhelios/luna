use crate::types::LLMConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json;
use std::env;

const ANSWER_SYSTEM_PROMPT: &str = r###"You are a senior software engineer assistant. You can only answer based on the provided Retrieved Context.
- Do not fabricate non-existent files/functions/line numbers.
- Each conclusion must be cited in the format `path:start..end` (where start/end are line numbers, enclosed in backticks), and the citation must be enclosed in backticks.
- References can only come from the header line of the Retrieved Context (such as "## [00] path:start..=end"). Do not make up non-existent line numbers or reference files that have not appeared.
- If the context is insufficient to answer, please clearly state what information is missing and suggest using search/refill to retrieve it."###;

fn looks_like_citation_token(tok: &str) -> bool {
    // Allow tok to carry punctuation/backticks (e.g., "`a/b.rs:12..=34`" or "a/b.rs:12..34,")
    let t = tok.trim_matches(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '`' | ',' | '.' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\''
            )
    });
    let has_colon = t.contains(':');
    let has_range = t.contains("..=") || t.contains("..");
    let has_digit = t.chars().any(|c| c.is_ascii_digit());
    // Rough requirement to look like a "path": contains '/' or '.' (file extension)
    let has_path_hint = t.contains('/') || t.contains('.');
    has_colon && has_range && has_digit && has_path_hint
}

fn parse_line_range_1based(s: &str) -> Option<(usize, usize)> {
    // Support "12..34" or "12..=34"
    let (a, b) = if let Some((a, b)) = s.split_once("..=") {
        (a, b)
    } else if let Some((a, b)) = s.split_once("..") {
        (a, b)
    } else {
        return None;
    };
    let start = a.trim().parse::<usize>().ok()?;
    let end = b.trim().parse::<usize>().ok()?;
    Some((start, end))
}

fn extract_allowed_citation_ranges(prompt_context: &str) -> Vec<(String, usize, usize)> {
    // Parse from render_prompt_context output in the form:
    // "## [00] path/to/file.rs:12..=34"
    let mut out = Vec::new();
    for line in prompt_context.lines() {
        let line = line.trim();
        if !line.starts_with("## [") {
            continue;
        }
        // Remove "## [xx] " prefix
        let Some((_, rest)) = line.split_once("] ") else {
            continue;
        };
        // path typically doesn't contain ':' (except Windows drive letters), use rfind(':') to get the last ':'
        let Some(pos) = rest.rfind(':') else {
            continue;
        };
        let path = rest[..pos].trim().to_string();
        let range = rest[pos + 1..].trim();
        let Some((start, end)) = parse_line_range_1based(range) else {
            continue;
        };
        out.push((path, start, end));
    }
    out
}

fn extract_backticked_citations(s: &str) -> Vec<String> {
    // 抽取所有反引号包裹的片段作为候选引用。
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i + 1;
            if let Some(end) = bytes[start..].iter().position(|&b| b == b'`') {
                let seg = String::from_utf8_lossy(&bytes[start..start + end]).to_string();
                if looks_like_citation_token(&seg) {
                    out.push(seg);
                }
                i = start + end + 1;
                continue;
            }
            break;
        }
        i += 1;
    }
    out
}

fn citation_within_allowed(citation: &str, allowed: &[(String, usize, usize)]) -> bool {
    // citation: "path:12..=34" or "path:12..34"
    let Some(pos) = citation.rfind(':') else {
        return false;
    };
    let path = citation[..pos].trim();
    let range = citation[pos + 1..].trim();
    let Some((start, end)) = parse_line_range_1based(range) else {
        return false;
    };
    allowed
        .iter()
        .any(|(p, s, e)| p == path && start >= *s && end <= *e)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

fn make_system_user_message(system: &str, user: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: system.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user.to_string(),
        },
    ]
}

fn chat_completion(cfg: &LLMConfig, messages: Vec<ChatMessage>) -> anyhow::Result<String> {
    let req = ChatCompletionRequest {
        model: cfg.model.clone(),
        messages: messages,
        temperature: cfg.temperature,
        stream: false,
    };

    let url = format!("{}/chat/completions", cfg.api_base.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let resp = client
        .post(url)
        .bearer_auth(cfg.api_key.clone())
        .json(&req)
        .send()?;

    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("LLM request failed: status={} body={}", status, text);
    }
    let parsed: ChatCompletionResponse = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("LLM response parses error: {e}; body={}", text))?;
    let content = parsed
        .choices
        .get(0)
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(content)
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_base: "https://open.bigmodel.cn/api/paas/v4/".to_string(),
            api_key: String::new(),
            model: "glm-4-flash".to_string(),
            temperature: 0.2,
        }
    }
}

impl LLMConfig {
    /// Read config from environment variables
    pub fn from_env() -> Result<Self> {
        let mut cfg = Self::default();
        if let Ok(v) = env::var("LLM_API_BASE") {
            if !v.trim().is_empty() {
                cfg.api_base = v;
            }
        }
        if let Ok(v) = env::var("LLM_MODEL") {
            if !v.trim().is_empty() {
                cfg.model = v;
            }
        }
        cfg.api_key = env::var("LLM_API_KEY")
            .map_err(|_| anyhow::anyhow!("missing environment var: LLM_API_KEY"))?;
        if cfg.api_key.trim().is_empty() {
            anyhow::bail!("LLM_API_KEY is empty!");
        }
        Ok(cfg)
    }
}

pub fn llm_chat(cfg: &LLMConfig, system: &str, user: &str) -> anyhow::Result<String> {
    chat_completion(cfg, make_system_user_message(system, user))
}

/// Minimal LLM call: send context and question to model for answer
/// TODO(wonder): streaming output
pub fn llm_answer(cfg: &LLMConfig, question: &str, prompt_context: &str) -> anyhow::Result<String> {
    let user = format!(
        "{}\n\n# User Question\n\n{}\n",
        prompt_context,
        question.trim()
    );
    let mut ans = llm_chat(cfg, ANSWER_SYSTEM_PROMPT, &user)?;

    // Some models may not strictly follow the "must cite" system rule; here we retry up to 2 times.
    let allowed = extract_allowed_citation_ranges(prompt_context);
    for _ in 0..2 {
        let citations = extract_backticked_citations(&ans);
        let ok = !citations.is_empty()
            && citations
                .iter()
                .all(|c| citation_within_allowed(c, &allowed));
        if ok {
            break;
        }

        let allowed_hint = if allowed.is_empty() {
            String::new()
        } else {
            let mut s = String::new();
            s.push_str("\n\nAllowed citation ranges are as follows (must select from these and wrap with backticks):\n");
            for (p, a, b) in allowed.iter().take(12) {
                s.push_str(&format!("- `{p}:{a}..={b}`\n"));
            }
            s
        };

        let system = format!(
            "{}\n\nAdditional rule (must comply): Your previous answer's citations were non-compliant (missing/not wrapped with backticks/cited outside Retrieved Context range). Please rewrite the answer, ensuring each point includes at least one citation in the format `path:start..end`, wrapped with backticks.{}\nOnly output the rewritten answer.",
            ANSWER_SYSTEM_PROMPT, allowed_hint
        );
        let user2 = format!(
            "{}\n\n# User Question\n\n{}\n\n# Previous Answer (invalid, missing citations)\n\n{}\n",
            prompt_context,
            question.trim(),
            ans.trim()
        );
        ans = llm_chat(cfg, &system, &user2)?;
    }

    Ok(ans)
}
