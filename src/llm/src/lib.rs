//! LLM Adapter Layer

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub timeout_secs: u64,
    pub max_retries: usize,
    pub retry_delay_ms: u64,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_base: "https://open.bigmodel.cn/api/paas/v4/".to_string(),
            api_key: String::new(),
            model: "glm-4-flash".to_string(),
            temperature: 0.2,
            max_tokens: None,
            timeout_secs: 120,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

impl LLMConfig {
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

        if let Ok(v) = env::var("LLM_TEMPERATURE") {
            if let Ok(t) = v.trim().parse::<f32>() {
                cfg.temperature = t;
            }
        }

        cfg.api_key =
            env::var("LLM_API_KEY").map_err(|_| anyhow!("missing environment var: LLM_API_KEY"))?;

        if cfg.api_key.trim().is_empty() {
            anyhow::bail!("LLM_API_KEY_KEY is empty!");
        }

        Ok(cfg)
    }
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

pub struct LLMClient {
    config: LLMConfig,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        Self { config }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(LLMConfig::from_env()?))
    }

    pub fn chat(&self, messages: Vec<(String, String)>) -> Result<String> {
        let mut last_error: Option<String> = None;

        for attempt in 0..self.config.max_retries {
            match self.chat_attempt(&messages) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    last_error = Some(format!("{}", e));

                    if attempt < self.config.max_retries - 1 {
                        let delay = Duration::from_millis(self.config.retry_delay_ms);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        match last_error {
            Some(err) => Err(anyhow!(err)),
            None => Err(anyhow!("All retries exhausted")),
        }
    }

    fn chat_attempt(&self, messages: &[(String, String)]) -> Result<String> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();

        let req = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: chat_messages,
            temperature: self.config.temperature,
            stream: false,
        };

        let url = format!(
            "{}/chat/completions",
            self.config.api_base.trim_end_matches('/')
        );

        let timeout = Duration::from_secs(self.config.timeout_secs);
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()?;

        let resp = client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&req)
            .send()?;

        let status = resp.status();
        let text = resp.text().unwrap_or_default();

        if !status.is_success() {
            anyhow::bail!("LLM request failed: status={} body={}", status, text);
        }

        let parsed: ChatCompletionResponse = serde_json::from_str(&text)
            .map_err(|e| anyhow!("LLM response parse error: {}; body={}", e, text))?;

        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(content)
    }

    pub fn chat_system_user(&self, system: &str, user: &str) -> Result<String> {
        self.chat(vec![
            ("system".to_string(), system.to_string()),
            ("user".to_string(), user.to_string()),
        ])
    }
}

pub fn llm_chat(cfg: &LLMConfig, system: &str, user: &str) -> Result<String> {
    let client = LLMClient::new(cfg.clone());
    client.chat_system_user(system, user)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = LLMConfig::default();
        assert_eq!(cfg.model, "glm-4-flash");
        assert_eq!(cfg.temperature, 0.2);
    }

    #[test]
    fn test_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("hello"));
    }
}
