//! OpenAI-compatible HTTP client for LLM completion.
//!
//! Supports OpenAI, OpenRouter, SiliconFlow, and other OpenAI-compatible APIs.

use std::time::Duration;

use error::{LunaError, Result};
use serde::{Deserialize, Serialize};

use crate::{CompletionRequest, CompletionResponse, LLMClient};

/// OpenAI API request body
#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// OpenAI API response
#[derive(Debug, Clone, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    error: Option<ApiError>,
}

#[derive(Debug, Clone, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Clone, Deserialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    ty: Option<String>,
}

/// Configuration for OpenAI-compatible client
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API base URL (e.g., "https://api.openai.com/v1")
    pub base_url: String,
    /// API key
    pub api_key: String,
    /// Model name (e.g., "gpt-4o-mini", "claude-3.5-sonnet")
    pub model: String,
    /// Request timeout
    pub timeout: Duration,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Max tokens per request
    pub max_tokens: Option<u32>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_owned(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_owned(),
            timeout: Duration::from_secs(60),
            temperature: 0.3,
            max_tokens: Some(4096),
        }
    }
}

impl OpenAIConfig {
    /// Create config from environment variables
    ///
    /// Variables:
    /// - `LUNA_LLM_API_KEY` (required)
    /// - `LUNA_LLM_BASE_URL` (optional, default: OpenAI)
    /// - `LUNA_LLM_MODEL` (optional, default: gpt-4o-mini)
    /// - `LUNA_LLM_TIMEOUT_SECS` (optional, default: 60)
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("LUNA_LLM_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }

        let base_url = std::env::var("LUNA_LLM_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());

        let model = std::env::var("LUNA_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());

        let timeout_secs = std::env::var("LUNA_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);

        Some(Self {
            base_url,
            api_key,
            model,
            timeout: Duration::from_secs(timeout_secs),
            temperature: 0.3,
            max_tokens: Some(4096),
        })
    }

    /// Create config for OpenRouter
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: "https://openrouter.ai/api/v1".to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            ..Self::default()
        }
    }

    /// Create config for SiliconFlow
    pub fn siliconflow(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.siliconflow.cn/v1".to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            ..Self::default()
        }
    }
}

/// OpenAI-compatible HTTP client
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIClient {
    /// Create a new client with the given config
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LunaError::invalid_input(
                "OpenAI API key is empty. Set LUNA_LLM_API_KEY environment variable.",
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LunaError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { config, client })
    }

    /// Try to create client from environment variables
    pub fn try_from_env() -> Option<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config).ok()
    }

    fn build_url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }
}

impl LLMClient for OpenAIClient {
    fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse> {
        let request_body = ChatRequest {
            model: self.config.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: "You are a helpful coding assistant. Respond with concise, accurate answers.".to_owned(),
                },
                Message {
                    role: "user".to_owned(),
                    content: req.prompt,
                },
            ],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let url = self.build_url();
        let api_key = self.config.api_key.clone();
        let client = self.client.clone();

        let result = tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                let resp = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {api_key}"))
                    .header("Content-Type", "application/json")
                    .json(&request_body)
                    .send()
                    .await;

                match resp {
                    Ok(r) => {
                        let status = r.status();
                        match r.json::<ChatResponse>().await {
                            Ok(body) => Ok((status, body)),
                            Err(e) => Err(LunaError::internal(format!(
                                "Failed to parse LLM response: {e}"
                            ))),
                        }
                    }
                    Err(e) => {
                        if e.is_timeout() {
                            Err(LunaError::internal(format!(
                                "LLM request timeout after {:?}",
                                std::time::Duration::from_secs(60)
                            )))
                        } else {
                            Err(LunaError::internal(format!("LLM request failed: {e}")))
                        }
                    }
                }
            })
        });

        let (status, body) = result?;

        // Handle API error
        if let Some(err) = body.error {
            return Err(LunaError::internal(format!(
                "LLM API error ({}): {}",
                err.ty.as_deref().unwrap_or("unknown"),
                err.message
            )));
        }

        if !status.is_success() {
            return Err(LunaError::internal(format!(
                "LLM API returned HTTP {status}"
            )));
        }

        let content = body
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(CompletionResponse { content })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_config_from_env() {
        // This test would need env vars set, so we just verify the structure
        let config = OpenAIConfig {
            api_key: "test-key".to_owned(),
            ..OpenAIConfig::default()
        };
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_openrouter_config() {
        let config = OpenAIConfig::openrouter("my-key", "anthropic/claude-3.5-sonnet");
        assert_eq!(config.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(config.model, "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_siliconflow_config() {
        let config = OpenAIConfig::siliconflow("my-key", "Qwen/Qwen2.5-Coder-32B-Instruct");
        assert_eq!(config.base_url, "https://api.siliconflow.cn/v1");
        assert_eq!(config.model, "Qwen/Qwen2.5-Coder-32B-Instruct");
    }

    #[test]
    fn test_deepseek_config() {
        let config = OpenAIConfig {
            base_url: "https://api.deepseek.com/v1".to_owned(),
            api_key: "test-key".to_owned(),
            model: "deepseek-chat".to_owned(),
            ..OpenAIConfig::default()
        };
        assert_eq!(config.base_url, "https://api.deepseek.com/v1");
        assert_eq!(config.model, "deepseek-chat");
    }
}
