//! LLM Adapter Layer
//!
//! This module provides a unified interface for interacting with various
//! LLM providers (OpenAI-compatible, DeepSeek, GLM, etc.).
//!
//! Design Principles:
//! - Provider-agnostic: Support multiple LLM providers through a common interface
//! - Simple blocking API for now (streaming can be added later)
//! - Environment-based configuration
//! - Clear error messages

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::env;

// ============================================================================
// Configuration
// ============================================================================

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// API base URL (e.g., "https://api.openai.com/v1")
    pub api_base: String,

    /// API key for authentication
    pub api_key: String,

    /// Model identifier (e.g., "gpt-4", "deepseek-chat")
    pub model: String,

    /// Sampling temperature (0.0 - 2.0)
    pub temperature: f32,
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
    ///
    /// Environment variables:
    /// - `LLM_API_BASE`: API base URL (optional, uses default if not set)
    /// - `LLM_API_KEY`: API key (required)
    /// - `LLM_MODEL`: Model name (optional, uses default if not set)
    /// - `LLM_TEMPERATURE`: Temperature (optional, uses default if not set)
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

        cfg.api_key = env::var("LLM_API_KEY")
            .map_err(|_| anyhow!("missing environment var: LLM_API_KEY"))?;

        if cfg.api_key.trim().is_empty() {
            anyhow::bail!("LLM_API_KEY is empty!");
        }

        Ok(cfg)
    }
}

// ============================================================================
// API Types
// ============================================================================

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

// ============================================================================
// Client
// ============================================================================

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

    /// Send a chat completion request
    pub fn chat(&self, messages: Vec<(String, String)>) -> Result<String> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage { role, content })
            .collect();

        let req = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: chat_messages,
            temperature: self.config.temperature,
            stream: false,
        };

        let url = format!("{}/chat/completions", self.config.api_base.trim_end_matches('/'));
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
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

    /// Convenience method for system + user message
    pub fn chat_system_user(&self, system: &str, user: &str) -> Result<String> {
        self.chat(vec![
            ("system".to_string(), system.to_string()),
            ("user".to_string(), user.to_string()),
        ])
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a client from config and send a chat request
pub fn llm_chat(cfg: &LLMConfig, system: &str, user: &str) -> Result<String> {
    let client = LLMClient::new(cfg.clone());
    client.chat_system_user(system, user)
}

// ============================================================================
// Tests
// ============================================================================

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
