//! LLM Client - OpenAI-compatible API client

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use error::{LunaError, Result};

/// A request to the LLM
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    pub prompt: String,
}

/// A response from the LLM
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
}

/// LLM Client trait
pub trait LLMClient: Send + Sync {
    fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse>;
}

/// OpenAI-compatible client configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Duration,
    pub temperature: f32,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_owned(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_owned(),
            timeout: Duration::from_secs(60),
            temperature: 0.3,
        }
    }
}

impl OpenAIConfig {
    /// Create config from environment
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("LUNA_LLM_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }

        let base_url = std::env::var("LUNA_LLM_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());

        let model = std::env::var("LUNA_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());

        Some(Self {
            base_url,
            api_key,
            model,
            ..Self::default()
        })
    }
}

/// OpenAI-compatible HTTP client
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    config: OpenAIConfig,
    client: Client,
}

impl OpenAIClient {
    /// Create a new client
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LunaError::invalid_input(
                "OpenAI API key is empty. Set LUNA_LLM_API_KEY environment variable.",
            ));
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LunaError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { config, client })
    }

    /// Try to create client from environment
    pub fn try_from_env() -> Option<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config).ok()
    }
}

// Request/Response structures for OpenAI API
#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Clone, Deserialize)]
struct Choice {
    message: Message,
}

impl LLMClient for OpenAIClient {
    fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse> {
        let request_body = ChatRequest {
            model: self.config.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_owned(),
                    content: "You are a helpful coding assistant.".to_owned(),
                },
                Message {
                    role: "user".to_owned(),
                    content: req.prompt,
                },
            ],
            temperature: self.config.temperature,
        };

        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));

        // Use tokio's block_in_place for async HTTP call in sync context
        let result: Result<ChatResponse> = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let resp = self
                    .client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.config.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(|e| LunaError::network(format!("HTTP request failed: {e}")))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_owned());
                    return Err(LunaError::network(format!("API error {status}: {text}")));
                }

                let body: ChatResponse = resp
                    .json()
                    .await
                    .map_err(|e| LunaError::internal(format!("Failed to parse response: {e}")))?;

                Ok(body)
            })
        });

        let response = result?;
        let content = response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(CompletionResponse { content })
    }
}

/// Disabled client - returns error on use
#[derive(Debug, Default)]
pub struct DisabledClient;

impl LLMClient for DisabledClient {
    fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse> {
        Err(LunaError::invalid_input(
            "LLM client is not configured. Set LUNA_LLM_API_KEY environment variable.",
        ))
    }
}
