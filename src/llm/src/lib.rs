//! LLM client interfaces

mod openai;

pub use openai::{OpenAIClient, OpenAIConfig};

use error::{LunaError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    // A simple prompt string (system + user already composed)
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    // Raw model output
    pub content: String,
}

/// Minimial LLM client
pub trait LLMClient: Send + Sync {
    fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse>;
}

#[derive(Debug, Default)]
pub struct DisabledClient;

impl LLMClient for DisabledClient {
    fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse> {
        Err(LunaError::invalid_input(
            "LLM client is not configured. Set LUNA_LLM_API_KEY environment variable.",
        ))
    }
}

/// A client that returns a fixed string
#[derive(Debug, Clone)]
pub struct StaticClient {
    content: String,
}

impl StaticClient {
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

impl LLMClient for StaticClient {
    fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse> {
        Ok(CompletionResponse {
            content: self.content.clone(),
        })
    }
}

/// A client that pops response from a queue

#[derive(Debug, Default)]
pub struct MockClient {
    queue: std::sync::Mutex<std::collections::VecDeque<String>>,
}

impl MockClient {
    #[must_use]
    pub fn new(response: Vec<String>) -> Self {
        Self {
            queue: std::sync::Mutex::new(response.into()),
        }
    }

    pub fn push(&self, response: impl Into<String>) {
        let mut q = self.queue.lock().expect("mock queue lock");
        q.push_back(response.into());
    }
}

impl LLMClient for MockClient {
    fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse> {
        let mut q = self.queue.lock().expect("mock queue lock");
        match q.pop_front() {
            Some(s) => Ok(CompletionResponse { content: s }),
            None => Err(LunaError::invalid_input("MockClient queue is empty")),
        }
    }
}
