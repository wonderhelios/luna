use crate::RunMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionRef {
    New { title: Option<String> },
    Existing { session_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMeta {
    pub trace: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    pub request_id: String,
    pub mode: RunMode,
    pub session: SessionRef,
    pub input: String,
    pub cwd: Option<PathBuf>,
    pub meta: RequestMeta,
}

impl RunRequest {
    pub fn chat_turn(session: SessionRef, input: impl Into<String>) -> Self {
        Self {
            request_id: session::gen_id("req"),
            mode: RunMode::ChatTurn,
            session,
            input: input.into(),
            cwd: None,
            meta: RequestMeta { trace: true },
        }
    }

    pub fn with_cwd(mut self, cwd: PathBuf) -> Self {
        self.cwd = Some(cwd);
        self
    }
}
