use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeEvent {
    SessionCreated { session_id: String },
    SessionLoaded { session_id: String },
    UserMessageAppended,
    AssistantMessageAppended,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub request_id: String,
    pub session_id: String,
    pub output: String,
    pub events: Vec<RuntimeEvent>,
}
