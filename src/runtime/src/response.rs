use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeEvent {
    SessionCreated {
        session_id: String,
    },
    SessionLoaded {
        session_id: String,
    },
    UserMessageAppended,
    AssistantMessageAppended,

    /// Runtime detected a symbol query in the user input.
    FoundIdentifier {
        name: String,
    },
    /// ScopeGraph-based search started.
    ScopeGraphSearchStarted {
        repo_root: String,
    },
    /// ScopeGraph-based search completed.
    ScopeGraphSearchCompleted {
        matches: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub request_id: String,
    pub session_id: String,
    pub output: String,
    pub events: Vec<RuntimeEvent>,
}
