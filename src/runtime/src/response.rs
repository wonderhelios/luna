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

    /// TPAR: task classification completed for this turn
    TparTaskClassified {
        task: String,
    },
    /// TPAR: plan built for this trun
    TparPlanBuilt {
        plan: String,
    },
    /// TPAR: step start
    TparStepStarted {
        step_id: usize,
        step: String,
    },
    /// TPAR: step completed
    TparStepCompleted {
        step_id: usize,
        ok: bool,
    },
    /// TPAR: review/reflect phase completed
    TparReviewed {
        ok: bool,
    },

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

/// A sink for runtime events
pub trait EventSink {
    fn emit(&mut self, event: &RuntimeEvent);

    /// Some sinks are streaming-only and may not keep an in-memory buffer
    fn snapshot(&self) -> Option<&[RuntimeEvent]> {
        None
    }
}

impl EventSink for Vec<RuntimeEvent> {
    fn emit(&mut self, event: &RuntimeEvent) {
        self.push(event.clone());
    }
    fn snapshot(&self) -> Option<&[RuntimeEvent]> {
        Some(self.as_slice())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    pub request_id: String,
    pub session_id: String,
    pub output: String,
    pub events: Vec<RuntimeEvent>,
}
