use crate::recorder::{NoopTrajectoryRecorder, TrajectoryRecorder};
use crate::recorder_jsonl::JsonlTrajectoryRecorder;
use crate::safety::{RuleBasedSafetyGuard, SafetyGuard};
use session::{InMemorySessionStore, JsonlSessionStore, SessionStore};
use std::sync::Arc;
use tools::ToolRegistry;

#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Rough guard for user input size.
    pub max_input_chars: usize,
    /// Maximum bytes read from files and terminal output.
    pub max_io_bytes: usize,
    /// Maximum planned step count for a single turn.
    pub max_steps: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_input_chars: 32_000,
            max_io_bytes: 64 * 1024,
            max_steps: 12,
        }
    }
}

/// Runtime-wide denpendency injection
pub struct RuntimeConfig {
    session_store: Arc<dyn SessionStore>,
    trajectory: Arc<dyn TrajectoryRecorder>,
    safety: Arc<dyn SafetyGuard>,
    tools: Arc<ToolRegistry>,
    budget: TokenBudget,
}

impl RuntimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session_store(mut self, session_store: Arc<dyn SessionStore>) -> Self {
        self.session_store = session_store;
        self
    }

    pub fn with_trajectory(mut self, trajectory: Arc<dyn TrajectoryRecorder>) -> Self {
        self.trajectory = trajectory;
        self
    }

    pub fn with_safety(mut self, safety: Arc<dyn SafetyGuard>) -> Self {
        self.safety = safety;
        self
    }

    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn session_store(&self) -> Arc<dyn SessionStore> {
        Arc::clone(&self.session_store)
    }

    pub fn trajectory(&self) -> Arc<dyn TrajectoryRecorder> {
        Arc::clone(&self.trajectory)
    }

    pub fn safety(&self) -> Arc<dyn SafetyGuard> {
        Arc::clone(&self.safety)
    }

    pub fn tools(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.tools)
    }

    pub fn budget(&self) -> TokenBudget {
        self.budget.clone()
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let session_store: Arc<dyn SessionStore> = JsonlSessionStore::try_default()
            .map(|s| Arc::new(s) as Arc<dyn SessionStore>)
            .unwrap_or_else(|| Arc::new(InMemorySessionStore::new()));
        let trajectory: Arc<dyn TrajectoryRecorder> = JsonlTrajectoryRecorder::try_default()
            .map(|r| Arc::new(r) as Arc<dyn TrajectoryRecorder>)
            .unwrap_or_else(|| Arc::new(NoopTrajectoryRecorder));
        let safety: Arc<dyn SafetyGuard> = Arc::new(RuleBasedSafetyGuard::new(32));
        let tools = Arc::new(ToolRegistry::new());
        Self {
            session_store,
            trajectory,
            safety,
            tools,
            budget: TokenBudget::default(),
        }
    }
}
