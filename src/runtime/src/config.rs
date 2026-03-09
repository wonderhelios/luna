use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;

use session::{InMemorySessionStore, JsonlSessionStore, SessionStore};
use tools::ToolRegistry;

use crate::intent_classifier::{ClassifierConfig, ClassifierKind, IntentClassifier};
use crate::planner;
use crate::recorder::{NoopTrajectoryRecorder, TrajectoryRecorder};
use crate::recorder_jsonl::JsonlTrajectoryRecorder;
use crate::safety::{RuleBasedSafetyGuard, SafetyGuard};

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
    planner: Arc<dyn planner::TaskPlanner>,
    intent_classifier: Arc<dyn IntentClassifier>,
    memory: memory::MemoryStore,
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

    pub fn with_planner(mut self, planner: Arc<dyn planner::TaskPlanner>) -> Self {
        self.planner = planner;
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
    pub fn planner(&self) -> Arc<dyn planner::TaskPlanner> {
        Arc::clone(&self.planner)
    }
    pub fn intent_classifier(&self) -> Arc<dyn IntentClassifier> {
        Arc::clone(&self.intent_classifier)
    }

    pub fn with_intent_classifier(mut self, classifier: Arc<dyn IntentClassifier>) -> Self {
        self.intent_classifier = classifier;
        self
    }

    pub fn memory(&self) -> &memory::MemoryStore {
        &self.memory
    }

    pub fn with_memory(mut self, memory: memory::MemoryStore) -> Self {
        self.memory = memory;
        self
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

        // Try to create real LLM client from env
        let llm_client: Arc<dyn llm::LLMClient> =
            llm::OpenAIClient::try_from_env()
                .map(|c| Arc::new(c) as Arc<dyn llm::LLMClient>)
                .unwrap_or_else(|| Arc::new(llm::DisabledClient));

        // Auto-enable LLM planner if client is configured, or respect explicit LUNA_PLANNER setting
        let has_llm_client = llm_client.as_ref().type_id() != std::any::TypeId::of::<llm::DisabledClient>();
        let prefer_llm = std::env::var("LUNA_PLANNER")
            .ok()
            .map(|v| v.eq_ignore_ascii_case("llm"))
            .unwrap_or(has_llm_client); // Auto-enable if LLM client is available

        let rule = Arc::new(planner::RuleBasedPlanner::new()) as Arc<dyn planner::TaskPlanner>;
        let llm_planner = Arc::new(planner::LLMBasedPlanner::new(Arc::clone(&llm_client), 12))
            as Arc<dyn planner::TaskPlanner>;
        let planner: Arc<dyn planner::TaskPlanner> =
            Arc::new(planner::PlannerSelector::new(prefer_llm, rule, llm_planner));

        // Initialize intent classifier based on environment
        let classifier_config = ClassifierConfig {
            kind: if has_llm_client {
                ClassifierKind::Hybrid
            } else {
                ClassifierKind::RuleBased
            },
            ..Default::default()
        };
        let intent_classifier = crate::intent_classifier::create_classifier(
            &classifier_config,
            if has_llm_client { Some(Arc::clone(&llm_client)) } else { None }
        );

        // Initialize memory store
        let memory = memory::auto_detect(
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        );

        Self {
            session_store,
            trajectory,
            safety,
            tools,
            budget: TokenBudget::default(),
            planner,
            intent_classifier,
            memory,
        }
    }
}
