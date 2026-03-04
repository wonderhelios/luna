use crate::recorder::{NoopTrajectoryRecorder, TrajectoryRecorder};
use session::{InMemorySessionStore, SessionStore};
use std::sync::Arc;

/// Runtime-wide denpendency injection
pub struct RuntimeConfig {
    session_store: Arc<dyn SessionStore>,
    trajectory: Arc<dyn TrajectoryRecorder>,
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

    pub fn session_store(&self) -> Arc<dyn SessionStore> {
        Arc::clone(&self.session_store)
    }

    pub fn trajectory(&self) -> Arc<dyn TrajectoryRecorder> {
        Arc::clone(&self.trajectory)
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            session_store: Arc::new(InMemorySessionStore::new()),
            trajectory: Arc::new(NoopTrajectoryRecorder),
        }
    }
}
