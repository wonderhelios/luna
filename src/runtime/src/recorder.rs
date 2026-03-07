use crate::request::RunRequest;
use crate::response::RunResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use session::Role;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrajectoryEvent {
    SessionCreated,
    SessionLoaded,
    MessageAppended { role: Role, bytes: usize },
}

/// A structured training-ready step: (state, action, reward, outcome).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub ts_ms: u64,
    pub session_id: String,
    pub request_id: String,
    pub state: Value,
    pub action: Value,
    pub reward: f32,
    pub outcome: Value,
}

// Trajectory recorder interface
pub trait TrajectoryRecorder: Send + Sync {
    fn on_run_start(&self, _req: &RunRequest) {}
    fn on_event(&self, _session_id: &str, _event: TrajectoryEvent) {}
    fn on_step(&self, _step: &TrajectoryStep) {}
    fn on_run_end(
        &self,
        _session_id: Option<&str>,
        _result: std::result::Result<&RunResponse, &error::LunaError>,
    ) {
    }
}

pub struct NoopTrajectoryRecorder;

impl TrajectoryRecorder for NoopTrajectoryRecorder {}
