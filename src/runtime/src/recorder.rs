use crate::request::RunRequest;
use crate::response::RunResponse;
use session::Role;

#[derive(Debug, Clone)]
pub enum TrajectoryEvent {
    SessionCreated,
    SessionLoaded,
    MessageAppended { role: Role, bytes: usize },
}

// Trajectory recorder interface
pub trait TrajectoryRecorder: Send + Sync {
    fn on_run_start(&self, _req: &RunRequest) {}
    fn on_event(&self, _session_id: &str, _event: TrajectoryEvent) {}
    fn on_run_end(
        &self,
        _session_id: Option<&str>,
        _result: std::result::Result<&RunResponse, &anyhow::Error>,
    ) {
    }
}

pub struct NoopTrajectoryRecorder;

impl TrajectoryRecorder for NoopTrajectoryRecorder {}
