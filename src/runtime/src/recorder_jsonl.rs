use crate::recorder::{TrajectoryEvent, TrajectoryRecorder, TrajectoryStep};
use error::{LunaError, Result};
use std::{fs, io::Write, path::PathBuf};

/// Append-only JSONL trajectory recorder.
///
/// Layout: `<LUNA_HOME>/trajectories/<session_id>.jsonl`.
#[derive(Debug, Clone)]
pub struct JsonlTrajectoryRecorder {
    base_dir: PathBuf,
}

impl JsonlTrajectoryRecorder {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    pub fn try_default() -> Option<Self> {
        let home = session::LunaHome::from_env()?;
        Some(Self::new(home.base_dir().to_path_buf()))
    }

    fn dir(&self) -> PathBuf {
        self.base_dir.join("trajectories")
    }

    fn path_for(&self, session_id: &str) -> PathBuf {
        self.dir().join(format!("{session_id}.jsonl"))
    }

    fn ensure_dir(&self) -> Result<()> {
        let dir = self.dir();
        fs::create_dir_all(&dir).map_err(|e| LunaError::io(Some(dir), e))?;
        Ok(())
    }

    fn append_line(&self, session_id: &str, v: &serde_json::Value) -> Result<()> {
        self.ensure_dir()?;
        let path = self.path_for(session_id);
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| LunaError::io(Some(path.clone()), e))?;
        let line = serde_json::to_string(v)?;
        f.write_all(line.as_bytes())
            .map_err(|e| LunaError::io(Some(path.clone()), e))?;
        f.write_all(b"\n")
            .map_err(|e| LunaError::io(Some(path), e))?;
        Ok(())
    }

    fn write_step(&self, step: &TrajectoryStep) -> Result<()> {
        let v = serde_json::to_value(step)?;
        self.append_line(&step.session_id, &v)
    }

    fn write_event(&self, session_id: &str, event: &TrajectoryEvent) -> Result<()> {
        // Keep events as a simple JSON object too.
        let v = serde_json::json!({
            "type": "event",
            "session_id": session_id,
            "event": event,
        });
        self.append_line(session_id, &v)
    }
}

impl TrajectoryRecorder for JsonlTrajectoryRecorder {
    fn on_event(&self, session_id: &str, event: TrajectoryEvent) {
        if let Err(err) = self.write_event(session_id, &event) {
            tracing::warn!("trajectory recorder failed to write event: {err}");
        }
    }

    fn on_step(&self, step: &TrajectoryStep) {
        if let Err(err) = self.write_step(step) {
            tracing::warn!("trajectory recorder failed to write step: {err}");
        }
    }
}
