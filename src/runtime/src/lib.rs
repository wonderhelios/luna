//! Luna Runtime - Core runtime

pub mod config;
pub mod intent;
pub mod recorder;
pub mod render;
pub mod request;
pub mod response;
pub mod router;
pub mod runtime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunMode {
    ChatTurn,
}

pub use {
    config::RuntimeConfig,
    recorder::{NoopTrajectoryRecorder, TrajectoryEvent, TrajectoryRecorder},
    request::{RequestMeta, RunRequest, SessionRef},
    response::{RunResponse, RuntimeEvent},
    runtime::LunaRuntime,
};
