//! Luna Runtime - Core runtime

pub mod command;
pub mod config;
pub mod context_bridge;
pub mod intent;
pub mod planner;
pub mod recorder;
pub mod recorder_jsonl;
pub mod refill_trigger;
pub mod render;
pub mod request;
pub mod response;
pub mod router;
pub mod runtime;
pub mod safety;
pub mod tpar;

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
