//! ReAct Agent Public API
//!
//! This module only exposes ReAct agent types and entry functions:
//! - `ReactOptions`: configuration options
//! - `ReactAgent`: main Agent type
//! - `react_ask`: convenient entry point
//! - `FixLoop`: automatic fix loop (M3)
//! - `AgentLoop` trait: generic loop framework
//!
//! Concrete loop implementations are in `loop_impl`, safety-related state and helper methods are in `safety`.

mod fix_loop;
mod loop_framework;
mod loop_impl;
mod safety;

pub use fix_loop::{
    run_fix_loop, ConvergedReason, FixAction, FixActionResult, FixIteration, FixLoop, FixLoopConfig,
    FixLoopResult,
};
pub use loop_framework::{AgentLoop, LoopRunner, LoopRunnerConfig, NotConvergedReason};
pub use loop_impl::{react_ask, ReactAgent, ReactOptions};
pub use safety::ReActSafetyState;
