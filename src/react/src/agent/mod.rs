//! ReAct Agent Public API
//!
//! 这个模块只负责对外暴露 ReAct agent 的类型与入口函数：
//! - `ReactOptions`：配置项
//! - `ReactAgent`：主 Agent 类型
//! - `react_ask`：便捷调用入口
//!
//! 具体的循环实现放在 `loop_impl` 中，安全相关状态和辅助方法放在 `safety` 中。

mod loop_impl;
mod safety;

pub use loop_impl::{react_ask, ReactAgent, ReactOptions};
pub use safety::ReActSafetyState;
