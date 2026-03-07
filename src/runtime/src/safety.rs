use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    Command,
    EditFile,
    Terminal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub kind: ActionKind,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyDecision {
    Allow,
    Warn { msg: String },
    Deny { msg: String },
}

#[derive(Debug, Clone)]
pub struct SafetyContext {
    pub session_id: String,
}

pub trait SafetyGuard: Send + Sync {
    fn check(&self, ctx: &SafetyContext, action: &Action) -> SafetyDecision;
    fn record(&self, ctx: &SafetyContext, action: &Action);
}

/// A minimal rule-based safety guard for Phase-2.
///
/// - Duplicate edits: warn if the same edit intent repeats in recent history.
/// - Dangerous terminal commands: deny obvious destructive patterns.
#[derive(Debug, Default)]
pub struct RuleBasedSafetyGuard {
    recent: Mutex<HashMap<String, VecDeque<String>>>,
    max_recent: usize,
}

impl RuleBasedSafetyGuard {
    pub fn new(max_recent: usize) -> Self {
        Self {
            recent: Mutex::new(HashMap::new()),
            max_recent,
        }
    }

    fn digest(action: &Action) -> String {
        // Stable-ish: kind + JSON.
        // Phase2 不追求最优哈希，只要能稳定命中重复即可。
        let payload = serde_json::to_string(&action.payload).unwrap_or_default();
        format!("{:?}:{payload}", action.kind)
    }

    fn is_dangerous_terminal(cmd: &str) -> Option<&'static str> {
        let s = cmd.trim();
        let lower = s.to_ascii_lowercase();
        // Extremely conservative deny list.
        [
            "rm -rf /", "mkfs", "dd if=", "shutdown", "reboot", "curl", "wget", "| sh", "|bash",
            "sudo ",
        ]
        .into_iter()
        .find(|pat| lower.contains(pat))
    }
}

impl SafetyGuard for RuleBasedSafetyGuard {
    fn check(&self, ctx: &SafetyContext, action: &Action) -> SafetyDecision {
        match action.kind {
            ActionKind::Terminal => {
                if let Some(cmd) = action.payload.get("cmd").and_then(|v| v.as_str()) {
                    if let Some(pat) = Self::is_dangerous_terminal(cmd) {
                        return SafetyDecision::Deny {
                            msg: format!("危险命令拦截：命中 `{pat}`"),
                        };
                    }
                }
                SafetyDecision::Allow
            }
            ActionKind::EditFile => {
                let d = Self::digest(action);
                let guard = self.recent.lock();
                if let Some(q) = guard.get(&ctx.session_id) {
                    if q.iter().any(|x| x == &d) {
                        return SafetyDecision::Warn {
                            msg: "⚠️ Warning: 重复编辑，请检查逻辑".to_owned(),
                        };
                    }
                }
                SafetyDecision::Allow
            }
            ActionKind::Command => SafetyDecision::Allow,
        }
    }

    fn record(&self, ctx: &SafetyContext, action: &Action) {
        let d = Self::digest(action);
        let mut guard = self.recent.lock();
        let q = guard.entry(ctx.session_id.clone()).or_default();
        q.push_back(d);
        while q.len() > self.max_recent {
            q.pop_front();
        }
    }
}
