//! TPAR (Task → Plan → Act → Review/Reflect) execution pipeline.
//!
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use error::ResultExt as _;

use crate::config::TokenBudget;
use crate::recorder::{TrajectoryRecorder, TrajectoryStep};
use crate::response::{EventSink, RuntimeEvent};
use crate::{intent, render, safety};

#[derive(Clone)]
pub struct TurnContext {
    pub session_id: String,
    pub request_id: String,
    pub cwd: Option<PathBuf>,
    pub safety_guard: Arc<dyn safety::SafetyGuard>,
    pub trajectory: Arc<dyn TrajectoryRecorder>,
    pub tools: Arc<tools::ToolRegistry>,
    pub budget: TokenBudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Query,
    Explain,
    Edit,
    Terminal,
    Chat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeEntityKind {
    Identifier,
    Path,
    Line,
    Command,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeEntity {
    pub kind: CodeEntityKind,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_type: TaskType,
    pub raw_input: String,
    pub intent: intent::Intent,
    pub entities: Vec<CodeEntity>,
}

impl Task {
    fn name(&self) -> &'static str {
        match self.task_type {
            TaskType::Query => "query",
            TaskType::Explain => "explain",
            TaskType::Edit => "edit",
            TaskType::Terminal => "terminal",
            TaskType::Chat => "chat",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub estimated_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanStep {
    ToolCall {
        call: tools::ToolCall,
    },
    Intelligence {
        style: render::RenderStyle,
        query: String,
    },
    Echo {
        text: String,
    },
}

impl PlanStep {
    fn label(&self) -> String {
        match self {
            PlanStep::ToolCall { call } => format!("tool:{}", call.name),
            PlanStep::Intelligence { style, .. } => match style {
                render::RenderStyle::Navigation => "intelligence:navigation".to_owned(),
                render::RenderStyle::Explain => "intelligence:explain".to_owned(),
            },
            PlanStep::Echo { .. } => "echo".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub ok: bool,
    pub output: String,
}

#[derive(Debug, Clone)]
pub enum ReviewResult {
    Success,
    NeedsRevision { reason: String },
    NeedsRollback { reason: String },
}

/// Run a single TPAR turn.
pub fn run_turn(
    user_input: &str,
    ctx: TurnContext,
    events: &mut dyn EventSink,
) -> error::Result<String> {
    if user_input.chars().count() > ctx.budget.max_input_chars {
        let msg = format!(
            "❌ Input too long: exceeds max_input_chars={}. Please shorten or split your request.",
            ctx.budget.max_input_chars
        );
        events.emit(&RuntimeEvent::TparTaskClassified {
            task: "rejected".to_owned(),
        });
        events.emit(&RuntimeEvent::TparReviewed { ok: false });
        return Ok(msg);
    }

    // Task
    let task = TaskAnalyzer::analyze(user_input);
    events.emit(&RuntimeEvent::TparTaskClassified {
        task: task.name().to_owned(),
    });

    // Plan
    let plan = RuleBasedPlanner::plan(&task, ctx.budget.max_steps)?;
    events.emit(&RuntimeEvent::TparPlanBuilt {
        plan: format!("steps={}", plan.steps.len()),
    });

    // Act
    let mut exec = ActExecutor::new(
        &ctx.session_id,
        &ctx.request_id,
        ctx.cwd.as_deref(),
        ctx.safety_guard,
        ctx.trajectory,
        ctx.tools,
        ctx.budget,
    );
    let (out, review) = exec.execute(&plan, &task, events)?;

    // Review/Reflect
    let ok = matches!(review, ReviewResult::Success);
    events.emit(&RuntimeEvent::TparReviewed { ok });

    Ok(out)
}

struct TaskAnalyzer;

impl TaskAnalyzer {
    fn analyze(input: &str) -> Task {
        let raw = input.trim().to_owned();

        if let Some((path, line_1, new_line)) = parse_edit_line_to(input) {
            return Task {
                task_type: TaskType::Edit,
                raw_input: raw,
                intent: intent::Intent::Other,
                entities: vec![
                    CodeEntity {
                        kind: CodeEntityKind::Path,
                        value: path,
                    },
                    CodeEntity {
                        kind: CodeEntityKind::Line,
                        value: line_1.to_string(),
                    },
                    CodeEntity {
                        kind: CodeEntityKind::Identifier,
                        value: new_line,
                    },
                ],
            };
        }

        if let Some((path, line_1)) = parse_edit_intent_min(input) {
            return Task {
                task_type: TaskType::Edit,
                raw_input: raw,
                intent: intent::Intent::Other,
                entities: vec![
                    CodeEntity {
                        kind: CodeEntityKind::Path,
                        value: path,
                    },
                    CodeEntity {
                        kind: CodeEntityKind::Line,
                        value: line_1.to_string(),
                    },
                ],
            };
        }

        if let Some(cmd) = parse_terminal_intent(input) {
            return Task {
                task_type: TaskType::Terminal,
                raw_input: raw,
                intent: intent::Intent::Other,
                entities: vec![CodeEntity {
                    kind: CodeEntityKind::Command,
                    value: cmd,
                }],
            };
        }

        let inferred_intent = intent::classify_intent(input);
        let task_type = match inferred_intent {
            intent::Intent::SymbolNavigation => TaskType::Query,
            intent::Intent::ExplainSymbol => TaskType::Explain,
            intent::Intent::Other => TaskType::Chat,
        };
        let entities = intent::extract_identifiers_dedup(input)
            .into_iter()
            .take(8)
            .map(|s| CodeEntity {
                kind: CodeEntityKind::Identifier,
                value: s.to_owned(),
            })
            .collect::<Vec<_>>();
        Task {
            task_type,
            raw_input: raw,
            intent: inferred_intent,
            entities,
        }
    }
}

struct RuleBasedPlanner;

impl RuleBasedPlanner {
    fn plan(task: &Task, max_steps: usize) -> error::Result<Plan> {
        let mut steps = Vec::<PlanStep>::new();

        match task.task_type {
            TaskType::Query => {
                steps.push(PlanStep::Intelligence {
                    style: render::RenderStyle::Navigation,
                    query: task.raw_input.clone(),
                });
            }
            TaskType::Explain => {
                steps.push(PlanStep::Intelligence {
                    style: render::RenderStyle::Explain,
                    query: task.raw_input.clone(),
                });
            }
            TaskType::Edit => {
                let path = task
                    .entities
                    .iter()
                    .find(|e| e.kind == CodeEntityKind::Path)
                    .map(|e| e.value.clone())
                    .unwrap_or_default();
                steps.push(PlanStep::ToolCall {
                    call: tools::ToolCall {
                        name: "read_file".to_owned(),
                        args: serde_json::json!({ "path": path }),
                    },
                });

                // If we have explicit replacement, plan a real edit.
                if let Some((line_1, new_line)) = extract_edit_payload(task) {
                    steps.push(PlanStep::ToolCall {
                        call: tools::ToolCall {
                            name: "edit_file".to_owned(),
                            args: serde_json::json!({ "path": path, "line_1": line_1, "new_line": new_line }),
                        },
                    });
                    steps.push(PlanStep::ToolCall {
                        call: tools::ToolCall {
                            name: "read_file".to_owned(),
                            args: serde_json::json!({ "path": path }),
                        },
                    });
                } else {
                    // Still produce an edit step, but it will fail fast with a clear error.
                    steps.push(PlanStep::ToolCall {
                        call: tools::ToolCall {
                            name: "edit_file".to_owned(),
                            args: serde_json::json!({ "path": path }),
                        },
                    });
                }
            }
            TaskType::Terminal => {
                let cmd = task
                    .entities
                    .iter()
                    .find(|e| e.kind == CodeEntityKind::Command)
                    .map(|e| e.value.clone())
                    .unwrap_or_default();
                steps.push(PlanStep::ToolCall {
                    call: tools::ToolCall {
                        name: "run_terminal".to_owned(),
                        args: serde_json::json!({ "cmd": cmd }),
                    },
                });
            }
            TaskType::Chat => {
                steps.push(PlanStep::Echo {
                    text: format!("received: {}", task.raw_input),
                });
            }
        }

        if steps.len() > max_steps {
            return Err(error::LunaError::invalid_input(format!(
                "planned steps too many: {} > {}",
                steps.len(),
                max_steps
            )));
        }

        let estimated_tokens = estimate_tokens(&task.raw_input) + steps.len() * 30;
        Ok(Plan {
            steps,
            estimated_tokens,
        })
    }
}

struct ActExecutor {
    session_id: String,
    request_id: String,
    cwd: Option<PathBuf>,
    safety_guard: Arc<dyn safety::SafetyGuard>,
    trajectory: Arc<dyn TrajectoryRecorder>,
    tools: Arc<tools::ToolRegistry>,
    budget: TokenBudget,
    // For rollback: keep original content for any edited files.
    original_files: HashMap<PathBuf, String>,
}

impl ActExecutor {
    fn new(
        session_id: &str,
        request_id: &str,
        cwd: Option<&Path>,
        safety_guard: Arc<dyn safety::SafetyGuard>,
        trajectory: Arc<dyn TrajectoryRecorder>,
        tools: Arc<tools::ToolRegistry>,
        budget: TokenBudget,
    ) -> Self {
        Self {
            session_id: session_id.to_owned(),
            request_id: request_id.to_owned(),
            cwd: cwd.map(|p| p.to_path_buf()),
            safety_guard,
            trajectory,
            tools,
            budget,
            original_files: HashMap::new(),
        }
    }

    fn execute(
        &mut self,
        plan: &Plan,
        task: &Task,
        events: &mut dyn EventSink,
    ) -> error::Result<(String, ReviewResult)> {
        let repo_root = crate::router::resolve_repo_root(self.cwd.as_deref());
        let tool_ctx = tools::ToolContext {
            repo_root: repo_root.clone(),
            cwd: self.cwd.clone(),
            max_bytes: self.budget.max_io_bytes,
        };

        let mut final_output = String::new();

        for (i, step) in plan.steps.iter().enumerate() {
            let step_id = i + 1;
            events.emit(&RuntimeEvent::TparStepStarted {
                step_id,
                step: step.label(),
            });

            let outcome = self.execute_step(step, task, &tool_ctx, repo_root.as_deref(), events);

            let (ok, out_text, review) = match outcome {
                Ok(v) => (v.ok, v.output, None),
                Err(err) => {
                    let reason = format!("Step {step_id} failed: {err}");
                    // If we already edited something, rollback.
                    let needs_rollback = !self.original_files.is_empty();
                    let review = if needs_rollback {
                        Some(ReviewResult::NeedsRollback { reason })
                    } else {
                        Some(ReviewResult::NeedsRevision { reason })
                    };
                    (false, String::new(), review)
                }
            };

            events.emit(&RuntimeEvent::TparStepCompleted { step_id, ok });
            self.trajectory.on_step(&TrajectoryStep {
                ts_ms: now_micros(),
                session_id: self.session_id.clone(),
                request_id: self.request_id.clone(),
                state: serde_json::json!({
                    "task_type": task.task_type,
                    "step_id": step_id,
                    "step": step.label(),
                }),
                action: serde_json::to_value(step).unwrap_or(Value::Null),
                reward: if ok { 0.2 } else { -0.5 },
                outcome: serde_json::json!({ "ok": ok, "output_len": out_text.len() }),
            });

            if !out_text.is_empty() {
                if !final_output.is_empty() {
                    final_output.push('\n');
                }
                final_output.push_str(&out_text);
            }

            if let Some(review) = review {
                if matches!(review, ReviewResult::NeedsRollback { .. }) {
                    let _ = self.rollback(&tool_ctx);
                }
                let msg = match &review {
                    ReviewResult::NeedsRevision { reason } => format!("❌ 需要重试：{reason}"),
                    ReviewResult::NeedsRollback { reason } => {
                        format!("❌ 已回滚：{reason}")
                    }
                    ReviewResult::Success => "".to_owned(),
                };
                if !msg.is_empty() {
                    if !final_output.is_empty() {
                        final_output.push('\n');
                    }
                    final_output.push_str(&msg);
                }
                return Ok((final_output, review));
            }
        }

        if final_output.is_empty() {
            final_output = format!("received: {}", task.raw_input);
        }
        Ok((final_output, ReviewResult::Success))
    }

    fn execute_step(
        &mut self,
        step: &PlanStep,
        task: &Task,
        tool_ctx: &tools::ToolContext,
        repo_root: Option<&Path>,
        events: &mut dyn EventSink,
    ) -> error::Result<StepOutcome> {
        match step {
            PlanStep::Echo { text } => Ok(StepOutcome {
                ok: true,
                output: text.clone(),
            }),
            PlanStep::Intelligence { style: _, query } => {
                let router = crate::router::RuntimeRouter::default();
                let out = router
                    .maybe_handle(query, repo_root.or(self.cwd.as_deref()), events)?
                    .unwrap_or_else(|| format!("received: {query}"));
                Ok(StepOutcome {
                    ok: true,
                    output: out,
                })
            }
            PlanStep::ToolCall { call } => {
                self.check_step_safety(task, call)?;

                // For rollback: snapshot before editing.
                if call.name == "edit_file" {
                    if let Some(path) = call.args.get("path").and_then(|v| v.as_str()) {
                        let abs = tool_ctx.resolve_path(Path::new(path));
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            self.original_files.entry(abs.clone())
                        {
                            if let Ok(s) = std::fs::read_to_string(&abs) {
                                e.insert(s);
                            }
                        }
                    }
                }

                let res = self.tools.run(tool_ctx, call)?;
                if res.ok {
                    Ok(StepOutcome {
                        ok: true,
                        output: res.stdout,
                    })
                } else {
                    Err(error::LunaError::invalid_input(res.stderr))
                }
            }
        }
    }

    fn check_step_safety(&self, task: &Task, call: &tools::ToolCall) -> error::Result<()> {
        let ctx = safety::SafetyContext {
            session_id: self.session_id.clone(),
        };
        let action = match call.name.as_str() {
            "run_terminal" => safety::Action {
                kind: safety::ActionKind::Terminal,
                payload: serde_json::json!({
                    "cmd": call.args.get("cmd").and_then(|v| v.as_str()).unwrap_or_default(),
                }),
            },
            "edit_file" => safety::Action {
                kind: safety::ActionKind::EditFile,
                payload: call.args.clone(),
            },
            _ => safety::Action {
                kind: safety::ActionKind::Command,
                payload: serde_json::json!({ "tool": call.name, "task": task.task_type }),
            },
        };

        match self.safety_guard.check(&ctx, &action) {
            safety::SafetyDecision::Allow => {
                self.safety_guard.record(&ctx, &action);
                Ok(())
            }
            safety::SafetyDecision::Warn { msg } => {
                // Phase2 MVP: warn and stop (requires user confirmation / revised plan).
                self.safety_guard.record(&ctx, &action);
                Err(error::LunaError::invalid_input(msg))
            }
            safety::SafetyDecision::Deny { msg } => Err(error::LunaError::invalid_input(msg)),
        }
    }

    fn rollback(&mut self, tool_ctx: &tools::ToolContext) -> error::Result<()> {
        for (abs, content) in self.original_files.drain() {
            let _ = tool_ctx; // keep signature stable.
            std::fs::write(&abs, content)
                .map_err(|e| error::LunaError::io(Some(abs.clone()), e))
                .with_context(|| format!("rollback write: {}", abs.display()))?;
        }
        Ok(())
    }
}

fn extract_edit_payload(task: &Task) -> Option<(usize, String)> {
    let line_1 = task
        .entities
        .iter()
        .find(|e| e.kind == CodeEntityKind::Line)
        .and_then(|e| e.value.parse::<usize>().ok());
    let new_line = task
        .entities
        .iter()
        .find(|e| e.kind == CodeEntityKind::Identifier)
        .map(|e| e.value.clone());
    match (line_1, new_line) {
        (Some(l), Some(s)) => Some((l, s)),
        _ => None,
    }
}

fn parse_terminal_intent(input: &str) -> Option<String> {
    let s = input.trim();
    for prefix in ["运行 ", "执行 ", "run "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            let cmd = rest.trim();
            if !cmd.is_empty() {
                return Some(cmd.to_owned());
            }
        }
    }
    None
}

fn parse_edit_intent_min(input: &str) -> Option<(String, usize)> {
    // "修改 <path> 第 <line> 行"
    let s = input.trim();
    let rest = s.strip_prefix("修改 ")?;
    let idx = rest.find('第')?;
    let (path_part, rest) = rest.split_at(idx);
    let path = path_part.trim();
    if path.is_empty() {
        return None;
    }
    let rest = rest.strip_prefix('第')?.trim_start();
    let mut digits = String::new();
    for ch in rest.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        break;
    }
    let line_1 = digits.parse::<usize>().ok()?;
    if line_1 == 0 || !rest.contains('行') {
        return None;
    }
    Some((path.to_owned(), line_1))
}

fn parse_edit_line_to(input: &str) -> Option<(String, usize, String)> {
    // Supported:
    // - "修改 <path> 第 <line> 行 为 <new_line>"
    // - "修改 <path> 第 <line> 行 成 <new_line>"
    let s = input.trim();
    let rest = s.strip_prefix("修改 ")?;
    let idx = rest.find('第')?;
    let (path_part, rest) = rest.split_at(idx);
    let path = path_part.trim();
    if path.is_empty() {
        return None;
    }
    let rest = rest.strip_prefix('第')?.trim_start();
    let mut digits = String::new();
    let mut consumed = 0usize;
    for (i, ch) in rest.chars().enumerate() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            consumed = i + ch.len_utf8();
            continue;
        }
        break;
    }
    let line_1 = digits.parse::<usize>().ok()?;
    if line_1 == 0 {
        return None;
    }
    let after_num = rest.get(consumed..)?.trim_start();
    let after_line = after_num.strip_prefix("行")?.trim_start();
    let marker = if let Some(v) = after_line.strip_prefix("为") {
        v
    } else if let Some(v) = after_line.strip_prefix("成") {
        v
    } else if let Some(v) = after_line.strip_prefix("= ") {
        v
    } else {
        return None;
    };
    let new_line = marker.trim();
    if new_line.is_empty() {
        return None;
    }
    Some((path.to_owned(), line_1, new_line.to_owned()))
}

fn estimate_tokens(s: &str) -> usize {
    // Very rough heuristic: 1 token ~= 4 chars (ASCII-ish).
    s.chars().count().div_ceil(4)
}

// repo root resolution is shared with runtime router.

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::NoopTrajectoryRecorder;
    use crate::router::RuntimeRouter;
    use crate::safety::RuleBasedSafetyGuard;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("luna_tpar_{name}_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tpar_edit_file_line_replacement_works() {
        let dir = tmp_dir("edit");
        let file = dir.join("a.txt");
        std::fs::write(&file, "hello\nworld\n").unwrap();

        let mut events = Vec::new();
        let out = run_turn(
            &format!("修改 {} 第 2 行 为 WORLD", file.display()),
            TurnContext {
                session_id: "local:test".to_owned(),
                request_id: "req:test".to_owned(),
                cwd: Some(dir.clone()),
                safety_guard: Arc::new(RuleBasedSafetyGuard::new(8)),
                trajectory: Arc::new(NoopTrajectoryRecorder),
                tools: Arc::new(tools::ToolRegistry::new()),
                budget: TokenBudget {
                    max_input_chars: 2048,
                    max_io_bytes: 1024,
                    max_steps: 8,
                },
            },
            &mut events,
        )
        .unwrap();

        assert!(out.contains("edited:"), "out={out}");
        let updated = std::fs::read_to_string(&file).unwrap();
        assert_eq!(updated, "hello\nWORLD\n");
    }

    #[test]
    fn tpar_dangerous_terminal_is_denied() {
        let dir = tmp_dir("term");
        let mut events = Vec::new();
        let out = run_turn(
            "运行 rm -rf /",
            TurnContext {
                session_id: "local:test".to_owned(),
                request_id: "req:test".to_owned(),
                cwd: Some(dir),
                safety_guard: Arc::new(RuleBasedSafetyGuard::new(8)),
                trajectory: Arc::new(NoopTrajectoryRecorder),
                tools: Arc::new(tools::ToolRegistry::new()),
                budget: TokenBudget {
                    max_input_chars: 2048,
                    max_io_bytes: 1024,
                    max_steps: 8,
                },
            },
            &mut events,
        )
        .unwrap();
        assert!(out.contains("危险命令拦截"), "out={out}");
    }

    #[test]
    fn phase1_compare_scopegraph_vs_text_search() {
        // Create a minimal git-like repo root so router can resolve it.
        let dir = tmp_dir("cmp");
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src/lib.rs");
        std::fs::write(
            &file,
            "pub fn foo() -> i32 { 1 }\n\nfn call() { let _ = foo(); }\n",
        )
        .unwrap();

        // Naive text search sees multiple occurrences.
        let content = std::fs::read_to_string(&file).unwrap();
        let occ = content.matches("foo").count();
        assert!(occ >= 2, "expected >=2 text matches, got {occ}");

        // ScopeGraph-based router should resolve the definition.
        let mut events = Vec::new();
        let router = RuntimeRouter::default();
        let out = router
            .maybe_handle("foo 在哪里定义", Some(dir.as_path()), &mut events)
            .unwrap()
            .unwrap();
        assert!(out.contains("✅"), "out={out}");
    }

    #[test]
    #[ignore]
    fn bench_scopegraph_goto_definition_latency_smoke() {
        let dir = tmp_dir("bench");
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src/lib.rs");
        std::fs::write(&file, "pub fn foo() -> i32 { 1 }\n").unwrap();

        let router = RuntimeRouter::default();
        let iters = 30;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut events = Vec::new();
            let _ = router
                .maybe_handle("foo 在哪里定义", Some(dir.as_path()), &mut events)
                .unwrap();
        }
        let dur = start.elapsed();
        eprintln!("avg goto_definition: {:?}", dur / iters);
    }
}
