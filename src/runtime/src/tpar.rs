//! TPAR (Task → Plan → Act → Review/Reflect) execution pipeline.
//!
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use error::{LunaError, ResultExt as _};

use crate::config::TokenBudget;
use crate::intent_classifier::{ClassificationContext, Intent, IntentClassifier};
use crate::planner::{PlannerContext, TaskPlanner};
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
    pub planner: Arc<dyn TaskPlanner>,
    /// Intent classifier for understanding user input
    pub intent_classifier: Arc<dyn IntentClassifier>,
    /// Project memory (learned commands, preferences)
    pub memory: Option<memory::ProjectMemory>,
    /// Memory store for persistence
    pub memory_store: Option<memory::MemoryStore>,
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

    /// Create a Task from ClassificationResult (new intent classifier)
    fn from_classification(result: &crate::intent_classifier::ClassificationResult) -> Self {
        use crate::intent_classifier::{EntityKind, Intent};

        let task_type = match &result.intent {
            Intent::SymbolNavigation { .. } => TaskType::Query,
            Intent::ExplainSymbol { .. } => TaskType::Explain,
            Intent::Edit { .. } => TaskType::Edit,
            Intent::Terminal { .. } => TaskType::Terminal,
            Intent::Search { .. } => TaskType::Query, // Map Search to Query for now
            Intent::Chat | Intent::Help | Intent::ClarificationNeeded { .. } => TaskType::Chat,
        };

        // Convert new entities to old CodeEntity format
        let entities: Vec<CodeEntity> = result
            .entities
            .iter()
            .map(|e| {
                let kind = match e.kind {
                    EntityKind::Identifier => CodeEntityKind::Identifier,
                    EntityKind::Path => CodeEntityKind::Path,
                    EntityKind::Line => CodeEntityKind::Line,
                    EntityKind::Command => CodeEntityKind::Command,
                    _ => CodeEntityKind::Identifier, // Default to Identifier for other kinds
                };
                CodeEntity {
                    kind,
                    value: e.value.clone(),
                }
            })
            .collect();

        // Map new Intent to old intent::Intent for backward compatibility
        let old_intent = match &result.intent {
            Intent::SymbolNavigation { .. } => intent::Intent::SymbolNavigation,
            Intent::ExplainSymbol { .. } => intent::Intent::ExplainSymbol,
            _ => intent::Intent::Other,
        };

        Self {
            task_type,
            raw_input: result.raw_input.clone(),
            intent: old_intent,
            entities,
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
    Think {
        text: String,
    },
    Intelligence {
        style: render::RenderStyle,
        query: String,
    },
    Verify {
        cmd: String,
    },
    Echo {
        text: String,
    },
}

impl PlanStep {
    fn label(&self) -> String {
        match self {
            PlanStep::ToolCall { call } => format!("tool:{}", call.name),
            PlanStep::Think { .. } => "think".to_owned(),
            PlanStep::Intelligence { style, .. } => match style {
                render::RenderStyle::Navigation => "intelligence:navigation".to_owned(),
                render::RenderStyle::Explain => "intelligence:explain".to_owned(),
            },
            PlanStep::Verify { .. } => "verify".to_owned(),
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

    // Task - using intent classifier
    let class_ctx = ClassificationContext::with_cwd(
        ctx.cwd.clone().unwrap_or_else(|| PathBuf::from("."))
    );
    let task = TaskAnalyzer::analyze(user_input, ctx.intent_classifier.as_ref(), &class_ctx);
    events.emit(&RuntimeEvent::TparTaskClassified {
        task: task.name().to_owned(),
    });

    // Collect context chunks from task entities
    let context_chunks = collect_context_from_task(&task, ctx.cwd.as_deref());

    // Plan with context
    let plan = ctx.planner.plan(
        &task,
        &PlannerContext {
            budget: ctx.budget.clone(),
            context_chunks,
            repo_root: ctx.cwd.clone(),
            memory: ctx.memory.clone(),
        },
        events,
    )?;
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
        ctx.memory.clone(),
        ctx.memory_store.clone(),
    );
    let (out, review) = exec.execute(&plan, &task, events)?;

    // Review/Reflect
    let ok = matches!(review, ReviewResult::Success);
    events.emit(&RuntimeEvent::TparReviewed { ok });

    Ok(out)
}

struct TaskAnalyzer;

impl TaskAnalyzer {
    /// Analyze user input using the provided intent classifier
    fn analyze(
        input: &str,
        classifier: &dyn IntentClassifier,
        ctx: &ClassificationContext,
    ) -> Task {
        let raw = input.trim().to_owned();

        // Priority 1: Explicit edit patterns (most specific)
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

        // Priority 2: Minimal edit patterns
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

        // Priority 3: Terminal command patterns
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

        // Priority 4: Use intent classifier for natural language input
        let classification = classifier.classify(input, ctx);
        Task::from_classification(&classification)
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
    // Project memory for learning commands
    memory: Option<memory::ProjectMemory>,
    // Memory store for persistence
    memory_store: Option<memory::MemoryStore>,
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
        memory: Option<memory::ProjectMemory>,
        memory_store: Option<memory::MemoryStore>,
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
            memory,
            memory_store,
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

        let mut step_outputs: Vec<(usize, String, String)> = Vec::new(); // (step_id, step_label, output)

        for (i, step) in plan.steps.iter().enumerate() {
            let step_id = i + 1;
            let step_label = step.label();
            events.emit(&RuntimeEvent::TparStepStarted {
                step_id,
                step: step_label.clone(),
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
                    "step": step_label,
                }),
                action: serde_json::to_value(step).unwrap_or(Value::Null),
                reward: if ok { 0.2 } else { -0.5 },
                outcome: serde_json::json!({ "ok": ok, "output_len": out_text.len() }),
            });

            // Collect non-empty outputs with truncation
            let trimmed = out_text.trim();
            if !trimmed.is_empty() {
                let display = if trimmed.len() > 500 {
                    format!("{}... [truncated, {} more chars]", &trimmed[..500], trimmed.len() - 500)
                } else {
                    trimmed.to_string()
                };
                step_outputs.push((step_id, step_label, display));
            }

            if let Some(review) = review {
                if matches!(review, ReviewResult::NeedsRollback { .. }) {
                    let _ = self.rollback(&tool_ctx);
                }
                // Build formatted output on error
                let mut final_output = String::new();
                final_output.push_str(&format!("❌ Step {} failed\n", step_id));
                if let ReviewResult::NeedsRevision { reason } = &review {
                    final_output.push_str(&format!("Reason: {}\n", reason));
                }
                return Ok((final_output, review));
            }
        }

        // Build formatted final output
        let final_output = if step_outputs.is_empty() {
            format!("✅ Completed: {}", task.raw_input)
        } else {
            let mut out = String::new();
            out.push_str(&format!("📋 Plan executed ({} steps)\n", plan.steps.len()));
            out.push_str("═".repeat(40).as_str());
            out.push('\n');
            for (step_id, label, output) in step_outputs {
                out.push_str(&format!("\n[Step {}] {}\n", step_id, label));
                out.push_str("─".repeat(30).as_str());
                out.push('\n');
                out.push_str(&output);
                out.push('\n');
            }
            out.push('\n');
            out.push_str("═".repeat(40).as_str());
            out.push('\n');
            out.push_str("✅ Done\n");
            out
        };

        // Save memory before returning
        self.save_memory();

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
            PlanStep::Think { text } => Ok(StepOutcome {
                ok: true,
                output: format!("🤔 {}", text),
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
            PlanStep::Verify { cmd } => {
                let call = tools::ToolCall {
                    name: "run_terminal".to_owned(),
                    args: serde_json::json!({"cmd":cmd}),
                };
                self.check_step_safety(task, &call)?;
                let res = self.tools.run(tool_ctx, &call)?;
                // Record to memory (Verify is likely a test command)
                self.record_terminal_result(cmd, res.ok);
                if res.ok {
                    Ok(StepOutcome {
                        ok: true,
                        output: res.stdout,
                    })
                } else {
                    Err(LunaError::invalid_input(res.stderr))
                }
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
                // Record terminal commands to memory
                if call.name == "run_terminal" {
                    if let Some(cmd) = call.args.get("cmd").and_then(|v| v.as_str()) {
                        self.record_terminal_result(cmd, res.ok);
                    }
                }
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

    /// Record a terminal command result to memory
    fn record_terminal_result(&mut self, cmd: &str, success: bool) {
        use memory::CommandType;

        // Infer command type from the command string
        let cmd_type = self.infer_command_type(cmd);

        if let Some(ref mut memory) = self.memory {
            memory.record_command(cmd_type, cmd, success);
        }
    }

    /// Infer command type from command string
    fn infer_command_type(&self, cmd: &str) -> memory::CommandType {
        use memory::CommandType;

        let lower = cmd.to_lowercase();

        // Build patterns
        if lower.contains("build") || lower.contains("cargo build") || lower.contains("make") || lower.contains("npm run build") {
            return CommandType::Build;
        }

        // Test patterns
        if lower.contains("test") || lower.contains("cargo test") || lower.contains("npm test") || lower.contains("pytest") {
            return CommandType::Test;
        }

        // Check/Lint patterns
        if lower.contains("clippy") || lower.contains("check") || lower.contains("lint") || lower.contains("fmt") {
            return CommandType::Check;
        }

        // Run patterns
        if lower.contains("run") || lower.contains("start") || lower.contains("serve") {
            return CommandType::Run;
        }

        // Install patterns
        if lower.contains("install") || lower.contains("cargo add") || lower.contains("npm install") {
            return CommandType::Install;
        }

        // Clean patterns
        if lower.contains("clean") || lower.contains("cargo clean") {
            return CommandType::Clean;
        }

        // Default to Custom
        CommandType::Custom
    }

    /// Save memory to disk if modified
    fn save_memory(&self) {
        if let (Some(ref memory), Some(ref store)) = (&self.memory, &self.memory_store) {
            if let Err(e) = store.save(memory) {
                tracing::warn!("Failed to save project memory: {}", e);
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

pub(crate) fn parse_terminal_intent(input: &str) -> Option<String> {
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

pub(crate) fn parse_edit_intent_min(input: &str) -> Option<(String, usize)> {
    // "修改 <path> 第 <line> 行" 或 "修改一下 <path> 第 <line> 行"
    let s = input.trim();
    let rest = s
        .strip_prefix("修改 ")
        .or_else(|| s.strip_prefix("修改一下"))?
        .trim_start();
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

pub(crate) fn parse_edit_line_to(input: &str) -> Option<(String, usize, String)> {
    // Supported:
    // - "修改 <path> 第 <line> 行 为 <new_line>"
    // - "修改 <path> 第 <line> 行 成 <new_line>"
    // - "修改一下 <path> 第 <line> 行 为 <new_line>"
    let s = input.trim();
    let rest = s
        .strip_prefix("修改 ")
        .or_else(|| s.strip_prefix("修改一下"))?
        .trim_start();
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

// repo root resolution is shared with runtime router.
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Collect context chunks from task entities
///
/// This is a simplified implementation that extracts file paths and symbols
/// from the task entities. A full implementation would use RefillPipeline
/// for comprehensive context retrieval.
fn collect_context_from_task(
    task: &Task,
    cwd: Option<&Path>,
) -> Vec<context::ContextChunk> {
    use context::{ContextChunk, ContextType, SourceLocation, TextRange};
    use std::path::PathBuf;

    let mut chunks = Vec::new();

    // Extract paths from entities and try to read them
    for entity in &task.entities {
        if entity.kind == CodeEntityKind::Path {
            let path = PathBuf::from(&entity.value);
            let abs_path = cwd.map(|c| c.join(&path)).unwrap_or_else(|| path.clone());

            // Try to read file if it exists
            if let Ok(content) = std::fs::read_to_string(&abs_path) {
                let lines: Vec<&str> = content.lines().collect();
                let line_count = lines.len().min(50); // Limit to 50 lines

                let summary = if lines.len() > 50 {
                    format!(
                        "// File: {} ({} lines total, showing first 50)\n{}",
                        entity.value,
                        lines.len(),
                        lines[..50].join("\n")
                    )
                } else {
                    format!(
                        "// File: {} ({} lines)\n{}",
                        entity.value,
                        lines.len(),
                        content
                    )
                };

                let source = SourceLocation {
                    repo_root: cwd.unwrap_or(Path::new(".")).to_path_buf(),
                    rel_path: path,
                    range: TextRange::new(1, line_count),
                };

                chunks.push(ContextChunk::new(summary, source, ContextType::FileOverview));
            }
        }
    }

    // If no files found but task mentions symbols, add placeholder context
    if chunks.is_empty() && !task.entities.is_empty() {
        let symbols: Vec<String> = task
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Identifier)
            .map(|e| e.value.clone())
            .collect();

        if !symbols.is_empty() {
            let context_text = format!(
                "// Task mentions symbols: {}\n// Working directory: {}",
                symbols.join(", "),
                cwd.map(|p| p.display().to_string())
                    .unwrap_or_else(|| ".".to_string())
            );

            let source = SourceLocation {
                repo_root: cwd.unwrap_or(Path::new(".")).to_path_buf(),
                rel_path: PathBuf::from("task_context"),
                range: TextRange::new(1, 1),
            };

            chunks.push(ContextChunk::new(context_text, source, ContextType::CodeSnippet));
        }
    }

    chunks
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
                planner: Arc::new(crate::planner::RuleBasedPlanner::new()),
                intent_classifier: Arc::new(crate::intent_classifier::RuleBasedClassifier::new()),
                memory: None,
                memory_store: None,
            },
            &mut events,
        )
        .unwrap();

        assert!(out.contains("edited:"), "out={out}");
        let updated = std::fs::read_to_string(&file).unwrap();
        assert_eq!(updated, "hello\nWORLD\n");
    }

    #[test]
    fn test_parse_edit_intent_with_yixia() {
        // Test "修改一下" format
        let result = parse_edit_intent_min("修改一下 main.rs 第 10 行");
        assert!(result.is_some(), "Should parse '修改一下' format");
        let (path, line) = result.unwrap();
        assert_eq!(path, "main.rs");
        assert_eq!(line, 10);

        // Test without "一下"
        let result2 = parse_edit_intent_min("修改 main.rs 第 10 行");
        assert!(result2.is_some());
        let (path2, line2) = result2.unwrap();
        assert_eq!(path2, "main.rs");
        assert_eq!(line2, 10);
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
                planner: Arc::new(crate::planner::RuleBasedPlanner::new()),
                intent_classifier: Arc::new(crate::intent_classifier::RuleBasedClassifier::new()),
                memory: None,
                memory_store: None,
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
