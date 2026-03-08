//! Phase 3B: Planner abstraction (Rule-based + LLM-based + selector/fallback).

use std::sync::Arc;

use crate::config::TokenBudget;
use crate::response::RuntimeEvent;
use crate::tpar::{CodeEntityKind, Plan, PlanStep, Task, TaskType};

/// Planner-only context.
///
/// Keep it minimal to avoid tight coupling with execution.
#[derive(Clone)]
pub struct PlannerContext {
    pub budget: TokenBudget,
    /// Context chunks from Context Pipeline
    pub context_chunks: Vec<context::ContextChunk>,
    /// Repository root for path resolution
    pub repo_root: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for PlannerContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlannerContext")
            .field("budget", &self.budget)
            .field("context_chunks", &self.context_chunks.len())
            .field("repo_root", &self.repo_root)
            .finish()
    }
}

pub trait TaskPlanner: Send + Sync {
    fn kind(&self) -> &'static str;
    fn plan(
        &self,
        task: &Task,
        ctx: &PlannerContext,
        events: &mut dyn crate::response::EventSink,
    ) -> error::Result<Plan>;
}

/// Validate a plan before execution.
#[derive(Debug, Clone)]
pub struct PlanValidator {
    pub max_steps: usize,
}

impl PlanValidator {
    #[must_use]
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }

    pub fn validate(&self, plan: &Plan) -> error::Result<()> {
        if plan.steps.len() > self.max_steps {
            return Err(error::LunaError::invalid_input(format!(
                "planned steps too many: {} > {}",
                plan.steps.len(),
                self.max_steps
            )));
        }
        for step in &plan.steps {
            match step {
                PlanStep::ToolCall { call } => {
                    let ok = matches!(
                        call.name.as_str(),
                        "read_file" | "edit_file" | "run_terminal"
                    );
                    if !ok {
                        return Err(error::LunaError::invalid_input(format!(
                            "unknown tool in plan: {}",
                            call.name
                        )));
                    }
                }
                PlanStep::Verify { cmd } => {
                    if cmd.trim().is_empty() {
                        return Err(error::LunaError::invalid_input("verify step cmd is empty"));
                    }
                }
                PlanStep::Intelligence { query, .. } => {
                    if query.trim().is_empty() {
                        return Err(error::LunaError::invalid_input(
                            "intelligence step query is empty",
                        ));
                    }
                }
                PlanStep::Echo { .. } | PlanStep::Think { .. } => {}
            }
        }
        Ok(())
    }
}

/// A direct port of the existing rule-based planner.
#[derive(Debug, Default)]
pub struct RuleBasedPlanner;

impl RuleBasedPlanner {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn estimate_tokens(s: &str) -> usize {
        // Very rough heuristic: 1 token ~= 4 chars.
        s.chars().count().div_ceil(4)
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

    fn plan_inner(task: &Task) -> Plan {
        let mut steps = Vec::<PlanStep>::new();

        match task.task_type {
            TaskType::Query => {
                steps.push(PlanStep::Intelligence {
                    style: crate::render::RenderStyle::Navigation,
                    query: task.raw_input.clone(),
                });
            }
            TaskType::Explain => {
                steps.push(PlanStep::Intelligence {
                    style: crate::render::RenderStyle::Explain,
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

                if let Some((line_1, new_line)) = Self::extract_edit_payload(task) {
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

        let estimated_tokens = Self::estimate_tokens(&task.raw_input) + steps.len() * 30;
        Plan {
            steps,
            estimated_tokens,
        }
    }
}

impl TaskPlanner for RuleBasedPlanner {
    fn kind(&self) -> &'static str {
        "rule_based"
    }

    fn plan(
        &self,
        task: &Task,
        _ctx: &PlannerContext,
        _events: &mut dyn crate::response::EventSink,
    ) -> error::Result<Plan> {
        Ok(Self::plan_inner(task))
    }
}

/// LLM-based planner: request a JSON `Plan`.
///
/// MVP: strict JSON parsing + validation + retry-once + fallback.
#[derive(Clone)]
pub struct LLMBasedPlanner {
    client: Arc<dyn llm::LLMClient>,
    validator: PlanValidator,
}

impl LLMBasedPlanner {
    #[must_use]
    pub fn new(client: Arc<dyn llm::LLMClient>, max_steps: usize) -> Self {
        Self {
            client,
            validator: PlanValidator::new(max_steps),
        }
    }

    fn build_prompt(
        task: &Task,
        budget: &TokenBudget,
        repo_root: Option<&std::path::Path>,
        context_chunks: &[context::ContextChunk],
    ) -> String {
        let example = r#"{
  "steps": [
    {"kind": "think", "text": "Need to find the function"},
    {"kind": "intelligence", "style": "navigation", "query": "find fn main"},
    {"kind": "tool_call", "call": {"name": "read_file", "args": {"path": "src/main.rs"}}},
    {"kind": "echo", "text": "Found at line 10"}
  ],
  "estimated_tokens": 100
}"#;

        // Build context section
        let context_section = if context_chunks.is_empty() {
            String::from("No relevant code context found.")
        } else {
            let mut ctx = String::from("\n");
            for chunk in context_chunks {
                ctx.push_str(&chunk.format_for_prompt());
                ctx.push('\n');
            }
            ctx
        };

        // Build project root section
        let repo_section = repo_root
            .map(|p| format!("Project root: {}\n", p.display()))
            .unwrap_or_default();

        format!(
            "You are a planning engine for a code assistant.\n\
{}\
Relevant code context:\n{}\n\n\
Task type: {:?}\nUser input: {}\n\n\
Create a plan using these step kinds:\n\
- think: reasoning text (optional)\n\
- intelligence: query for code navigation/explanation (ONLY if context above does not contain the answer)\n\
- tool_call: invoke read_file, edit_file, or run_terminal\n\
- verify: run a command to verify changes\n\
- echo: final response to user (MUST provide a helpful answer based on the context, NOT just echo the user input)\n\n\
Available tools and their REQUIRED parameters:\n\
1. read_file: {{\"path\": \"file/path.rs\"}}\n\
2. edit_file: {{\"path\": \"file.rs\", \"line_1\": 10, \"new_line\": \"new content\"}} OR {{\"path\": \"file.rs\", \"start_line_1\": 10, \"end_line_1\": 15, \"replace_with\": \"new content\"}}\n\
3. run_terminal: {{\"cmd\": \"command to execute\"}} - NOTE: use 'cmd' key, NOT 'command'\n\n\
CRITICAL RULES:\n\
- The \"kind\" field MUST be exactly one of: \"think\", \"intelligence\", \"tool_call\", \"verify\", \"echo\" (NEVER use \"navigation\", \"search\", \"read\", \"edit\" or other values)\n\
- run_terminal args MUST have {{\"cmd\": \"...\"}}, not {{\"command\": \"...\"}}\n\
- All tool args must match the exact field names shown above\n\
- Use file paths from the context when available\n\
- If the context already shows the answer, use 'echo' to respond directly, NOT 'intelligence'\n\
- Return ONLY valid JSON, no markdown, no backticks\n\n\
Constraints:\n\
- Maximum {} steps\n\
- Return ONLY valid JSON\n\n\
Example output:\n{}\n",
            repo_section, context_section, task.task_type, task.raw_input, budget.max_steps, example
        )
    }

    fn build_repair_prompt(
        error: &error::LunaError,
        prev_output: &str,
        budget: &TokenBudget,
    ) -> String {
        format!(
            "The previous output was invalid JSON or schema.\n\
Error: {}\n\n\
Previous output:\n{}\n\n\
Please return ONLY valid JSON matching the Plan schema.\n\
Max steps: {}\n\
Required fields: steps[], estimated_tokens\n",
            error,
            prev_output.chars().take(500).collect::<String>(),
            budget.max_steps
        )
    }

    /// Extract JSON from markdown code blocks or return raw string
    fn extract_json(s: &str) -> String {
        let trimmed = s.trim();
        // Handle ```json ... ``` or ``` ... ```
        if let Some(start) = trimmed.find("```") {
            if let Some(end) = trimmed.rfind("```") {
                if start != end {
                    let code_block = &trimmed[start + 3..end];
                    // Remove language identifier if present (e.g., "json\n")
                    return code_block
                        .trim_start()
                        .lines()
                        .skip_while(|line| line.trim() == "json" || line.trim() == "JSON")
                        .collect::<Vec<_>>()
                        .join("\n")
                        .trim()
                        .to_string();
                }
            }
        }
        trimmed.to_string()
    }

    fn try_parse_and_validate(&self, s: &str) -> error::Result<Plan> {
        let json_str = Self::extract_json(s);

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| error::LunaError::invalid_input(format!("invalid json: {e}")))?;
        let plan: Plan = serde_json::from_value(v)
            .map_err(|e| error::LunaError::invalid_input(format!("invalid plan schema: {e}")))?;
        self.validator.validate(&plan)?;
        Ok(plan)
    }
}

impl TaskPlanner for LLMBasedPlanner {
    fn kind(&self) -> &'static str {
        "llm_based"
    }

    fn plan(
        &self,
        task: &Task,
        ctx: &PlannerContext,
        events: &mut dyn crate::response::EventSink,
    ) -> error::Result<Plan> {
        let prompt = Self::build_prompt(task, &ctx.budget, ctx.repo_root.as_deref(), &ctx.context_chunks);

        let ev = RuntimeEvent::TparPlanBuilt {
            plan: "planner=llm (deepseek) request".to_owned(),
        };
        events.emit(&ev);

        let out = self
            .client
            .complete(llm::CompletionRequest { prompt })?
            .content;

        // 1st attempt
        match self.try_parse_and_validate(&out) {
            Ok(plan) => Ok(plan),
            Err(first_err) => {
                // Retry once with a repair instruction.
                let repair_prompt = Self::build_repair_prompt(&first_err, &out, &ctx.budget);
                let out2 = self
                    .client
                    .complete(llm::CompletionRequest {
                        prompt: repair_prompt,
                    })?
                    .content;
                self.try_parse_and_validate(&out2)
            }
        }
    }
}

/// Choose a planner and provide safe fallback.
#[derive(Clone)]
pub struct PlannerSelector {
    prefer_llm: bool,
    rule: Arc<dyn TaskPlanner>,
    llm: Arc<dyn TaskPlanner>,
}

impl PlannerSelector {
    #[must_use]
    pub fn new(prefer_llm: bool, rule: Arc<dyn TaskPlanner>, llm: Arc<dyn TaskPlanner>) -> Self {
        Self {
            prefer_llm,
            rule,
            llm,
        }
    }

    fn should_use_llm(&self, task: &Task) -> bool {
        if !self.prefer_llm {
            return false;
        }
        // Use LLM for complex tasks that benefit from planning
        matches!(
            task.task_type,
            TaskType::Query | TaskType::Explain | TaskType::Edit | TaskType::Chat
        )
    }
}

impl TaskPlanner for PlannerSelector {
    fn kind(&self) -> &'static str {
        "selector"
    }

    fn plan(
        &self,
        task: &Task,
        ctx: &PlannerContext,
        events: &mut dyn crate::response::EventSink,
    ) -> error::Result<Plan> {
        let validator = PlanValidator::new(ctx.budget.max_steps);

        let try_llm = self.should_use_llm(task);
        if try_llm {
            // Try LLM first, fallback to rule-based on failure
            match self.llm.plan(task, ctx, events) {
                Ok(plan) => {
                    validator.validate(&plan)?;
                    return Ok(plan);
                }
                Err(e) => {
                    tracing::warn!("LLM planner failed, falling back to rule-based: {}", e);
                    // Fall through to rule-based
                }
            }
        }

        // Fallback to rule-based
        let plan = self.rule.plan(task, ctx, events)?;
        validator.validate(&plan)?;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_task(task_type: TaskType, raw: &str) -> Task {
        Task {
            task_type,
            raw_input: raw.to_owned(),
            intent: crate::intent::Intent::Other,
            entities: Vec::new(),
        }
    }

    #[test]
    fn selector_falls_back_when_llm_returns_invalid_json() {
        let ctx = PlannerContext {
            budget: TokenBudget {
                max_input_chars: 2048,
                max_io_bytes: 1024,
                max_steps: 8,
            },
            context_chunks: Vec::new(),
            repo_root: None,
        };

        // Provide two responses: first fails, second also fails (triggering fallback)
        let llm_client = Arc::new(llm::MockClient::new(vec![
            "not json".to_owned(),      // First attempt - invalid
            "still not json".to_owned(), // Repair attempt - also invalid, triggering fallback
        ]));
        let llm_planner = Arc::new(LLMBasedPlanner::new(llm_client, ctx.budget.max_steps))
            as Arc<dyn TaskPlanner>;
        let rule_planner = Arc::new(RuleBasedPlanner::new()) as Arc<dyn TaskPlanner>;

        let selector = PlannerSelector::new(true, rule_planner, llm_planner);

        let task = mk_task(TaskType::Query, "foo 在哪里定义");
        let mut events = Vec::<RuntimeEvent>::new();

        let plan = selector.plan(&task, &ctx, &mut events).unwrap();
        assert!(!plan.steps.is_empty());
        assert!(matches!(plan.steps[0], PlanStep::Intelligence { .. }));
    }

    #[test]
    fn test_extract_json_from_markdown() {
        // Test with json language tag
        let md = "```json\n{\"steps\": [], \"estimated_tokens\": 100}\n```";
        let result = LLMBasedPlanner::extract_json(md);
        assert_eq!(result, "{\"steps\": [], \"estimated_tokens\": 100}");

        // Test without language tag
        let md2 = "```\n{\"steps\": [], \"estimated_tokens\": 100}\n```";
        let result2 = LLMBasedPlanner::extract_json(md2);
        assert_eq!(result2, "{\"steps\": [], \"estimated_tokens\": 100}");

        // Test without markdown
        let raw = "{\"steps\": [], \"estimated_tokens\": 100}";
        let result3 = LLMBasedPlanner::extract_json(raw);
        assert_eq!(result3, "{\"steps\": [], \"estimated_tokens\": 100}");
    }

    #[test]
    fn test_parse_deepseek_real_output() {
        // Simulate actual DeepSeek output format
        let deepseek_output = r#"```json
{
  "steps": [
    {"kind": "think", "text": "Need to analyze the project"},
    {"kind": "tool_call", "call": {"name": "run_terminal", "args": {"cmd": "cargo build"}}},
    {"kind": "echo", "text": "Build completed"}
  ],
  "estimated_tokens": 150
}
```"#;

        let client = Arc::new(llm::MockClient::new(vec![deepseek_output.to_owned()]));
        let planner = LLMBasedPlanner::new(client, 12);
        let ctx = PlannerContext {
            budget: TokenBudget {
                max_input_chars: 2048,
                max_io_bytes: 1024,
                max_steps: 8,
            },
            context_chunks: Vec::new(),
            repo_root: None,
        };

        let task = mk_task(TaskType::Chat, "修复项目");
        let mut events = Vec::<RuntimeEvent>::new();

        let plan = planner.plan(&task, &ctx, &mut events).expect("Should parse DeepSeek output");
        assert_eq!(plan.steps.len(), 3);
        assert!(matches!(plan.steps[0], PlanStep::Think { .. }));
        assert!(matches!(plan.steps[1], PlanStep::ToolCall { .. }));
    }

    /// Test that verifies planner selector uses LLM for Chat tasks when prefer_llm=true
    #[test]
    #[ignore = "requires DeepSeek API key"]
    fn test_planner_selector_uses_llm_for_chat() {
        // Set up env like user did
        std::env::set_var("LUNA_LLM_API_KEY", "sk-da97464036724c46a93d2e410015642b");
        std::env::set_var("LUNA_LLM_BASE_URL", "https://api.deepseek.com/v1");
        std::env::set_var("LUNA_LLM_MODEL", "deepseek-chat");

        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        rt.block_on(async {
            // Create client like RuntimeConfig does
            let llm_client: Arc<dyn llm::LLMClient> =
                llm::OpenAIClient::try_from_env()
                    .map(|c| Arc::new(c) as Arc<dyn llm::LLMClient>)
                    .expect("Should create DeepSeek client from env");

            let llm_planner = Arc::new(LLMBasedPlanner::new(Arc::clone(&llm_client), 12))
                as Arc<dyn TaskPlanner>;
            let rule_planner = Arc::new(RuleBasedPlanner::new()) as Arc<dyn TaskPlanner>;

            // prefer_llm = true
            let selector = PlannerSelector::new(true, rule_planner, llm_planner);

            // Verify should_use_llm returns true for Chat
            let task = mk_task(TaskType::Chat, "修复项目");

            let ctx = PlannerContext {
                budget: TokenBudget {
                    max_input_chars: 2048,
                    max_io_bytes: 1024,
                    max_steps: 8,
                },
                context_chunks: Vec::new(),
                repo_root: None,
            };
            let mut events = Vec::<RuntimeEvent>::new();

            println!("Calling planner with Chat task...");
            match selector.plan(&task, &ctx, &mut events) {
                Ok(plan) => {
                    println!("Got plan with {} steps", plan.steps.len());
                    for (i, step) in plan.steps.iter().enumerate() {
                        println!("Step {}: {:?}", i, step);
                    }
                    // If LLM worked, we should have multiple steps
                    assert!(plan.steps.len() > 1, "LLM should generate multi-step plan");
                }
                Err(e) => {
                    panic!("Plan failed: {}", e);
                }
            }
        });
    }
}
