//! FixLoop - Automated compile-error repair loop
//!
//! This module implements the M3 "Junior Coder" capability:
//! Detect compile errors → Analyze → Fix → Verify, in a loop.
//!
//! Architecture:
//! - Implements the `AgentLoop` trait for standardized loop behavior
//! - Uses ErrorParserRegistry to understand build output
//! - Delegates reasoning to ReactAgent for individual fixes
//!
//! Note: This implementation uses the loop_framework to handle generic
//! iteration logic, focusing only on FixLoop-specific behavior.

use super::loop_framework::{AgentLoop, LoopRunner};
use super::{ReactAgent, ReactOptions};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer;

use llm::LLMConfig;
use tools::{
    edit_file, run_terminal, EditOp, ErrorParserRegistry, ErrorRecord, ErrorSummary, Location,
    TerminalResult,
};

/// Configuration for the fix loop
#[derive(Debug, Clone)]
pub struct FixLoopConfig {
    /// Maximum number of fix iterations
    pub max_iterations: usize,
    /// Build command to run (e.g., "cargo build")
    pub build_command: String,
    /// Working directory for the build command
    pub working_dir: std::path::PathBuf,
    /// Whether to allow edits (safety switch)
    pub allow_edits: bool,
    /// React options for the inner agent
    pub react_options: ReactOptions,
    /// Repository root for context operations
    pub repo_root: std::path::PathBuf,
    /// Tokenizer for context operations
    pub tokenizer: Tokenizer,
    /// LLM configuration
    pub llm_config: LLMConfig,
}

impl Default for FixLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            build_command: "cargo build".to_string(),
            working_dir: std::env::current_dir().unwrap_or_else(|_| Path::new(".").into()),
            allow_edits: true,
            react_options: ReactOptions::default(),
            repo_root: std::env::current_dir().unwrap_or_else(|_| Path::new(".").into()),
            tokenizer: super::super::util::demo_tokenizer(),
            llm_config: LLMConfig::default(),
        }
    }
}

/// A single fix action
#[derive(Debug, Clone)]
pub enum FixAction {
    /// Analyze an error and suggest fixes
    Analyze { error: ErrorRecord },
    /// Apply an edit to fix an error
    Edit {
        path: String,
        prompt: String,
        error_message: String,
    },
    /// Skip (unfixable or already fixed)
    Skip { reason: String },
}

/// Complete fix loop result
#[derive(Debug, Clone)]
pub struct FixLoopResult {
    /// Final error count (0 if successful)
    pub final_error_count: usize,
    /// Files modified
    pub modified_files: HashSet<String>,
    /// All iterations performed
    pub iterations: Vec<FixIteration>,
    /// Whether converged successfully
    pub converged: bool,
}

/// Result of a single fix iteration
#[derive(Debug, Clone)]
pub struct FixIteration {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Parsed errors at start of iteration
    pub errors: Vec<ErrorRecord>,
    /// Error summary
    pub summary: ErrorSummary,
    /// Actions taken this iteration
    pub actions: Vec<FixActionResult>,
}

/// Result of executing a fix action
#[derive(Debug, Clone)]
pub enum FixActionResult {
    /// Analyzed an error
    Analyzed { location: Location, message: String },
    /// Applied an edit
    Edited { path: String, lines: (usize, usize), success: bool },
    /// Skipped
    Skipped { reason: String },
}

/// Reason for convergence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergedReason {
    /// All errors fixed successfully
    Success,
    /// Reached maximum iterations
    MaxIterations,
    /// Encountered unfixable errors (infra/dependency)
    UnfixableErrors,
    /// No progress made in recent iterations
    NoProgress,
    /// Build passed (no errors)
    BuildPassed,
}

impl std::fmt::Display for ConvergedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvergedReason::Success => write!(f, "all errors fixed"),
            ConvergedReason::MaxIterations => write!(f, "max iterations reached"),
            ConvergedReason::UnfixableErrors => write!(f, "unfixable errors encountered"),
            ConvergedReason::NoProgress => write!(f, "no progress made"),
            ConvergedReason::BuildPassed => write!(f, "build passed"),
        }
    }
}

/// The FixLoop state machine implementing AgentLoop
pub struct FixLoop {
    config: FixLoopConfig,
    error_parser: ErrorParserRegistry,
    modified_files: HashSet<String>,
    iterations: Vec<FixIteration>,
}

impl FixLoop {
    /// Create a new FixLoop with the given configuration
    pub fn new(config: FixLoopConfig) -> Self {
        let mut error_parser = ErrorParserRegistry::new();

        // Register cargo parser for common Rust commands
        error_parser.register("cargo build", tools::CargoErrorParser::new());
        error_parser.register("cargo test", tools::CargoErrorParser::new());
        error_parser.register("cargo check", tools::CargoErrorParser::new());

        Self {
            config,
            error_parser,
            modified_files: HashSet::new(),
            iterations: Vec::new(),
        }
    }

    /// Run the complete fix loop using LoopRunner
    pub fn run(&mut self) -> Result<FixLoopResult> {
        let max_iterations = self.config.max_iterations;
        let mut runner = LoopRunner::new(self).max_iterations(max_iterations);

        match runner.run() {
            Ok(errors) => {
                let error_count = errors.len();
                Ok(FixLoopResult {
                    final_error_count: error_count,
                    modified_files: self.modified_files.clone(),
                    iterations: self.iterations.clone(),
                    converged: error_count == 0,
                })
            }
            Err(_e) => {
                // Check if we have a final result despite error
                let final_errors = self.check().unwrap_or_default();
                let error_count = final_errors.len();

                Ok(FixLoopResult {
                    final_error_count: error_count,
                    modified_files: self.modified_files.clone(),
                    iterations: self.iterations.clone(),
                    converged: false,
                })
            }
        }
    }

    /// Get modified files
    pub fn modified_files(&self) -> &HashSet<String> {
        &self.modified_files
    }

    /// Get iterations
    pub fn iterations(&self) -> &[FixIteration] {
        &self.iterations
    }

    /// Run the build command
    fn run_build(&self) -> Result<TerminalResult> {
        run_terminal(
            &self.config.build_command,
            Some(&self.config.working_dir),
            false,
        )
        .with_context(|| format!("Failed to run build command: {}", self.config.build_command))
    }

    /// Parse errors from build output
    fn parse_errors(&self, build_result: &TerminalResult) -> Vec<ErrorRecord> {
        let output = if build_result.stderr.is_empty() {
            &build_result.stdout
        } else {
            &build_result.stderr
        };

        self.error_parser
            .parse(&self.config.build_command, output, build_result.exit_code)
    }

    /// Extract code block from markdown-formatted response
    /// Extracts the LAST code block (in case LLM returns both original and fixed)
    fn extract_code_block(answer: &str) -> String {
        // Look for ```rust or ``` code blocks - find the LAST one
        let patterns = ["```rust\n", "```\n"];
        let mut last_match: Option<(usize, usize)> = None; // (start_of_content, end_of_content)

        for pattern in &patterns {
            let mut search_start = 0;
            while let Some(start) = answer[search_start..].find(pattern) {
                let absolute_start = search_start + start;
                let after_start = absolute_start + pattern.len();
                if let Some(end) = answer[after_start..].find("```") {
                    let absolute_end = after_start + end;
                    last_match = Some((after_start, absolute_end));
                    search_start = absolute_end + 3; // Skip past this code block
                } else {
                    break;
                }
            }
        }

        if let Some((start, end)) = last_match {
            return answer[start..end].trim().to_string();
        }

        // Fallback: return the whole answer if no code block found
        answer.trim().to_string()
    }

    /// Execute a fix action
    fn execute_action(&mut self, action: FixAction) -> Result<FixActionResult> {
        match action {
            FixAction::Analyze { error } => {
                let location = error.locations.first().cloned().unwrap_or_else(|| {
                    Location::new("unknown", 0)
                });

                // Build analysis prompt
                let analysis_prompt = format!(
                    "Analyze this compilation error and explain what needs to be fixed:\n\n\
                     Error: {}\n\
                     Location: {}\n\
                     Message: {}\n\n\
                     Explain the root cause and suggest the fix.",
                    error.error_code.as_deref().unwrap_or("ERROR"),
                    location.raw,
                    error.message
                );

                let agent = ReactAgent::new(
                    self.config.llm_config.clone(),
                    self.config.react_options.clone(),
                );

                match agent.ask(
                    &self.config.repo_root,
                    &analysis_prompt,
                    &self.config.tokenizer,
                ) {
                    Ok((answer, _, _)) => Ok(FixActionResult::Analyzed {
                        location,
                        message: answer,
                    }),
                    Err(e) => Ok(FixActionResult::Skipped {
                        reason: format!("Analysis failed: {}", e),
                    }),
                }
            }

            FixAction::Edit {
                path,
                prompt,
                error_message: _,
            } => {
                if !self.config.allow_edits {
                    return Ok(FixActionResult::Skipped {
                        reason: "Edits disabled (dry run)".to_string(),
                    });
                }

                // Step 1: Build file path and read content
                let file_path = self.config.repo_root.join(&path);

                // Read the file content to include in prompt
                let file_content = std::fs::read_to_string(&file_path).unwrap_or_default();

                // Step 2: Use LLM to generate the fix
                let agent = ReactAgent::new(
                    self.config.llm_config.clone(),
                    self.config.react_options.clone(),
                );

                let fix_prompt = format!(
                    "Current file content ({}):
```rust
{}
```

Errors to fix:
{}

Please provide the complete fixed code for this file. \
Respond with ONLY the fixed code block in markdown format:
```rust
// fixed code here
```",
                    path, file_content, prompt
                );

                match agent.ask(&self.config.repo_root, &fix_prompt, &self.config.tokenizer) {
                    Ok((answer, _, _)) => {
                        // Extract code block from answer
                        let fixed_code = Self::extract_code_block(&answer);

                        if fixed_code.is_empty() {
                            return Ok(FixActionResult::Analyzed {
                                location: Location::new(&path, 0),
                                message: format!("LLM did not provide fix. Answer: {}", answer),
                            });
                        }

                        // Apply the fix directly using edit_file
                        // file_path already defined above
                        match std::fs::read_to_string(&file_path) {
                            Ok(content) => {
                                let lines: Vec<&str> = content.lines().collect();
                                let end_line = lines.len();

                                // Replace entire file content
                                let op = EditOp::ReplaceLines {
                                    start_line: 1,
                                    end_line,
                                    new_content: fixed_code,
                                };

                                match edit_file(&file_path, &op, true) {
                                    Ok(result) => {
                                        if result.success {
                                            self.modified_files.insert(path.clone());
                                            Ok(FixActionResult::Edited {
                                                path,
                                                lines: (1, end_line),
                                                success: true,
                                            })
                                        } else {
                                            Ok(FixActionResult::Skipped {
                                                reason: format!("Edit failed: {:?}", result.error),
                                            })
                                        }
                                    }
                                    Err(e) => Ok(FixActionResult::Skipped {
                                        reason: format!("Edit error: {}", e),
                                    }),
                                }
                            }
                            Err(e) => Ok(FixActionResult::Skipped {
                                reason: format!("Failed to read file: {}", e),
                            }),
                        }
                    }
                    Err(e) => Ok(FixActionResult::Skipped {
                        reason: format!("LLM failed: {}", e),
                    }),
                }
            }

            FixAction::Skip { reason } => Ok(FixActionResult::Skipped { reason }),
        }
    }
}

impl AgentLoop for FixLoop {
    type CheckResult = Vec<ErrorRecord>;
    type Action = FixAction;

    fn name(&self) -> &str {
        "FixLoop"
    }

    fn check(&mut self) -> Result<Self::CheckResult> {
        let build_result = self.run_build()?;
        let errors = self.parse_errors(&build_result);
        Ok(errors)
    }

    fn is_converged(&self, result: &Self::CheckResult) -> bool {
        result.is_empty()
    }

    fn is_unfixable(&self, result: &Self::CheckResult) -> bool {
        let summary = ErrorSummary::from_records(result);
        summary.has_unfixable
    }

    fn plan_actions(&self, result: &Self::CheckResult) -> Vec<Self::Action> {
        let mut actions = Vec::new();

        // Group errors by file
        let mut errors_by_file: std::collections::HashMap<String, Vec<&ErrorRecord>> =
            std::collections::HashMap::new();

        for error in result {
            for loc in &error.locations {
                errors_by_file.entry(loc.path.clone()).or_default().push(error);
            }
        }

        // Create actions for each file
        for (path, file_errors) in errors_by_file {
            let fixable_errors: Vec<_> = file_errors
                .iter()
                .filter(|e| e.kind.is_fixable())
                .collect();

            if fixable_errors.is_empty() {
                actions.push(FixAction::Skip {
                    reason: format!("No fixable errors in {}", path),
                });
                continue;
            }

            // Build fix prompt for this file
            let error_list = fixable_errors
                .iter()
                .map(|e| {
                    let loc = e
                        .locations
                        .first()
                        .map(|l| format!("{}:{}", l.path, l.line))
                        .unwrap_or_else(|| path.clone());
                    format!(
                        "- [{}] {} at {}\n  {}",
                        e.error_code.as_deref().unwrap_or("ERROR"),
                        e.message,
                        loc,
                        e.suggestion.as_deref().unwrap_or("")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Fix the following compilation errors in {}:\n\n{}\n\n\
                 Use the edit_file tool to make the necessary changes.",
                path, error_list
            );

            // Create action for the first error (simplified)
            if let Some(first_error) = fixable_errors.first() {
                actions.push(FixAction::Edit {
                    path: path.clone(),
                    prompt,
                    error_message: first_error.message.clone(),
                });
            }
        }

        actions
    }

    fn execute(&mut self, action: Self::Action) -> Result<()> {
        let result = self.execute_action(action)?;

        // Store result in current iteration
        if let Some(last) = self.iterations.last_mut() {
            last.actions.push(result);
        }

        Ok(())
    }

    fn on_iteration_start(&mut self, iteration: usize) {
        self.iterations.push(FixIteration {
            iteration,
            errors: Vec::new(),
            summary: ErrorSummary::default(),
            actions: Vec::new(),
        });
    }

    fn on_iteration_end(&mut self, _iteration: usize, result: &Self::CheckResult) {
        if let Some(last) = self.iterations.last_mut() {
            last.errors = result.clone();
            last.summary = ErrorSummary::from_records(result);
        }
    }
}

/// Convenience function to run a fix loop with default configuration
pub fn run_fix_loop(
    repo_root: &Path,
    build_command: &str,
    tokenizer: &Tokenizer,
    llm_config: &LLMConfig,
) -> Result<FixLoopResult> {
    let config = FixLoopConfig {
        build_command: build_command.to_string(),
        working_dir: repo_root.to_path_buf(),
        repo_root: repo_root.to_path_buf(),
        tokenizer: tokenizer.clone(),
        llm_config: llm_config.clone(),
        ..Default::default()
    };

    let mut fix_loop = FixLoop::new(config);
    fix_loop.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_loop_config_default() {
        let config = FixLoopConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.build_command, "cargo build");
        assert!(config.allow_edits);
    }

    #[test]
    fn test_converged_reason_display() {
        assert_eq!(ConvergedReason::Success.to_string(), "all errors fixed");
        assert_eq!(
            ConvergedReason::MaxIterations.to_string(),
            "max iterations reached"
        );
    }

    #[test]
    fn test_fix_loop_implements_agent_loop() {
        fn assert_implements_agent_loop<T: AgentLoop>() {}
        assert_implements_agent_loop::<FixLoop>();
    }

    #[test]
    fn test_fix_loop_name() {
        let config = FixLoopConfig::default();
        let fix_loop = FixLoop::new(config);
        assert_eq!(fix_loop.name(), "FixLoop");
    }
}
