//! Agent Loop Framework
//!
//! Generic framework for iterative agent workflows with convergence checking.
//!
//! # Usage
//!
//! ```rust,ignore
//! impl AgentLoop for FixLoop {
//!     type CheckResult = Vec<ErrorRecord>;
//!     type Action = FixAction;
//!
//!     fn check(&mut self) -> Result<Self::CheckResult> { ... }
//!     fn is_converged(&self, result: &Self::CheckResult) -> bool { ... }
//!     fn plan_actions(&self, result: &Self::CheckResult) -> Vec<Self::Action> { ... }
//!     fn execute(&mut self, action: Self::Action) -> Result<()> { ... }
//! }
//!
//! let result = LoopRunner::new(FixLoop::new(config))
//!     .max_iterations(5)
//!     .run()?;
//! ```

use anyhow::{Context, Result};
use std::time::Instant;

/// Core trait for agent loops with convergence checking
///
/// Implementors define:
/// - What "checking" means (e.g., run cargo build, run tests)
/// - When the loop has converged (e.g., no errors, tests pass)
/// - How to plan actions based on check results
/// - How to execute individual actions
pub trait AgentLoop {
    /// Type returned by check() - represents current state
    type CheckResult;

    /// Type of individual actions
    type Action;

    /// Name of this loop (for logging/debugging)
    fn name(&self) -> &str {
        "AgentLoop"
    }

    /// Check current state
    ///
    /// Called at the beginning of each iteration to assess progress.
    fn check(&mut self) -> Result<Self::CheckResult>;

    /// Determine if the loop has converged
    ///
    /// Returns true if no more iterations are needed.
    fn is_converged(&self, result: &Self::CheckResult) -> bool;

    /// Check if result represents an unfixable state
    ///
    /// Default implementation always returns false.
    fn is_unfixable(&self, _result: &Self::CheckResult) -> bool {
        false
    }

    /// Plan actions based on check result
    ///
    /// Returns a list of actions to execute this iteration.
    fn plan_actions(&self, result: &Self::CheckResult) -> Vec<Self::Action>;

    /// Execute a single action
    ///
    /// Returns Ok(()) on success, Err on failure.
    fn execute(&mut self, action: Self::Action) -> Result<()>;

    /// Called before each iteration
    ///
    /// Default implementation does nothing.
    fn on_iteration_start(&mut self, _iteration: usize) {}

    /// Called after each iteration
    ///
    /// Default implementation does nothing.
    fn on_iteration_end(&mut self, _iteration: usize, _result: &Self::CheckResult) {}

    /// Called when loop converges successfully
    ///
    /// Default implementation does nothing.
    fn on_converged(&mut self, _result: &Self::CheckResult) {}

    /// Called when loop exits without converging
    ///
    /// Default implementation does nothing.
    fn on_not_converged(&mut self, _result: &Self::CheckResult, _reason: NotConvergedReason) {}
}

/// Reason why loop didn't converge
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotConvergedReason {
    /// Maximum iterations reached
    MaxIterations,
    /// Unfixable state encountered
    Unfixable,
    /// No progress made
    NoProgress,
    /// Execution error
    ExecutionError,
}

impl std::fmt::Display for NotConvergedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotConvergedReason::MaxIterations => write!(f, "max iterations reached"),
            NotConvergedReason::Unfixable => write!(f, "unfixable state encountered"),
            NotConvergedReason::NoProgress => write!(f, "no progress made"),
            NotConvergedReason::ExecutionError => write!(f, "execution error"),
        }
    }
}

/// Configuration for loop runner
#[derive(Debug, Clone, Copy)]
pub struct LoopRunnerConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Whether to stop on first execution error
    pub stop_on_error: bool,
    /// Whether to track and check for progress
    pub check_progress: bool,
}

impl Default for LoopRunnerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            stop_on_error: false,
            check_progress: true,
        }
    }
}

/// Generic loop runner
///
/// Drives an AgentLoop implementation until convergence or termination.
pub struct LoopRunner<'a, L: AgentLoop> {
    loop_impl: &'a mut L,
    config: LoopRunnerConfig,
}

impl<'a, L: AgentLoop> LoopRunner<'a, L> {
    /// Create a new runner with default config
    pub fn new(loop_impl: &'a mut L) -> Self {
        Self {
            loop_impl,
            config: LoopRunnerConfig::default(),
        }
    }

    /// Set max iterations
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set stop on error
    pub fn stop_on_error(mut self, stop: bool) -> Self {
        self.config.stop_on_error = stop;
        self
    }

    /// Set progress checking
    pub fn check_progress(mut self, check: bool) -> Self {
        self.config.check_progress = check;
        self
    }

    /// Run the loop until convergence
    ///
    /// Returns the final check result if converged, otherwise returns error.
    pub fn run(&mut self) -> Result<L::CheckResult> {
        let _start_time = Instant::now();
        let mut previous_results: Vec<L::CheckResult> = Vec::new();

        for iteration in 0..self.config.max_iterations {
            self.loop_impl.on_iteration_start(iteration);

            // Check current state
            let result = self
                .loop_impl
                .check()
                .with_context(|| format!("{}: check failed at iteration {}", self.loop_impl.name(), iteration))?;

            // Check convergence
            if self.loop_impl.is_converged(&result) {
                self.loop_impl.on_converged(&result);
                return Ok(result);
            }

            // Check unfixable
            if self.loop_impl.is_unfixable(&result) {
                self.loop_impl
                    .on_not_converged(&result, NotConvergedReason::Unfixable);
                return Err(anyhow::anyhow!(
                    "{}: encountered unfixable state at iteration {}",
                    self.loop_impl.name(),
                    iteration
                ));
            }

            // Check progress
            if self.config.check_progress && iteration > 0 {
                let no_progress = previous_results
                    .last()
                    .map(|prev| self.is_same_state(prev, &result))
                    .unwrap_or(false);

                if no_progress && iteration >= 2 {
                    self.loop_impl
                        .on_not_converged(&result, NotConvergedReason::NoProgress);
                    return Err(anyhow::anyhow!(
                        "{}: no progress made at iteration {}",
                        self.loop_impl.name(),
                        iteration
                    ));
                }
            }

            // Plan and execute actions
            let actions = self.loop_impl.plan_actions(&result);

            if actions.is_empty() {
                // No actions but not converged - likely stuck
                self.loop_impl
                    .on_not_converged(&result, NotConvergedReason::NoProgress);
                return Err(anyhow::anyhow!(
                    "{}: no actions planned at iteration {} but not converged",
                    self.loop_impl.name(),
                    iteration
                ));
            }

            for action in actions {
                if let Err(e) = self.loop_impl.execute(action) {
                    if self.config.stop_on_error {
                        self.loop_impl
                            .on_not_converged(&result, NotConvergedReason::ExecutionError);
                        return Err(e.context(format!(
                            "{}: action execution failed at iteration {}",
                            self.loop_impl.name(),
                            iteration
                        )));
                    }
                    // Continue on error if not stop_on_error
                }
            }

            self.loop_impl.on_iteration_end(iteration, &result);
            previous_results.push(result);
        }

        // Max iterations reached
        let final_result = self.loop_impl.check()?;
        self.loop_impl
            .on_not_converged(&final_result, NotConvergedReason::MaxIterations);

        Err(anyhow::anyhow!(
            "{}: max iterations ({}) reached without convergence",
            self.loop_impl.name(),
            self.config.max_iterations
        ))
    }

    /// Check if two results represent the same state (for progress detection)
    ///
    /// Default implementation always returns false. Override for custom comparison.
    fn is_same_state(&self, _a: &L::CheckResult, _b: &L::CheckResult) -> bool {
        false
    }
}

/// Trait for results that can be checked for progress
///
/// Implement this to enable custom progress checking.
pub trait ProgressCheck {
    /// Check if this result represents progress compared to previous
    fn is_progress(&self, previous: &Self) -> Option<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation: Counter that increments until threshold
    struct CounterLoop {
        current: i32,
        threshold: i32,
    }

    impl AgentLoop for CounterLoop {
        type CheckResult = i32;
        type Action = i32;

        fn name(&self) -> &str {
            "CounterLoop"
        }

        fn check(&mut self) -> Result<Self::CheckResult> {
            Ok(self.current)
        }

        fn is_converged(&self, result: &Self::CheckResult) -> bool {
            *result >= self.threshold
        }

        fn plan_actions(&self, _result: &Self::CheckResult) -> Vec<Self::Action> {
            vec![1] // Increment by 1
        }

        fn execute(&mut self, action: Self::Action) -> Result<()> {
            self.current += action;
            Ok(())
        }
    }

    #[test]
    fn test_loop_runner_converges() {
        let mut loop_impl = CounterLoop {
            current: 0,
            threshold: 3,
        };

        let mut runner = LoopRunner::new(&mut loop_impl).max_iterations(10);
        let result = runner.run().unwrap();

        assert_eq!(result, 3); // Should converge at threshold
    }

    #[test]
    fn test_loop_runner_max_iterations() {
        let mut loop_impl = CounterLoop {
            current: 0,
            threshold: 100,
        };

        let mut runner = LoopRunner::new(&mut loop_impl).max_iterations(5);
        let err = runner.run().unwrap_err();

        assert!(err.to_string().contains("max iterations"));
    }

    // Test with unfixable state
    struct UnfixableLoop {
        iteration: usize,
    }

    impl AgentLoop for UnfixableLoop {
        type CheckResult = bool; // true = good, false = bad/unfixable
        type Action = ();

        fn check(&mut self) -> Result<Self::CheckResult> {
            self.iteration += 1;
            Ok(false) // Always bad
        }

        fn is_converged(&self, result: &Self::CheckResult) -> bool {
            *result
        }

        fn is_unfixable(&self, _result: &Self::CheckResult) -> bool {
            self.iteration >= 2
        }

        fn plan_actions(&self, _result: &Self::CheckResult) -> Vec<Self::Action> {
            vec![()]
        }

        fn execute(&mut self, _action: Self::Action) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_loop_runner_unfixable() {
        let mut loop_impl = UnfixableLoop { iteration: 0 };

        let mut runner = LoopRunner::new(&mut loop_impl).max_iterations(10);
        let err = runner.run().unwrap_err();

        assert!(err.to_string().contains("unfixable"));
    }
}
