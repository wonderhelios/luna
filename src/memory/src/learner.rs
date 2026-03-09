//! Command learning system - track which commands work for each project

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of command
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandType {
    /// Build the project
    Build,
    /// Run tests
    Test,
    /// Check/lint code
    Check,
    /// Format code
    Format,
    /// Run the application
    Run,
    /// Clean build artifacts
    Clean,
    /// Install dependencies
    Install,
    /// Deploy
    Deploy,
    /// Custom command
    Custom,
}

impl CommandType {
    /// Get a description of this command type
    pub fn description(&self) -> &'static str {
        match self {
            CommandType::Build => "Build the project",
            CommandType::Test => "Run tests",
            CommandType::Check => "Check/lint code",
            CommandType::Format => "Format code",
            CommandType::Run => "Run the application",
            CommandType::Clean => "Clean build artifacts",
            CommandType::Install => "Install dependencies",
            CommandType::Deploy => "Deploy the application",
            CommandType::Custom => "Custom command",
        }
    }
}

/// A learned command with success statistics
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LearnedCommand {
    /// The command string
    pub command: String,
    /// Number of successful executions
    pub success_count: u32,
    /// Number of failed executions
    pub failure_count: u32,
    /// Last successful execution timestamp (seconds since epoch)
    pub last_success: Option<u64>,
    /// Last failed execution timestamp
    pub last_failure: Option<u64>,
    /// Average execution time (if known)
    pub avg_duration_ms: Option<u64>,
}

impl LearnedCommand {
    /// Create a new learned command
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            success_count: 0,
            failure_count: 0,
            last_success: None,
            last_failure: None,
            avg_duration_ms: None,
        }
    }

    /// Record a successful execution
    pub fn record_success(&mut self, duration_ms: Option<u64>) {
        self.success_count += 1;
        self.last_success = Some(current_timestamp());

        // Update average duration
        if let Some(duration) = duration_ms {
            self.avg_duration_ms = Some(
                self.avg_duration_ms
                    .map(|avg| (avg * (self.success_count - 1) as u64 + duration) / self.success_count as u64)
                    .unwrap_or(duration),
            );
        }
    }

    /// Record a failed execution
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(current_timestamp());
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Check if this command is reliable (>80% success rate)
    pub fn is_reliable(&self) -> bool {
        self.success_rate() >= 0.8 && self.success_count >= 1
    }

    /// Get total execution count
    pub fn total_runs(&self) -> u32 {
        self.success_count + self.failure_count
    }
}

/// Command history - tracks learned commands by type
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct CommandHistory {
    /// Commands by type
    pub commands: HashMap<CommandType, Vec<LearnedCommand>>,
}

impl CommandHistory {
    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Record a command execution result
    pub fn record(&mut self, cmd_type: CommandType, command: &str, success: bool) {
        let commands = self.commands.entry(cmd_type).or_default();

        // Find existing command or create new one
        if let Some(existing) = commands.iter_mut().find(|c| c.command == command) {
            if success {
                existing.record_success(None);
            } else {
                existing.record_failure();
            }
        } else {
            let mut new_cmd = LearnedCommand::new(command);
            if success {
                new_cmd.record_success(None);
            } else {
                new_cmd.record_failure();
            }
            commands.push(new_cmd);
        }

        // Sort by success rate (best first)
        commands.sort_by(|a, b| {
            let rate_cmp = b.success_rate().partial_cmp(&a.success_rate()).unwrap();
            if rate_cmp == std::cmp::Ordering::Equal {
                // Tie-breaker: more total runs
                b.total_runs().cmp(&a.total_runs())
            } else {
                rate_cmp
            }
        });
    }

    /// Get the best command for a specific type
    pub fn best(&self, cmd_type: CommandType) -> Option<&str> {
        self.commands
            .get(&cmd_type)
            .and_then(|cmds| cmds.iter().find(|c| c.is_reliable()))
            .map(|c| c.command.as_str())
    }

    /// Get all commands for a type (sorted by reliability)
    pub fn get_commands(&self, cmd_type: CommandType) -> &[LearnedCommand] {
        self.commands.get(&cmd_type).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if we have any successful command for this type
    pub fn has_successful(&self, cmd_type: CommandType) -> bool {
        self.commands
            .get(&cmd_type)
            .map(|cmds| cmds.iter().any(|c| c.success_count > 0))
            .unwrap_or(false)
    }

    /// Get the most recently successful command
    pub fn most_recent_success(&self, cmd_type: CommandType) -> Option<&str> {
        self.commands
            .get(&cmd_type)
            .and_then(|cmds| {
                cmds.iter()
                    .filter(|c| c.last_success.is_some())
                    .max_by_key(|c| c.last_success)
            })
            .map(|c| c.command.as_str())
    }

    /// Suggest commands to try (based on project type defaults if no learned commands)
    pub fn suggest_commands(&self, cmd_type: CommandType) -> Vec<&str> {
        let learned: Vec<&str> = self
            .commands
            .get(&cmd_type)
            .map(|cmds| {
                cmds.iter()
                    .filter(|c| c.success_count > 0)
                    .map(|c| c.command.as_str())
                    .collect()
            })
            .unwrap_or_default();

        learned
    }

    /// Clear history for a command type
    pub fn clear_type(&mut self, cmd_type: CommandType) {
        self.commands.remove(&cmd_type);
    }

    /// Get statistics summary
    pub fn stats(&self) -> CommandStats {
        let mut total_commands = 0;
        let mut successful_commands = 0;
        let mut total_runs = 0;

        for cmds in self.commands.values() {
            total_commands += cmds.len();
            for cmd in cmds {
                if cmd.success_count > 0 {
                    successful_commands += 1;
                }
                total_runs += cmd.total_runs();
            }
        }

        CommandStats {
            total_commands,
            successful_commands,
            total_runs,
        }
    }
}

/// Statistics about command history
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandStats {
    /// Total number of unique commands
    pub total_commands: usize,
    /// Commands with at least one success
    pub successful_commands: usize,
    /// Total execution count
    pub total_runs: u32,
}

/// Command learner - actively tries commands to learn what works
#[derive(Debug, Clone)]
pub struct CommandLearner {
    /// Command history
    history: CommandHistory,
    /// Project fingerprint (for default suggestions)
    project_type: Option<crate::fingerprint::ProjectType>,
}

impl CommandLearner {
    /// Create a new command learner
    pub fn new(project_type: Option<crate::fingerprint::ProjectType>) -> Self {
        Self {
            history: CommandHistory::default(),
            project_type,
        }
    }

    /// Get the next command to try for a given type
    ///
    /// Strategy:
    /// 1. Return the best known working command
    /// 2. Try default commands from project type
    /// 3. Return None if no suggestions
    pub fn suggest_command(&self, cmd_type: CommandType) -> Option<&str> {
        // First, try the best learned command
        if let Some(cmd) = self.history.best(cmd_type) {
            return Some(cmd);
        }

        // Second, try project type defaults
        if let Some(pt) = self.project_type {
            let default = match cmd_type {
                CommandType::Build => pt.default_build_command(),
                CommandType::Test => pt.default_test_command(),
                CommandType::Check => pt.default_check_command(),
                _ => None,
            };

            if default.is_some() {
                return default;
            }
        }

        // No suggestions
        None
    }

    /// Record the result of trying a command
    pub fn record_attempt(&mut self, cmd_type: CommandType, command: &str, success: bool) {
        self.history.record(cmd_type, command, success);
    }

    /// Get command history
    pub fn history(&self) -> &CommandHistory {
        &self.history
    }

    /// Get mutable command history
    pub fn history_mut(&mut self) -> &mut CommandHistory {
        &mut self.history
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_command_tracking() {
        let mut cmd = LearnedCommand::new("cargo build");

        cmd.record_success(Some(1000));
        cmd.record_success(Some(1200));
        cmd.record_failure();

        assert_eq!(cmd.success_count, 2);
        assert_eq!(cmd.failure_count, 1);
        assert!(cmd.last_success.is_some());
        assert!(cmd.last_failure.is_some());

        // 2/3 success rate
        assert!((cmd.success_rate() - 0.666).abs() < 0.01);

        // Not yet reliable (< 80%)
        assert!(!cmd.is_reliable());

        // Add two more successes to reach 4/5 = 80%
        cmd.record_success(Some(900));
        // 3/4 = 75% - still not reliable
        assert!(!cmd.is_reliable());

        cmd.record_success(Some(800));
        // 4/5 = 80% - now reliable
        assert!(cmd.is_reliable());
    }

    #[test]
    fn test_command_history_recording() {
        let mut history = CommandHistory::default();

        history.record(CommandType::Build, "make", true);
        history.record(CommandType::Build, "cargo build", true);
        history.record(CommandType::Build, "make", false); // Failed this time

        // Should prefer cargo build (100% success) over make (50%)
        assert_eq!(history.best(CommandType::Build), Some("cargo build"));
    }

    #[test]
    fn test_command_learner_suggestion() {
        let learner = CommandLearner::new(Some(crate::fingerprint::ProjectType::Rust));

        // Should suggest Rust default
        assert_eq!(
            learner.suggest_command(CommandType::Build),
            Some("cargo build")
        );
        assert_eq!(
            learner.suggest_command(CommandType::Test),
            Some("cargo test")
        );
    }

    #[test]
    fn test_most_recent_success() {
        let mut history = CommandHistory::default();

        history.record(CommandType::Build, "cargo build", true);
        // Manually set last_success to older timestamp
        if let Some(cmds) = history.commands.get_mut(&CommandType::Build) {
            cmds[0].last_success = Some(1000);
        }

        history.record(CommandType::Build, "make", true);
        if let Some(cmds) = history.commands.get_mut(&CommandType::Build) {
            cmds[0].last_success = Some(2000);
        }

        // make was more recent
        assert_eq!(history.most_recent_success(CommandType::Build), Some("make"));
    }
}
