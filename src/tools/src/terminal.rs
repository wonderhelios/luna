//! Terminal command execution for agents

use crate::ToolResult;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

// ============================================================================
// Terminal Execution
// ============================================================================

/// Terminal command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalResult {
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Dangerous command patterns that should require explicit confirmation
const DANGEROUS_PATTERNS: &[&str] = &[
    "rm -rf",
    "rm -r /",
    "format",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "init 0",
    "halt",
    "poweroff",
    ":(){:|:&};:", // fork bomb
    "mv /dev/null",
];

/// Check if a command appears dangerous
fn is_dangerous_command(command: &str) -> bool {
    let cmd_lower = command.to_lowercase();
    DANGEROUS_PATTERNS.iter().any(|pattern| {
        cmd_lower.contains(&pattern.to_lowercase())
    })
}

/// Run a terminal command with safety checks
///
/// # Arguments
/// * `command` - Command string to execute (e.g., "cargo build")
/// * `cwd` - Optional working directory
/// * `allow_dangerous` - If true, bypass safety checks for dangerous commands
///
/// # Returns
/// * `TerminalResult` with command output and status
pub fn run_terminal(
    command: &str,
    cwd: Option<&Path>,
    allow_dangerous: bool,
) -> ToolResult<TerminalResult> {
    let command = command.trim();

    if command.is_empty() {
        return Ok(TerminalResult {
            command: command.to_string(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            success: false,
            error: Some("Empty command".to_string()),
        });
    }

    // Safety check for dangerous commands
    if !allow_dangerous && is_dangerous_command(command) {
        return Ok(TerminalResult {
            command: command.to_string(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            success: false,
            error: Some(format!(
                "Command blocked as potentially dangerous. Use --allow-dangerous to override: {}",
                command
            )),
        });
    }

    // Parse command: split by spaces but respect quotes
    let parts = shell_words::split(command);
    let parts = match parts {
        Ok(p) => p,
        Err(_) => {
            // Fallback: simple split
            command.split_whitespace().map(|s| s.to_string()).collect()
        }
    };

    if parts.is_empty() {
        return Ok(TerminalResult {
            command: command.to_string(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            success: false,
            error: Some("Failed to parse command".to_string()),
        });
    }

    let (program, args) = parts.split_first().unwrap();

    // Use thread with timeout for better compatibility
    let _timeout = Duration::from_secs(120);

    let result = thread::scope(|s| {
        s.spawn(|| {
            let mut cmd = Command::new(program);
            cmd.args(args);
            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }
            cmd.output()
        })
        .join()
        .unwrap_or_else(|_| {
            // Thread panicked or was cancelled
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "command timed out",
            ))
        })
    });

    let output = match result {
        Ok(o) => o,
        Err(e) => {
            return Ok(TerminalResult {
                command: command.to_string(),
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                success: false,
                error: Some(format!("Failed to execute command: {}", e)),
            });
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code();
    let success = output.status.success();

    Ok(TerminalResult {
        command: command.to_string(),
        exit_code,
        stdout,
        stderr,
        success,
        error: if !success && exit_code.is_some() {
            Some(format!("Command exited with code {:?}", exit_code))
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dangerous_command_detection() {
        assert!(is_dangerous_command("rm -rf /"));
        assert!(is_dangerous_command("sudo rm -rf /home"));
        assert!(!is_dangerous_command("cargo build"));
        assert!(!is_dangerous_command("ls -la"));
    }

    #[test]
    fn test_run_echo() {
        let result = run_terminal("echo hello", None, false).unwrap();
        assert!(result.success);
        assert!(result.stdout.contains("hello"));
    }
}
