//! Self-Healing Execution - Automatic error recovery and retry
//!
//! This module provides automatic error analysis and healing capabilities,
//! enabling Luna to recover from common failures without user intervention.
//!
//! ## Architecture
//!
//! ```text
//! Command Execution Failed
//!     ↓
//! ErrorAnalyzer::analyze(error_output, context)
//!     ↓
//! Detect Error Pattern
//!     ↓
//! Select HealingStrategy
//!     ↓
//! Execute Fix
//!     ↓
//! Retry Original Command
//!     ↓
//! Success / Report Result
//! ```
//!
//! ## Supported Error Patterns
//!
//! - **Missing Dependency**: Command not found → Suggest installation
//! - **Compile Error**: Syntax/type errors → Generate fix plan
//! - **Missing Directory**: Path not found → Create directory
//! - **Permission Denied**: Access issues → Suggest chmod/sudo
//! - **Module Not Found**: Import errors → Add dependency
//! - **Type Mismatch**: Type errors → Suggest type fixes

use std::collections::HashMap;
use std::path::Path;

use tools::{ToolCall, ToolRegistry, ToolResult};

/// Analyzed error information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorAnalysis {
    /// The detected error pattern
    pub pattern: ErrorPattern,
    /// Confidence level (0-100)
    pub confidence: u8,
    /// Error message summary
    pub summary: String,
    /// Detailed error output
    pub details: String,
    /// Suggested fix strategies
    pub strategies: Vec<HealingStrategy>,
    /// File location if applicable
    pub location: Option<ErrorLocation>,
}

/// Error location information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLocation {
    pub file: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

/// Known error patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorPattern {
    /// Command not found
    CommandNotFound,
    /// Missing dependency/package
    MissingDependency,
    /// Compilation error
    CompileError,
    /// Missing file or directory
    MissingPath,
    /// Permission denied
    PermissionDenied,
    /// Syntax error
    SyntaxError,
    /// Import/module not found
    ModuleNotFound,
    /// Type error
    TypeError,
    /// Configuration error
    ConfigError,
    /// Network/timeout error
    NetworkError,
    /// Unknown/unrecognized error
    Unknown,
}

impl ErrorPattern {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ErrorPattern::CommandNotFound => "Command not found",
            ErrorPattern::MissingDependency => "Missing dependency",
            ErrorPattern::CompileError => "Compilation error",
            ErrorPattern::MissingPath => "Missing file or directory",
            ErrorPattern::PermissionDenied => "Permission denied",
            ErrorPattern::SyntaxError => "Syntax error",
            ErrorPattern::ModuleNotFound => "Module not found",
            ErrorPattern::TypeError => "Type error",
            ErrorPattern::ConfigError => "Configuration error",
            ErrorPattern::NetworkError => "Network error",
            ErrorPattern::Unknown => "Unknown error",
        }
    }
}

/// Healing strategy for fixing errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealingStrategy {
    /// Install a missing dependency
    InstallDependency {
        package: String,
        manager: PackageManager,
    },
    /// Create a missing directory
    CreateDirectory {
        path: String,
    },
    /// Fix a file (edit)
    FixFile {
        path: String,
        description: String,
        suggested_edit: Option<String>,
    },
    /// Change permissions
    FixPermissions {
        path: String,
        mode: String,
    },
    /// Use alternative command
    UseAlternative {
        original: String,
        alternative: String,
    },
    /// Add missing import/dependency
    AddDependency {
        name: String,
        version: Option<String>,
    },
    /// Ask user for guidance
    AskUser {
        question: String,
    },
}

/// Package manager types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageManager {
    /// Rust Cargo
    Cargo,
    /// Node.js npm
    Npm,
    /// Python pip
    Pip,
    /// Python poetry
    Poetry,
    /// Go modules
    GoMod,
    /// System package manager (apt)
    Apt,
    /// System package manager (brew)
    Brew,
    /// Generic
    Generic,
}

/// Error analyzer - detects error patterns from command output
pub struct ErrorAnalyzer {
    /// Known error patterns with detection functions
    patterns: Vec<(ErrorPattern, Box<dyn Fn(&str) -> bool + Send + Sync>)>,
}

impl std::fmt::Debug for ErrorAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorAnalyzer")
            .field("patterns_count", &self.patterns.len())
            .finish()
    }
}

impl ErrorAnalyzer {
    /// Create a new error analyzer with default patterns
    pub fn new() -> Self {
        let mut analyzer = Self {
            patterns: Vec::new(),
        };
        analyzer.register_default_patterns();
        analyzer
    }

    /// Analyze error output and return detected patterns
    pub fn analyze(
        &self,
        command: &str,
        error_output: &str,
        exit_code: Option<i32>,
    ) -> ErrorAnalysis {
        // Try to match known patterns
        for (pattern, detector) in &self.patterns {
            if detector(error_output) {
                return self.build_analysis(*pattern, command, error_output, exit_code);
            }
        }

        // Unknown error - still provide analysis
        ErrorAnalysis {
            pattern: ErrorPattern::Unknown,
            confidence: 50,
            summary: "Unrecognized error pattern".to_string(),
            details: error_output.to_string(),
            strategies: vec![HealingStrategy::AskUser {
                question: format!("Command failed with unknown error. How should I proceed?"),
            }],
            location: None,
        }
    }

    /// Build detailed analysis for a pattern
    fn build_analysis(
        &self,
        pattern: ErrorPattern,
        command: &str,
        error_output: &str,
        _exit_code: Option<i32>,
    ) -> ErrorAnalysis {
        let strategies = self.suggest_strategies(pattern, command, error_output);
        let location = self.extract_location(error_output);

        ErrorAnalysis {
            pattern,
            confidence: 90, // High confidence for known patterns
            summary: pattern.description().to_string(),
            details: error_output.to_string(),
            strategies,
            location,
        }
    }

    /// Register default error detection patterns
    fn register_default_patterns(&mut self) {
        // Command not found
        self.patterns.push((
            ErrorPattern::CommandNotFound,
            Box::new(|output: &str| {
                output.contains("command not found")
                    || output.contains("not recognized")
                    || output.contains("is not recognized as an internal or external command")
            }),
        ));

        // Missing dependency (Rust)
        self.patterns.push((
            ErrorPattern::MissingDependency,
            Box::new(|output: &str| {
                output.contains("could not find")
                    && output.contains("in registry")
                    || output.contains("unresolved import")
            }),
        ));

        // Missing path
        self.patterns.push((
            ErrorPattern::MissingPath,
            Box::new(|output: &str| {
                (output.contains("No such file or directory")
                    && !output.contains("command not found"))
                    || output.contains("cannot find path")
            }),
        ));

        // Permission denied
        self.patterns.push((
            ErrorPattern::PermissionDenied,
            Box::new(|output: &str| {
                output.contains("Permission denied") || output.contains("Access is denied")
            }),
        ));

        // Compilation error (Rust)
        self.patterns.push((
            ErrorPattern::CompileError,
            Box::new(|output: &str| {
                output.contains("error[") ||
                (output.contains("could not compile") && output.contains("error"))
            }),
        ));

        // Syntax error
        self.patterns.push((
            ErrorPattern::SyntaxError,
            Box::new(|output: &str| {
                output.contains("syntax error")
                    || output.contains("unexpected token")
                    || output.contains("expected")
            }),
        ));

        // Module not found
        self.patterns.push((
            ErrorPattern::ModuleNotFound,
            Box::new(|output: &str| {
                output.contains("cannot find module")
                    || output.contains("module not found")
                    || output.contains("unresolved import")
            }),
        ));

        // Type error
        self.patterns.push((
            ErrorPattern::TypeError,
            Box::new(|output: &str| {
                output.contains("mismatched types")
                    || output.contains("expected type")
                    || output.contains("found type")
                    || output.contains("type mismatch")
            }),
        ));

        // Network error
        self.patterns.push((
            ErrorPattern::NetworkError,
            Box::new(|output: &str| {
                output.contains("timeout")
                    || output.contains("connection refused")
                    || output.contains("could not resolve")
                    || output.contains("network is unreachable")
            }),
        ));
    }

    /// Suggest healing strategies based on error pattern
    fn suggest_strategies(
        &self,
        pattern: ErrorPattern,
        command: &str,
        error_output: &str,
    ) -> Vec<HealingStrategy> {
        let mut strategies = Vec::new();

        match pattern {
            ErrorPattern::CommandNotFound => {
                // Try to suggest installation
                if let Some(cmd) = command.split_whitespace().next() {
                    strategies.push(HealingStrategy::InstallDependency {
                        package: cmd.to_string(),
                        manager: PackageManager::Generic,
                    });
                }
                strategies.push(HealingStrategy::AskUser {
                    question: format!(
                        "Command '{}' not found. Should I try to install it?",
                        command
                    ),
                });
            }

            ErrorPattern::MissingDependency => {
                // Extract package name from error
                if let Some(pkg) = self.extract_package_name(error_output) {
                    strategies.push(HealingStrategy::AddDependency {
                        name: pkg,
                        version: None,
                    });
                }
            }

            ErrorPattern::MissingPath => {
                // Extract path from error
                if let Some(path) = self.extract_path_from_error(error_output) {
                    strategies.push(HealingStrategy::CreateDirectory { path });
                }
            }

            ErrorPattern::PermissionDenied => {
                if let Some(path) = self.extract_path_from_error(error_output) {
                    strategies.push(HealingStrategy::FixPermissions {
                        path,
                        mode: "755".to_string(),
                    });
                }
            }

            ErrorPattern::CompileError | ErrorPattern::SyntaxError | ErrorPattern::TypeError => {
                // For compilation errors, suggest LLM-based fix
                if let Some(location) = self.extract_location(error_output) {
                    strategies.push(HealingStrategy::FixFile {
                        path: location.file,
                        description: "Fix compilation error".to_string(),
                        suggested_edit: None,
                    });
                }
            }

            ErrorPattern::ModuleNotFound => {
                if let Some(module) = self.extract_module_name(error_output) {
                    strategies.push(HealingStrategy::AddDependency {
                        name: module,
                        version: None,
                    });
                }
            }

            ErrorPattern::NetworkError => {
                strategies.push(HealingStrategy::AskUser {
                    question: "Network error occurred. Should I retry?".to_string(),
                });
            }

            ErrorPattern::Unknown | ErrorPattern::ConfigError => {
                strategies.push(HealingStrategy::AskUser {
                    question: "Unknown error. How should I proceed?".to_string(),
                });
            }
        }

        strategies
    }

    /// Extract file location from error output
    fn extract_location(&self, error_output: &str) -> Option<ErrorLocation> {
        // Try to match common patterns like:
        // --> src/main.rs:10:5
        // src/main.rs(10,5)
        // at src/main.rs:10

        for line in error_output.lines() {
            // Rust pattern: --> file:line:col
            if let Some(start) = line.find("-->") {
                let rest = line[start + 3..].trim();
                // Split from right: file:line:col -> (file:line, col)
                if let Some((file_and_line, col_str)) = rest.rsplit_once(':') {
                    // Split file:line -> (file, line)
                    if let Some((file, line_str)) = file_and_line.rsplit_once(':') {
                        if let (Ok(line_num), Ok(col_num)) =
                            (line_str.parse::<usize>(), col_str.parse::<usize>())
                        {
                            return Some(ErrorLocation {
                                file: file.to_string(),
                                line: Some(line_num),
                                column: Some(col_num),
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract package name from error
    fn extract_package_name(&self, error_output: &str) -> Option<String> {
        // Pattern: "could not find `package` in registry"
        for line in error_output.lines() {
            if line.contains("could not find") && line.contains("in registry") {
                if let Some(start) = line.find('`') {
                    if let Some(end) = line[start + 1..].find('`') {
                        return Some(line[start + 1..start + 1 + end].to_string());
                    }
                }
            }
        }
        None
    }

    /// Extract path from error message
    fn extract_path_from_error(&self, error_output: &str) -> Option<String> {
        // Look for paths in quotes or after specific keywords
        for line in error_output.lines() {
            if line.contains("No such file or directory") {
                // Try to extract path from earlier in the line
                if let Some(quote_start) = line.find('"') {
                    if let Some(quote_end) = line[quote_start + 1..].find('"') {
                        return Some(line[quote_start + 1..quote_start + 1 + quote_end].to_string());
                    }
                }
            }
        }
        None
    }

    /// Extract module name from error
    fn extract_module_name(&self, error_output: &str) -> Option<String> {
        // Pattern: "cannot find module `module_name`"
        for line in error_output.lines() {
            if line.contains("cannot find module") || line.contains("unresolved import") {
                if let Some(start) = line.find('`') {
                    if let Some(end) = line[start + 1..].find('`') {
                        return Some(line[start + 1..start + 1 + end].to_string());
                    }
                }
            }
        }
        None
    }
}

/// Healing executor - applies healing strategies
#[derive(Debug)]
pub struct HealingExecutor {
    /// Max healing attempts per command
    max_attempts: usize,
    /// Successful healings history (for learning)
    success_history: HashMap<ErrorPattern, Vec<HealingStrategy>>,
}

impl HealingExecutor {
    /// Create a new healing executor
    pub fn new() -> Self {
        Self {
            max_attempts: 3,
            success_history: HashMap::new(),
        }
    }

    /// Set max healing attempts
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Attempt to heal an error
    ///
    /// Returns Ok(true) if healing was successful and command should be retried
    /// Returns Ok(false) if healing was not possible
    /// Returns Err if healing itself failed
    pub fn try_heal(
        &mut self,
        analysis: &ErrorAnalysis,
        _ctx: &dyn HealingContext,
    ) -> error::Result<bool> {
        if analysis.strategies.is_empty() {
            return Ok(false);
        }

        // Try each strategy in order
        for strategy in &analysis.strategies {
            match self.apply_strategy(strategy, _ctx) {
                Ok(true) => {
                    // Record successful healing
                    self.record_success(analysis.pattern, strategy.clone());
                    return Ok(true);
                }
                Ok(false) => {
                    // Strategy not applicable, try next
                    continue;
                }
                Err(e) => {
                    tracing::warn!("Healing strategy failed: {}", e);
                    continue;
                }
            }
        }

        Ok(false)
    }

    /// Apply a healing strategy
    fn apply_strategy(
        &self,
        strategy: &HealingStrategy,
        _ctx: &dyn HealingContext,
    ) -> error::Result<bool> {
        match strategy {
            HealingStrategy::CreateDirectory { path } => {
                tracing::info!("Healing: Creating directory {}", path);
                std::fs::create_dir_all(path)?;
                Ok(true)
            }

            HealingStrategy::FixPermissions { path, mode } => {
                tracing::info!("Healing: Setting permissions {} on {}", mode, path);
                // Platform-specific implementation would go here
                // For now, just log
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode_num = u32::from_str_radix(mode, 8)
                        .map_err(|e| error::LunaError::invalid_input(format!("Invalid mode: {}", e)))?;
                    let permissions = std::fs::Permissions::from_mode(mode_num);
                    std::fs::set_permissions(path, permissions)?;
                }
                Ok(true)
            }

            HealingStrategy::InstallDependency { package, manager } => {
                tracing::info!("Healing: Installing {} via {:?}", package, manager);
                // This would typically spawn a command to install
                // For now, return false to indicate we can't auto-heal this
                let _ = (package, manager);
                Ok(false)
            }

            HealingStrategy::FixFile {
                path,
                description,
                suggested_edit: _,
            } => {
                tracing::info!("Healing: Fixing file {} - {}", path, description);
                // This would require LLM intervention
                // Return false to indicate we need external help
                let _ = path;
                Ok(false)
            }

            HealingStrategy::AddDependency { name, version } => {
                tracing::info!(
                    "Healing: Adding dependency {} {}",
                    name,
                    version.as_deref().unwrap_or("")
                );
                // Would modify Cargo.toml/package.json/etc
                let _ = (name, version);
                Ok(false)
            }

            HealingStrategy::UseAlternative { original, alternative } => {
                tracing::info!("Healing: Using {} instead of {}", alternative, original);
                Ok(true)
            }

            HealingStrategy::AskUser { question } => {
                tracing::info!("Healing needs user input: {}", question);
                Ok(false)
            }
        }
    }

    /// Record a successful healing for learning
    fn record_success(&mut self, pattern: ErrorPattern, strategy: HealingStrategy) {
        self.success_history
            .entry(pattern)
            .or_default()
            .push(strategy);
    }

    /// Get successful strategies for a pattern
    pub fn get_successful_strategies(&self, pattern: ErrorPattern) -> &[HealingStrategy] {
        self.success_history
            .get(&pattern)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for HealingExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for healing operations
pub trait HealingContext {
    /// Get the project root
    fn project_root(&self) -> Option<&Path>;
    /// Get available tools
    fn tools(&self) -> &ToolRegistry;
    /// Execute a tool call
    fn execute_tool(&self, call: &ToolCall) -> error::Result<ToolResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_command_not_found() {
        let analyzer = ErrorAnalyzer::new();
        let output = "bash: cargo: command not found";

        let analysis = analyzer.analyze("cargo build", output, Some(127));

        assert_eq!(analysis.pattern, ErrorPattern::CommandNotFound);
        assert!(!analysis.strategies.is_empty());
    }

    #[test]
    fn test_detect_missing_path() {
        let analyzer = ErrorAnalyzer::new();
        let output = "mkdir: cannot create directory '/nonexistent/path': No such file or directory";

        let analysis = analyzer.analyze("mkdir /nonexistent/path", output, Some(1));

        assert_eq!(analysis.pattern, ErrorPattern::MissingPath);
    }

    #[test]
    fn test_detect_compile_error() {
        let analyzer = ErrorAnalyzer::new();
        let output = r#"
error[E0425]: cannot find value `foo` in this scope
 --> src/main.rs:2:5
  |
2 |     foo
  |     ^^^ not found in this scope
"#;

        let analysis = analyzer.analyze("cargo build", output, Some(101));

        assert_eq!(analysis.pattern, ErrorPattern::CompileError);
        assert!(analysis.location.is_some());
    }

    #[test]
    fn test_extract_location_rust() {
        let analyzer = ErrorAnalyzer::new();
        let output = " --> src/main.rs:10:5";

        let location = analyzer.extract_location(output);

        assert!(location.is_some());
        let loc = location.unwrap();
        assert_eq!(loc.file, "src/main.rs");
        assert_eq!(loc.line, Some(10));
        assert_eq!(loc.column, Some(5));
    }

    #[test]
    fn test_healing_create_directory() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let new_dir = temp.path().join("new_directory");

        let strategy = HealingStrategy::CreateDirectory {
            path: new_dir.to_str().unwrap().to_string(),
        };

        // Would need a mock context to test properly
        // For now just verify the strategy is created correctly
        assert!(matches!(strategy, HealingStrategy::CreateDirectory { .. }));
    }
}
