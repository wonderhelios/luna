//! Project Memory System
//!
//! This module provides persistent memory for Luna, enabling it to learn
//! and remember project-specific information across sessions.
//!
//! ## Core Concepts
//!
//! - `ProjectMemory`: The main memory structure containing all learned information
//! - `ProjectFingerprint`: Auto-detected project type and characteristics
//! - `CommandHistory`: Learned build/test/check commands that work
//! - `UserPreferences`: User's coding style and preferences
//! - `MemoryStore`: Persistence layer (LUNA.md or .luna/memory.json)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use memory::{ProjectMemory, MemoryStore, CommandType};
//!
//! // Load or create memory for current project
//! let store = MemoryStore::new(".luna/memory.json");
//! let mut memory = store.load().unwrap_or_default();
//!
//! // Record a successful command
//! memory.record_command(CommandType::Build, "cargo build --release", true);
//!
//! // Get the best build command
//! if let Some(cmd) = memory.best_command(CommandType::Build) {
//!     println!("Build with: {}", cmd);
//! }
//!
//! // Save memory
//! store.save(&memory).unwrap();
//! ```

mod fingerprint;
mod learner;
mod store;

use std::collections::HashMap;

pub use fingerprint::{ProjectFingerprint, ProjectType};
pub use learner::{CommandHistory, CommandType, CommandLearner};
pub use store::{MemoryStore, auto_detect};

/// Project memory - all learned information about a project
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProjectMemory {
    /// Version for migration compatibility
    pub version: u32,
    /// Project fingerprint (auto-detected)
    pub fingerprint: ProjectFingerprint,
    /// Learned commands
    pub commands: CommandHistory,
    /// User preferences
    pub preferences: UserPreferences,
    /// Code style preferences
    pub code_style: CodeStylePreferences,
    /// Custom metadata (project-specific)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Default for ProjectMemory {
    fn default() -> Self {
        Self {
            version: 1,
            fingerprint: ProjectFingerprint::default(),
            commands: CommandHistory::default(),
            preferences: UserPreferences::default(),
            code_style: CodeStylePreferences::default(),
            metadata: HashMap::new(),
        }
    }
}

impl ProjectMemory {
    /// Create new memory with auto-detected fingerprint
    pub fn detect(project_root: &std::path::Path) -> Self {
        Self {
            version: 1,
            fingerprint: ProjectFingerprint::detect(project_root),
            commands: CommandHistory::default(),
            preferences: UserPreferences::default(),
            code_style: CodeStylePreferences::default(),
            metadata: HashMap::new(),
        }
    }

    /// Record a command execution result
    pub fn record_command(&mut self, cmd_type: CommandType, command: &str, success: bool) {
        self.commands.record(cmd_type, command, success);
    }

    /// Get the best command for a specific type
    pub fn best_command(&self, cmd_type: CommandType) -> Option<&str> {
        self.commands.best(cmd_type)
    }

    /// Check if we have a working command for this type
    pub fn has_working_command(&self, cmd_type: CommandType) -> bool {
        self.commands.has_successful(cmd_type)
    }
}

/// User preferences for the assistant behavior
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct UserPreferences {
    /// Preferred programming style
    pub style: ProgrammingStyle,
    /// Whether to ask for confirmation before edits
    pub confirm_edits: bool,
    /// Whether to explain changes
    pub explain_changes: bool,
    /// Preferred output verbosity
    pub verbosity: Verbosity,
    /// Custom prompts to prepend to LLM requests
    #[serde(default)]
    pub custom_prompts: Vec<String>,
}

/// Programming style preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgrammingStyle {
    /// Concise, minimal code
    Concise,
    /// Verbose, explicit code
    Verbose,
    /// Performance-oriented
    Performance,
    /// Safety-oriented
    Safety,
    /// Balanced (default)
    #[default]
    Balanced,
}

/// Output verbosity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verbosity {
    /// Minimal output
    Quiet,
    /// Normal output (default)
    #[default]
    Normal,
    /// Detailed output
    Verbose,
    /// Debug-level output
    Debug,
}

/// Code style preferences
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct CodeStylePreferences {
    /// Preferred naming convention
    pub naming: NamingConvention,
    /// Indentation style
    pub indentation: IndentationStyle,
    /// Maximum line length
    pub max_line_length: Option<usize>,
    /// Import organization preference
    pub import_style: ImportStyle,
    /// Documentation style
    pub doc_style: DocStyle,
}

/// Naming convention preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamingConvention {
    /// Follow project conventions
    #[default]
    FollowProject,
    /// snake_case
    SnakeCase,
    /// camelCase
    CamelCase,
    /// PascalCase
    PascalCase,
    /// kebab-case
    KebabCase,
}

/// Indentation style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndentationStyle {
    /// Spaces (default: 4)
    #[default]
    Spaces4,
    Spaces2,
    Tabs,
}

/// Import organization style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportStyle {
    /// Group by type (std, external, internal)
    #[default]
    Grouped,
    /// Alphabetical
    Alphabetical,
    /// As-is
    Unsorted,
}

/// Documentation style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocStyle {
    /// Full documentation
    #[default]
    Full,
    /// Public API only
    PublicOnly,
    /// Minimal
    Minimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_project_memory_default() {
        let memory = ProjectMemory::default();
        assert_eq!(memory.version, 1);
        assert!(memory.commands.is_empty());
    }

    #[test]
    fn test_record_and_retrieve_command() {
        let mut memory = ProjectMemory::default();

        memory.record_command(CommandType::Build, "cargo build", true);
        memory.record_command(CommandType::Test, "cargo test", true);

        assert_eq!(memory.best_command(CommandType::Build), Some("cargo build"));
        assert_eq!(memory.best_command(CommandType::Test), Some("cargo test"));
        assert!(memory.has_working_command(CommandType::Build));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut memory = ProjectMemory::default();
        memory.record_command(CommandType::Build, "make", true);
        memory.preferences.style = ProgrammingStyle::Performance;

        let json = serde_json::to_string(&memory).unwrap();
        let restored: ProjectMemory = serde_json::from_str(&json).unwrap();

        assert_eq!(memory, restored);
    }
}
