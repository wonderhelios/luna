//! Build error parsing for automated fix loops
//!
//! This module provides structured parsing of build tool output (cargo, tsc, etc.)
//! to enable Agent-driven error fixing workflows.
//!
//! # Architecture
//!
//! - `BuildErrorParser` trait: Implemented by each tool-specific parser
//! - `ErrorParserRegistry`: Routes to the appropriate parser or falls back to default
//! - `ErrorRecord`: Unified representation of any build error
//!
//! # Usage
//!
//! ```rust,ignore
//! use tools::error_parse::{ErrorParserRegistry, CargoErrorParser};
//!
//! let mut registry = ErrorParserRegistry::new();
//! registry.register("cargo build", CargoErrorParser::new());
//! registry.register("cargo test", CargoErrorParser::new());
//!
//! let output = "error: example output";
//! let errors = registry.parse("cargo build", output, Some(101));
//! ```

use std::collections::HashMap;

mod cargo;
mod default;
mod error_record;

pub use cargo::CargoErrorParser;
pub use default::DefaultErrorParser;
pub use error_record::{ErrorKind, ErrorRecord, ErrorSummary, Location};

/// Trait for parsing build tool output into structured error records
///
/// Implementors should handle the specific output format of a build tool
/// and convert it into a vector of `ErrorRecord` structs.
pub trait BuildErrorParser: Send + Sync {
    /// Parse the raw output of a build command into structured error records
    ///
    /// # Arguments
    /// * `output` - The stderr or combined output from the build tool
    /// * `exit_code` - The exit code of the command, if available
    ///
    /// # Returns
    /// A vector of `ErrorRecord`. Empty if no errors were found or parsing failed.
    fn parse(&self, output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord>;

    /// Check if this parser supports the given command
    ///
    /// The command string is matched against known patterns.
    fn supports(&self, command: &str) -> bool;

    /// Get a description of this parser
    fn description(&self) -> &str;
}

/// Registry for managing multiple error parsers
///
/// Routes parsing requests to the appropriate parser based on command name,
/// with a fallback to a heuristic-based default parser.
#[derive(Default)]
pub struct ErrorParserRegistry {
    parsers: HashMap<String, Box<dyn BuildErrorParser>>,
    fallback: DefaultErrorParser,
}

impl ErrorParserRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
            fallback: DefaultErrorParser::new(),
        }
    }

    /// Register a parser for a specific command pattern
    ///
    /// # Example
    /// ```rust,no_run
    /// use tools::error_parse::{ErrorParserRegistry, CargoErrorParser};
    ///
    /// let mut registry = ErrorParserRegistry::new();
    /// registry.register("cargo build", CargoErrorParser::new());
    /// ```
    pub fn register<P: BuildErrorParser + 'static>(
        &mut self,
        command_pattern: impl Into<String>,
        parser: P,
    ) {
        self.parsers.insert(command_pattern.into(), Box::new(parser));
    }

    /// Parse output using the appropriate parser for the command
    ///
    /// If no specific parser is registered, falls back to `DefaultErrorParser`
    /// which uses regex heuristics to extract file:line:message patterns.
    pub fn parse(
        &self,
        command: &str,
        output: &str,
        exit_code: Option<i32>,
    ) -> Vec<ErrorRecord> {
        // Find a parser that supports this command
        for (pattern, parser) in &self.parsers {
            if command.contains(pattern) || parser.supports(command) {
                return parser.parse(output, exit_code);
            }
        }

        // Fallback to default parser
        self.fallback.parse(output, exit_code)
    }

    /// Check if a specific parser is registered for this command
    pub fn has_parser(&self, command: &str) -> bool {
        self.parsers.keys().any(|p| command.contains(p))
    }

    /// Get the number of registered parsers
    pub fn len(&self) -> usize {
        self.parsers.len()
    }

    /// Check if no parsers are registered
    pub fn is_empty(&self) -> bool {
        self.parsers.is_empty()
    }
}

/// Convenience function to parse cargo build output
pub fn parse_cargo_errors(output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
    let parser = CargoErrorParser::new();
    parser.parse(output, exit_code)
}

/// Convenience function to parse with default heuristic parser
pub fn parse_generic_errors(output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
    let parser = DefaultErrorParser::new();
    parser.parse(output, exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_registration() {
        let mut registry = ErrorParserRegistry::new();
        assert!(registry.is_empty());

        registry.register("cargo build", CargoErrorParser::new());
        assert_eq!(registry.len(), 1);
        assert!(registry.has_parser("cargo build"));
        assert!(!registry.has_parser("npm test"));
    }

    #[test]
    fn test_registry_uses_fallback() {
        let registry = ErrorParserRegistry::new();

        // No parsers registered, should use fallback
        let output = "src/main.rs:10:5: error: something went wrong";
        let errors = registry.parse("some_command", output, Some(1));

        // Default parser should extract this
        assert!(!errors.is_empty());
        assert_eq!(errors[0].locations[0].path, "src/main.rs");
        assert_eq!(errors[0].locations[0].line, 10);
    }
}
