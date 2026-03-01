//! Structured error record types for build error parsing
//!
//! This module defines the unified error representation that all
//! BuildErrorParsers must convert their tool-specific output into.

use serde::{Deserialize, Serialize};

/// A structured record of a build/compile error
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorRecord {
    /// The command that produced this error (e.g., "cargo build")
    pub command: String,

    /// Exit code of the command, if available
    pub exit_code: Option<i32>,

    /// Classification of the error type
    pub kind: ErrorKind,

    /// Primary error message (human-readable summary)
    pub message: String,

    /// Error code from the tool, if available (e.g., "E0425" from rustc)
    pub error_code: Option<String>,

    /// Locations associated with this error (usually 1, sometimes multiple)
    pub locations: Vec<Location>,

    /// Suggested fix or hint from the compiler, if available
    pub suggestion: Option<String>,

    /// Raw error text from the tool output (for debugging)
    pub raw: String,
}

/// Classification of error types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Syntax or semantic error in the code
    CompileError,

    /// Test assertion failed
    TestFailure,

    /// Missing dependency or package not found
    DependencyError,

    /// Toolchain/environment issue (not code-related)
    InfraError,

    /// Lint or warning treated as error
    LintError,

    /// Uncategorized
    Unknown,
}

impl ErrorKind {
    /// Determine if this error type is likely fixable by code changes
    pub fn is_fixable(&self) -> bool {
        matches!(self, ErrorKind::CompileError | ErrorKind::LintError | ErrorKind::TestFailure)
    }

    /// Determine if we should continue trying to fix or stop early
    pub fn should_stop(&self) -> bool {
        matches!(self, ErrorKind::DependencyError | ErrorKind::InfraError)
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::CompileError => write!(f, "compile error"),
            ErrorKind::TestFailure => write!(f, "test failure"),
            ErrorKind::DependencyError => write!(f, "dependency error"),
            ErrorKind::InfraError => write!(f, "infrastructure error"),
            ErrorKind::LintError => write!(f, "lint error"),
            ErrorKind::Unknown => write!(f, "unknown error"),
        }
    }
}

/// A specific location (file, line, column) where an error occurred
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Location {
    /// Absolute or relative path to the file
    pub path: String,

    /// 1-based line number
    pub line: usize,

    /// 1-based column number (0 if unknown)
    pub column: usize,

    /// Symbol name if we can extract it (e.g., function name, variable name)
    pub symbol: Option<String>,

    /// The specific line of code if available
    pub code_snippet: Option<String>,

    /// Raw location string from the error output
    pub raw: String,
}

impl Location {
    /// Create a new location with minimal information
    pub fn new(path: impl Into<String>, line: usize) -> Self {
        let path = path.into();
        Self {
            path: path.clone(),
            line,
            column: 0,
            symbol: None,
            code_snippet: None,
            raw: format!("{}:{}", path, line),
        }
    }

    /// Create a location with a raw string from parsed output
    pub fn with_raw(path: impl Into<String>, line: usize, raw: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            line,
            column: 0,
            symbol: None,
            code_snippet: None,
            raw: raw.into(),
        }
    }

    /// Set the raw location string (builder pattern)
    pub fn set_raw(mut self, raw: impl Into<String>) -> Self {
        self.raw = raw.into();
        self
    }

    /// Add column information
    pub fn with_column(mut self, column: usize) -> Self {
        self.column = column;
        self
    }

    /// Add symbol information
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Add code snippet
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.code_snippet = Some(snippet.into());
        self
    }
}

/// Summary statistics for a collection of errors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Total number of errors
    pub total: usize,

    /// Number of errors by kind
    pub by_kind: std::collections::HashMap<ErrorKind, usize>,

    /// Unique files with errors
    pub affected_files: std::collections::HashSet<String>,

    /// Whether any errors are unfixable (dependency/infra)
    pub has_unfixable: bool,
}

impl ErrorSummary {
    /// Create a summary from a list of error records
    pub fn from_records(records: &[ErrorRecord]) -> Self {
        let mut by_kind = std::collections::HashMap::new();
        let mut affected_files = std::collections::HashSet::new();
        let mut has_unfixable = false;

        for record in records {
            *by_kind.entry(record.kind).or_insert(0) += 1;

            for loc in &record.locations {
                affected_files.insert(loc.path.clone());
            }

            if !record.kind.is_fixable() || record.kind.should_stop() {
                has_unfixable = true;
            }
        }

        Self {
            total: records.len(),
            by_kind,
            affected_files,
            has_unfixable,
        }
    }

    /// Check if the error set indicates progress compared to another
    pub fn is_progress(&self, other: &ErrorSummary) -> Option<bool> {
        if self.total < other.total {
            Some(true)  // Fewer errors = progress
        } else if self.total > other.total {
            Some(false) // More errors = regression
        } else {
            // Same count, check if affected files changed
            if self.affected_files != other.affected_files {
                Some(true) // Different files might indicate shift in error location
            } else {
                None // No clear progress indicator
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_is_fixable() {
        assert!(ErrorKind::CompileError.is_fixable());
        assert!(ErrorKind::TestFailure.is_fixable());
        assert!(ErrorKind::LintError.is_fixable());
        assert!(!ErrorKind::DependencyError.is_fixable());
        assert!(!ErrorKind::InfraError.is_fixable());
    }

    #[test]
    fn test_error_kind_should_stop() {
        assert!(ErrorKind::DependencyError.should_stop());
        assert!(ErrorKind::InfraError.should_stop());
        assert!(!ErrorKind::CompileError.should_stop());
    }

    #[test]
    fn test_location_builder() {
        let loc = Location::new("src/main.rs", 42)
            .with_column(10)
            .with_symbol("my_function")
            .with_snippet("let x = 5;");

        assert_eq!(loc.path, "src/main.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, 10);
        assert_eq!(loc.symbol, Some("my_function".to_string()));
        assert_eq!(loc.code_snippet, Some("let x = 5;".to_string()));
    }

    #[test]
    fn test_error_summary() {
        let records = vec![
            ErrorRecord {
                command: "cargo build".to_string(),
                exit_code: Some(101),
                kind: ErrorKind::CompileError,
                message: "cannot find value".to_string(),
                error_code: Some("E0425".to_string()),
                locations: vec![Location::new("src/lib.rs", 10)],
                suggestion: None,
                raw: "error[E0425]: cannot find value".to_string(),
            },
            ErrorRecord {
                command: "cargo build".to_string(),
                exit_code: Some(101),
                kind: ErrorKind::CompileError,
                message: "mismatched types".to_string(),
                error_code: Some("E0308".to_string()),
                locations: vec![Location::new("src/lib.rs", 20)],
                suggestion: None,
                raw: "error[E0308]: mismatched types".to_string(),
            },
        ];

        let summary = ErrorSummary::from_records(&records);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.affected_files.len(), 1);
        assert!(!summary.has_unfixable);
    }
}
