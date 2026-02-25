//! Error types for the tools module

use std::fmt;
use std::io;
use std::path::PathBuf;

/// Errors that can occur during tool operations
#[derive(Debug)]
pub enum ToolError {
    /// I/O error
    Io(io::Error),

    /// File not found
    FileNotFound(PathBuf),

    /// Invalid file path
    InvalidPath(PathBuf),

    /// Parse error
    Parse(String),

    /// Edit operation failed
    EditFailed(String),

    /// Terminal command failed
    TerminalFailed(String),

    /// Search operation failed
    SearchFailed(String),

    /// Invalid arguments provided
    InvalidArgument(String),

    /// Line range out of bounds
    InvalidLineRange {
        /// Start line (0-based)
        start: usize,
        /// End line (0-based)
        end: usize,
        /// Total lines in file
        total: usize,
    },

    /// Other error
    Other(String),
}

impl ToolError {
    /// Create a new error for invalid arguments
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        ToolError::InvalidArgument(msg.into())
    }

    /// Create a new error for search failures
    pub fn search_failed<S: Into<String>>(msg: S) -> Self {
        ToolError::SearchFailed(msg.into())
    }

    /// Create a new error for edit failures
    pub fn edit_failed<S: Into<String>>(msg: S) -> Self {
        ToolError::EditFailed(msg.into())
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::Io(e) => write!(f, "I/O error: {}", e),
            ToolError::FileNotFound(p) => write!(f, "file not found: {}", p.display()),
            ToolError::InvalidPath(p) => write!(f, "invalid path: {}", p.display()),
            ToolError::Parse(s) => write!(f, "parse error: {}", s),
            ToolError::EditFailed(s) => write!(f, "edit failed: {}", s),
            ToolError::TerminalFailed(s) => write!(f, "terminal command failed: {}", s),
            ToolError::SearchFailed(s) => write!(f, "search failed: {}", s),
            ToolError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ToolError::InvalidLineRange { start, end, total } => write!(
                f,
                "invalid line range: {}..={} (file has {} lines)",
                start, end, total
            ),
            ToolError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ToolError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ToolError {
    fn from(e: io::Error) -> Self {
        ToolError::Io(e)
    }
}

impl From<anyhow::Error> for ToolError {
    fn from(e: anyhow::Error) -> Self {
        ToolError::Other(e.to_string())
    }
}

/// Result type alias for tool operations
pub type ToolResult<T> = Result<T, ToolError>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ToolError::FileNotFound(PathBuf::from("/test/path"));
        assert_eq!(format!("{}", err), "file not found: /test/path");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test");
        let tool_err: ToolError = io_err.into();
        assert!(matches!(tool_err, ToolError::Io(_)));
    }
}
