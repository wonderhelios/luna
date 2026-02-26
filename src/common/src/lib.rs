//! Common Types for Luna
//!
//! This crate provides shared types and newtype wrappers for type safety.
//!
//! Design Principles:
//! - Use newtype pattern for type safety
//! - Implement common traits (FromStr, Display, Serialize, Deserialize)
//! - Provide conversion methods

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use uuid::Uuid;

// ============================================================================
// Session Types
// ============================================================================

/// Session ID wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new random session ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into inner string
    pub fn into_inner(self) -> Self {
        self
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for SessionId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err("Session ID cannot be empty".to_string())
        } else {
            Ok(Self(s.to_string()))
        }
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Tool Types
// ============================================================================

/// Tool name wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolName(String);

impl ToolName {
    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into inner string
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Check if this is a known tool
    pub fn is_known(&self) -> bool {
        matches!(
            self.as_str(),
            "read_file" | "list_dir" | "edit_file" | "run_terminal"
        )
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ToolName {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err("Tool name cannot be empty".to_string())
        } else {
            Ok(Self(s.to_string()))
        }
    }
}

impl From<String> for ToolName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ToolName {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for ToolName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Query Types
// ============================================================================

/// Search query wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Query(String);

impl Query {
    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into inner
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Check if query is empty
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    /// Get query length
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if this looks like a natural language query
    pub fn is_natural_language(&self) -> bool {
        self.0.chars().any(|c| c.is_alphabetic() && !c.is_ascii())
            || (self.0.contains(' ') && self.0.len() > 20)
    }

    /// Check if this looks like code
    pub fn is_code_like(&self) -> bool {
        self.0.contains('_')
            || self.0.chars().any(|c| c.is_uppercase())
            || self
                .0
                .matches(|c: char| c.is_ascii_punctuation() && c != ' ')
                .count()
                > 0
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Query {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

impl From<String> for Query {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Query {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for Query {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Path Types
// ============================================================================

/// Repository root path wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RepoRoot(PathBuf);

impl RepoRoot {
    /// Create a new repo root
    pub fn new(path: PathBuf) -> Self {
        Self(path)
    }

    /// Get the inner path
    pub fn as_path(&self) -> &PathBuf {
        &self.0
    }

    /// Convert into inner path
    pub fn into_path(self) -> PathBuf {
        self.0
    }

    /// Check if path exists
    pub fn exists(&self) -> bool {
        self.0.exists()
    }

    /// Check if path is a directory
    pub fn is_dir(&self) -> bool {
        self.0.is_dir()
    }
}

impl fmt::Display for RepoRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl From<PathBuf> for RepoRoot {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}

impl From<&str> for RepoRoot {
    fn from(s: &str) -> Self {
        Self(PathBuf::from(s))
    }
}

impl From<String> for RepoRoot {
    fn from(s: String) -> Self {
        Self(PathBuf::from(s))
    }
}

impl AsRef<PathBuf> for RepoRoot {
    fn as_ref(&self) -> &PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for RepoRoot {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

// ============================================================================
// Confirmation ID Types
// ============================================================================

/// Confirmation ID wrapper for tool execution confirmation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConfirmationId(String);

impl ConfirmationId {
    /// Create a new random confirmation ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into inner string
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Default for ConfirmationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConfirmationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ConfirmationId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Err("Confirmation ID cannot be empty".to_string())
        } else {
            Ok(Self(s.to_string()))
        }
    }
}

impl From<String> for ConfirmationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ConfirmationId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for ConfirmationId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id() {
        let id = SessionId::new();
        assert!(!id.as_str().is_empty());

        let id2 = SessionId::from("test-session".to_string());
        assert_eq!(id2.as_str(), "test-session");
    }

    #[test]
    fn test_tool_name() {
        let name = ToolName::from("read_file");
        assert_eq!(name.as_str(), "read_file");
        assert!(name.is_known());

        let unknown = ToolName::from("unknown_tool");
        assert!(!unknown.is_known());
    }

    #[test]
    fn test_query() {
        let query = Query::from("test query");
        assert_eq!(query.as_str(), "test query");
        assert!(!query.is_empty());

        let nl_query = Query::from("how do I use this function?");
        assert!(nl_query.is_natural_language());
    }

    #[test]
    fn test_repo_root() {
        let root = RepoRoot::from("/test/path");
        assert_eq!(root.as_path(), &PathBuf::from("/test/path"));
    }

    #[test]
    fn test_confirmation_id() {
        let id = ConfirmationId::new();
        assert!(!id.as_str().is_empty());

        let id2 = ConfirmationId::from("test-confirmation".to_string());
        assert_eq!(id2.as_str(), "test-confirmation");
    }
}
