//! Session Management for Luna
//!
//! This crate provides session persistence and management:
//! - SessionStore trait for pluggable backends
//! - MemorySessionStore for in-memory sessions
//! - FileSessionStore for disk-based persistence
//!
//! Design Goals:
//! - Support multiple storage backends
//! - Enable session persistence across restarts
//! - Provide thread-safe operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use toolkit::ExecutionPolicy;

// ============================================================================
// Session Types
// ============================================================================

/// Pending tool call awaiting confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingToolCall {
    /// Tool name
    pub name: String,
    /// Repository root path
    pub repo_root: PathBuf,
    /// Tool arguments
    pub arguments: serde_json::Value,
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Execution policy
    pub policy: ExecutionPolicy,
    /// Pending tool calls (confirmation_id -> call)
    pub pending: HashMap<String, PendingToolCall>,
    /// Session metadata
    pub metadata: SessionMetadata,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last activity time
    pub last_activity_at: chrono::DateTime<chrono::Utc>,
    /// Session title (optional)
    pub title: Option<String>,
}

impl Default for SessionMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            created_at: now,
            last_activity_at: now,
            title: None,
        }
    }
}

// ============================================================================
// Session Store Trait
// ============================================================================

/// Trait for session storage backends
pub trait SessionStore: Send + Sync {
    /// Get a session by ID
    fn get(&self, session_id: &str) -> Result<Option<SessionState>, SessionError>;

    /// Insert a new session
    fn insert(&self, session_id: String, state: SessionState) -> Result<(), SessionError>;

    /// Update an existing session
    fn update(&self, session_id: &str, state: SessionState) -> Result<(), SessionError>;

    /// Delete a session
    fn delete(&self, session_id: &str) -> Result<(), SessionError>;

    /// List all session IDs
    fn list(&self) -> Result<Vec<String>, SessionError>;

    /// Check if a session exists
    fn contains(&self, session_id: &str) -> Result<bool, SessionError>;
}

// ============================================================================
// Session Errors
// ============================================================================

/// Session-related errors
#[derive(thiserror::Error, Debug)]
pub enum SessionError {
    /// Session not found
    #[error("Session not found: {0}")]
    NotFound(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Storage backend error
    #[error("Storage error: {0}")]
    Storage(String),
}

// ============================================================================
// Memory Session Store
// ============================================================================

/// In-memory session store (default)
#[derive(Debug, Clone)]
pub struct MemorySessionStore {
    sessions: Arc<RwLock<HashMap<String, SessionState>>>,
}

impl MemorySessionStore {
    /// Create a new in-memory session store
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for MemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore for MemorySessionStore {
    fn get(&self, session_id: &str) -> Result<Option<SessionState>, SessionError> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.get(session_id).cloned())
    }

    fn insert(&self, session_id: String, state: SessionState) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id, state);
        Ok(())
    }

    fn update(&self, session_id: &str, state: SessionState) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.to_string(), state);
        Ok(())
    }

    fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
        Ok(())
    }

    fn list(&self) -> Result<Vec<String>, SessionError> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.keys().cloned().collect())
    }

    fn contains(&self, session_id: &str) -> Result<bool, SessionError> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.contains_key(session_id))
    }
}

// ============================================================================
// File Session Store
// ============================================================================

/// File-based session store with persistence
#[derive(Debug)]
pub struct FileSessionStore {
    base_dir: PathBuf,
    memory: MemorySessionStore,
}

impl FileSessionStore {
    /// Create a new file-based session store
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self, SessionError> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_dir)?;

        Ok(Self {
            base_dir,
            memory: MemorySessionStore::new(),
        })
    }

    /// Get the file path for a session
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.jsonl", session_id))
    }

    /// Load session from disk
    fn load_session(&self, session_id: &str) -> Result<Option<SessionState>, SessionError> {
        let path = self.session_path(session_id);

        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path)?;
        let state: SessionState = serde_json::from_str(&content).map_err(|e| {
            SessionError::Serialization(format!("Failed to deserialize session: {}", e))
        })?;

        Ok(Some(state))
    }

    /// Save session to disk
    fn save_session(&self, session_id: &str, state: &SessionState) -> Result<(), SessionError> {
        let path = self.session_path(session_id);
        let content = serde_json::to_string_pretty(state).map_err(|e| {
            SessionError::Serialization(format!("Failed to serialize session: {}", e))
        })?;

        std::fs::write(&path, &content)?;
        Ok(())
    }
}

impl SessionStore for FileSessionStore {
    fn get(&self, session_id: &str) -> Result<Option<SessionState>, SessionError> {
        // Try memory first
        if let Ok(Some(state)) = self.memory.get(session_id) {
            return Ok(Some(state));
        }

        // Load from disk
        self.load_session(session_id)
    }

    fn insert(&self, session_id: String, state: SessionState) -> Result<(), SessionError> {
        // Save to disk
        self.save_session(&session_id, &state)?;

        // Update memory cache
        self.memory.insert(session_id, state)
    }

    fn update(&self, session_id: &str, state: SessionState) -> Result<(), SessionError> {
        // Save to disk
        self.save_session(session_id, &state)?;

        // Update memory cache
        self.memory.update(session_id, state)
    }

    fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        // Delete from disk
        let path = self.session_path(session_id);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }

        // Delete from memory
        self.memory.delete(session_id)
    }

    fn list(&self) -> Result<Vec<String>, SessionError> {
        // List all JSONL files in base directory
        let mut session_ids = Vec::new();

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().starts_with("jsonl") {
                    if let Some(stem) = path.file_stem() {
                        if let Some(id) = stem.to_str() {
                            session_ids.push(id.to_string());
                        }
                    }
                }
            }
        }

        Ok(session_ids)
    }

    fn contains(&self, session_id: &str) -> Result<bool, SessionError> {
        // Check memory first
        if let Ok(contains) = self.memory.contains(session_id) {
            if contains {
                return Ok(true);
            }
        }

        // Check disk
        Ok(self.session_path(session_id).exists())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_memory_session_store() {
        let store = MemorySessionStore::new();

        let session_id = "test-session".to_string();
        let state = SessionState {
            policy: ExecutionPolicy::default(),
            pending: HashMap::new(),
            metadata: SessionMetadata::default(),
        };

        assert!(store.insert(session_id.clone(), state.clone()).is_ok());
        assert!(store.contains(&session_id).unwrap());

        let retrieved = store.get(&session_id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().policy.allow_edit_file,
            state.policy.allow_edit_file
        );
    }

    #[test]
    fn test_file_session_store() -> Result<(), SessionError> {
        let temp_dir = TempDir::new().map_err(|e| SessionError::Io(e))?;
        let store = FileSessionStore::new(temp_dir.path())?;

        let session_id = "test-session".to_string();
        let state = SessionState {
            policy: ExecutionPolicy::default(),
            pending: HashMap::new(),
            metadata: SessionMetadata::default(),
        };

        store.insert(session_id.clone(), state)?;
        assert!(store.contains(&session_id)?);

        let retrieved = store.get(&session_id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    #[test]
    fn test_session_metadata() {
        let metadata = SessionMetadata::default();
        assert!(metadata.title.is_none());
        assert!(metadata.created_at <= metadata.last_activity_at);
    }
}
