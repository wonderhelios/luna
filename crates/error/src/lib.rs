//! Luna error handling
use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LunaError>;

#[derive(Error, Debug)]
pub enum LunaError {
    #[error("invalid input: {0}")] InvalidInput(String),
    #[error("not found: {0}")] NotFound(String),
    #[error("internal error: {0}")] Internal(String),
    #[error("io error{path}: {source}", path = path.as_ref().map(|p| format!(" ({})", p.display())).unwrap_or_default())]
    Io { path: Option<PathBuf>, #[source] source: std::io::Error },
    #[error("network error: {0}")] Network(String),
    #[error("timeout: {0}")] Timeout(String),
    #[error(transparent)] SerdeJson(#[from] serde_json::Error),
}

impl LunaError {
    pub fn invalid_input(msg: impl Into<String>) -> Self { Self::InvalidInput(msg.into()) }
    pub fn not_found(what: impl Into<String>) -> Self { Self::NotFound(what.into()) }
    pub fn internal(msg: impl Into<String>) -> Self { Self::Internal(msg.into()) }
    pub fn network(msg: impl Into<String>) -> Self { Self::Network(msg.into()) }
}

impl From<std::io::Error> for LunaError {
    fn from(err: std::io::Error) -> Self { Self::Io { path: None, source: err } }
}
