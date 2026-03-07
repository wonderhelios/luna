//! Luna error handling

use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LunaError>;

#[derive(Error, Debug)]
pub enum LunaError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error(
        "io error {path}: {source}",
        path = path
            .as_ref()
            .map(|p| format!(" ({})", p.display()))
            .unwrap_or_default()
    )]
    Io {
        path: Option<PathBuf>,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error("{msg}: {source}")]
    Context {
        msg: String,
        #[source]
        source: Box<LunaError>,
    },
}

impl LunaError {
    #[must_use]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    #[must_use]
    pub fn not_found(what: impl Into<String>) -> Self {
        Self::NotFound(what.into())
    }

    #[must_use]
    pub fn io(path: Option<PathBuf>, err: std::io::Error) -> Self {
        Self::Io { path, source: err }
    }

    #[must_use]
    pub fn context(self, msg: impl Into<String>) -> Self {
        Self::Context {
            msg: msg.into(),
            source: Box::new(self),
        }
    }
}

impl From<std::io::Error> for LunaError {
    fn from(err: std::io::Error) -> Self {
        Self::io(None, err)
    }
}

pub trait ResultExt<T> {
    fn context(self, msg: impl Into<String>) -> Result<T>;
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| e.context(msg))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| e.context(f()))
    }
}
