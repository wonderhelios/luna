//! Luna Error Types
//!
//! This crate provides unified error types for all Luna components.

/// Unified error type for Luna
#[derive(thiserror::Error, Debug)]
pub enum LunaError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {message}")]
    Parse { message: String },

    #[error("Tool error: {message}")]
    Tool { message: String },

    #[error("LLM error: {message}")]
    Llm { message: String },

    #[error("Search error: {message}")]
    Search { message: String },

    #[error("Config error: {message}")]
    Config { message: String },

    #[error("Session error: {message}")]
    Session { message: String },

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("Permission denied: {message}")]
    Permission { message: String },

    #[error("Timeout: {message}")]
    Timeout { message: String },
}

impl LunaError {
    pub fn parse<S: Into<String>>(message: S) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    pub fn tool<S: Into<String>>(message: S) -> Self {
        Self::Tool {
            message: message.into(),
        }
    }

    pub fn llm<S: Into<String>>(message: S) -> Self {
        Self::Llm {
            message: message.into(),
        }
    }

    pub fn search<S: Into<String>>(message: S) -> Self {
        Self::Search {
            message: message.into(),
        }
    }

    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    pub fn session<S: Into<String>>(message: S) -> Self {
        Self::Session {
            message: message.into(),
        }
    }

    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    pub fn not_found<S: Into<String>>(resource: S) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    pub fn permission<S: Into<String>>(message: S) -> Self {
        Self::Permission {
            message: message.into(),
        }
    }

    pub fn timeout<S: Into<String>>(message: S) -> Self {
        Self::Timeout {
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, LunaError>;

impl From<serde_json::Error> for LunaError {
    fn from(err: serde_json::Error) -> Self {
        Self::parse(format!("JSON error: {}", err))
    }
}

impl From<toml::de::Error> for LunaError {
    fn from(err: toml::de::Error) -> Self {
        Self::parse(format!("TOML parse error: {}", err))
    }
}

impl From<toml::ser::Error> for LunaError {
    fn from(err: toml::ser::Error) -> Self {
        Self::parse(format!("TOML serialize error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_constructors() {
        let err = LunaError::parse("test");
        assert!(matches!(err, LunaError::Parse { .. }));

        let err = LunaError::tool("test");
        assert!(matches!(err, LunaError::Tool { .. }));
    }
}
