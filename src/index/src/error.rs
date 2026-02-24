//! Error types for the index module

use intelligence::TreeSitterFileError;
use std::fmt;

/// Errors that can occur during chunking operations
#[derive(Debug)]
pub enum ChunkError {
    /// Failed to parse the file
    Parse(TreeSitterFileError),

    /// I/O error
    Io(std::io::Error),

    /// Other error
    Other(String),
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkError::Parse(e) => write!(f, "failed to parse file: {:?}", e),
            ChunkError::Io(e) => write!(f, "I/O error: {}", e),
            ChunkError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for ChunkError {}

impl From<std::io::Error> for ChunkError {
    fn from(e: std::io::Error) -> Self {
        ChunkError::Io(e)
    }
}

impl From<TreeSitterFileError> for ChunkError {
    fn from(e: TreeSitterFileError) -> Self {
        ChunkError::Parse(e)
    }
}
