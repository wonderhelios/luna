use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeChunk {
    pub path: String,
    #[serde(rename = "alias")]
    pub alias: usize,
    #[serde(rename = "snippet")]
    pub snippet: String,
    #[serde(rename = "start")]
    pub start_line: usize,
    #[serde(rename = "end")]
    pub end_line: usize,
}

impl CodeChunk {
    pub fn is_empty(&self) -> bool {
        self.snippet.trim().is_empty()
    }
}

impl fmt::Display for CodeChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}\n{}", self.alias, self.path, self.snippet)
    }
}
