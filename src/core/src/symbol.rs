//! Symbol definitions

use serde::{Deserialize, Serialize};
use crate::text_range::TextRange;

/// Symbol
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub kind: String,
    pub range: TextRange,
}

impl Symbol {
    pub fn new(kind: impl Into<String>, range: TextRange) -> Self {
        Self {
            kind: kind.into(),
            range,
        }
    }
}
