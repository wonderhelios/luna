//! Text range definitions

use serde::{Deserialize, Serialize};

/// Text position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub struct Position {
    /// Byte offset
    pub byte: usize,
    /// Line number
    pub line: usize,
    /// Column number
    pub column: usize,
}

impl Position {
    pub fn new(byte: usize, line: usize, column: usize) -> Self {
        Self { byte, line, column }
    }
}


/// Text range
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TextRange {
    pub start: Position,
    pub end: Position,
}

impl TextRange {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    /// Check if this range contains another range
    pub fn contains(&self, other: &TextRange) -> bool {
        self.start.byte <= other.start.byte && self.end.byte >= other.end.byte
    }

    /// Return the byte size of the range
    pub fn size(&self) -> usize {
        self.end.byte.saturating_sub(self.start.byte)
    }
}

impl From<tree_sitter::Range> for TextRange {
    fn from(range: tree_sitter::Range) -> Self {
        Self {
            start: Position {
                byte: range.start_byte,
                line: range.start_point.row,
                column: range.start_point.column,
            },
            end: Position {
                byte: range.end_byte,
                line: range.end_point.row,
                column: range.end_point.column,
            },
        }
    }
}
