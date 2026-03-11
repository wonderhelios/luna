use serde::{Deserialize, Serialize};

pub mod symbol;
pub mod text_range;

/// A position in a source file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub byte: usize,
}

impl Position {
    pub fn new(line: usize, column: usize, byte: usize) -> Self {
        Self { line, column, byte }
    }
}

/// A range of text in a source file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextRange {
    pub start: Position,
    pub end: Position,
}

impl TextRange {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    pub fn line_count(&self) -> usize {
        self.end.line.saturating_sub(self.start.line) + 1
    }

    /// Check if this range contains another range
    pub fn contains(&self, other: &TextRange) -> bool {
        self.start.byte <= other.start.byte && self.end.byte >= other.end.byte
    }

    /// Get the size of the range in bytes
    pub fn size(&self) -> usize {
        self.end.byte.saturating_sub(self.start.byte)
    }
}

impl From<tree_sitter::Range> for TextRange {
    fn from(range: tree_sitter::Range) -> Self {
        Self {
            start: Position {
                line: range.start_point.row,
                column: range.start_point.column,
                byte: range.start_byte,
            },
            end: Position {
                line: range.end_point.row,
                column: range.end_point.column,
                byte: range.end_byte,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub namespace: String,
    pub range: TextRange,
}
