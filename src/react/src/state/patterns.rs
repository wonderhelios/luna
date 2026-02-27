//! Definition pattern matching for code analysis
//!
//! This module provides pattern-based definition extraction
//! without requiring a full AST parse.

use core::code_chunk::ContextChunk;

/// Definition patterns for extracting names from code snippets
///
/// Each pattern contains:
/// - prefixes: the keyword patterns to match (e.g., "pub fn ")
/// - skip_generic: whether to skip generic parameters after the name
struct DefPattern {
    prefixes: &'static [&'static str],
    skip_generic: bool,
}

static DEF_PATTERNS: &[DefPattern] = &[
    // Rust-style definitions
    DefPattern {
        prefixes: &["pub struct ", "struct "],
        skip_generic: true, // struct Foo<T> { ... }
    },
    DefPattern {
        prefixes: &["pub enum ", "enum "],
        skip_generic: true, // enum Foo<T> { ... }
    },
    DefPattern {
        prefixes: &["pub fn ", "fn ", "async fn ", "pub async fn "],
        skip_generic: true, // fn foo<T>() { ... }
    },
    DefPattern {
        prefixes: &["pub trait ", "trait "],
        skip_generic: true,
    },
    DefPattern {
        prefixes: &["pub type ", "type "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub const ", "const "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub static ", "static "],
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["pub impl ", "impl "],
        skip_generic: true, // impl<T> Foo<T> { ... }
    },
    // C-style definitions
    DefPattern {
        prefixes: &["class ", "public class ", "private class ", "protected class "],
        skip_generic: true,
    },
    DefPattern {
        prefixes: &["def "], // Python
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["function ", "export function "], // JavaScript/TypeScript
        skip_generic: false,
    },
    DefPattern {
        prefixes: &["func "], // Go
        skip_generic: false,
    },
];

/// Extract the name of a definition from a ContextChunk
///
/// Uses AST-aware pattern matching to identify definition names
/// without requiring a full parse.
pub fn extract_definition_name(chunk: &ContextChunk) -> Option<String> {
    let snippet = chunk.snippet.trim_start();

    for pattern in DEF_PATTERNS {
        for &prefix in pattern.prefixes {
            if let Some(after_prefix) = snippet.strip_prefix(prefix) {
                return Some(extract_identifier(after_prefix, pattern.skip_generic));
            }
        }
    }

    None
}

/// Extract an identifier from the start of a string
///
/// Handles:
/// - Generic parameters: `Foo<T, U>` extracts `Foo`
/// - Method receivers: `fn foo(&self)` extracts `foo`
/// - Qualified names: `impl Foo for Bar` extracts `Foo`
fn extract_identifier(s: &str, skip_generic: bool) -> String {
    let s = s.trim_start();

    // Find the end of the identifier
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            end = i + c.len_utf8();
        } else if c == '<' && skip_generic {
            // Found generic parameter start, stop here
            break;
        } else if c == '(' || c == '{' || c == ':' || c == ' ' || c == '<' {
            // End of identifier
            break;
        } else {
            // Skip other characters (like & for self)
            break;
        }
    }

    s[..end].to_string()
}

/// Check if a ContextChunk appears to contain a type/function definition
///
/// Uses the same pattern set as `extract_definition_name` for consistency.
pub fn is_definition_chunk(chunk: &ContextChunk) -> bool {
    let snippet = chunk.snippet.trim_start();

    DEF_PATTERNS.iter().any(|pattern| {
        pattern
            .prefixes
            .iter()
            .any(|&prefix| snippet.starts_with(prefix))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_definition_name_struct() {
        let chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "pub struct MyStruct { x: i32 }".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert_eq!(
            extract_definition_name(&chunk),
            Some("MyStruct".to_string())
        );
    }

    #[test]
    fn test_extract_definition_name_fn() {
        let chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "fn my_function() -> i32 { 42 }".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert_eq!(
            extract_definition_name(&chunk),
            Some("my_function".to_string())
        );
    }

    #[test]
    fn test_is_definition_chunk() {
        let def_chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "pub struct Test {}".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert!(is_definition_chunk(&def_chunk));

        let non_def_chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "let x = 42;".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        assert!(!is_definition_chunk(&non_def_chunk));
    }

    #[test]
    fn test_extract_generic() {
        let chunk = ContextChunk {
            path: "test.rs".to_string(),
            alias: 0,
            snippet: "struct Foo<T, U> { x: T, y: U }".to_string(),
            start_line: 0,
            end_line: 0,
            reason: String::new(),
        };
        // Should extract "Foo", not "Foo<T"
        assert_eq!(
            extract_definition_name(&chunk),
            Some("Foo".to_string())
        );
    }
}
