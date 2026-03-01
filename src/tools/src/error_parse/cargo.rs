//! Cargo build/test error parser
//!
//! Parses rustc/cargo output into structured ErrorRecords.
//! Supports both human-readable and `--message-format=short` output.

use super::{BuildErrorParser, ErrorKind, ErrorRecord, Location};
use regex::Regex;

/// Parser for cargo build/test errors
pub struct CargoErrorParser {
    /// Regex for standard rustc error format: error[E####]: message
    error_code_re: Regex,
    /// Regex for file location: --> file:line:col
    location_re: Regex,
    /// Regex for short format: file:line:col: error: message
    short_format_re: Regex,
    /// Regex for error without code: error: message
    simple_error_re: Regex,
    /// Regex for warning treated as error
    warning_error_re: Regex,
}

impl Default for CargoErrorParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CargoErrorParser {
    /// Create a new Cargo error parser with compiled regexes
    pub fn new() -> Self {
        Self {
            // Match: error[E0425]: cannot find value `x` in this scope
            error_code_re: Regex::new(
                r"^error\[(E\d{4})\]\s*:\s*(.+)$"
            ).expect("valid regex"),

            // Match: --> src/main.rs:42:10
            location_re: Regex::new(
                r"^\s*-->\s+([^:]+):(\d+):(\d+)"
            ).expect("valid regex"),

            // Match short format: src/main.rs:42:10: error: message
            short_format_re: Regex::new(
                r"^([^:]+):(\d+):(\d+):\s*error(?:\[(E\d{4})\])?\s*:\s*(.+)$"
            ).expect("valid regex"),

            // Match simple error without code: error: message
            simple_error_re: Regex::new(
                r"^error\s*:\s*(.+)$"
            ).expect("valid regex"),

            // Match warning treated as error
            warning_error_re: Regex::new(
                r"^error:\s*(.+?)\s+\[-W(.+?)\]$"
            ).expect("valid regex"),
        }
    }

    /// Parse cargo/rustc output
    pub fn parse_cargo_output(&self, output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
        let mut errors = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];

            // Try short format first: file:line:col: error[CODE]: message
            if let Some(caps) = self.short_format_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let col_num = caps.get(3)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let code = caps.get(4).map(|m| m.as_str().to_string());
                let message = caps.get(5).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());

                errors.push(ErrorRecord {
                    command: "cargo".to_string(),
                    exit_code,
                    kind: ErrorKind::CompileError,
                    message,
                    error_code: code,
                    locations: vec![Location::new(&path, line_num).with_column(col_num)],
                    suggestion: self.extract_suggestion(&lines, i),
                    raw: line.to_string(),
                });
                i += 1;
                continue;
            }

            // Try error with code format: error[E####]: message
            if let Some(caps) = self.error_code_re.captures(line) {
                let code = caps.get(1).map(|m| m.as_str().to_string());
                let message = caps.get(2).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());

                // Look ahead for location
                let (location, consumed) = self.find_location(&lines, i + 1);

                errors.push(ErrorRecord {
                    command: "cargo".to_string(),
                    exit_code,
                    kind: ErrorKind::CompileError,
                    message,
                    error_code: code,
                    locations: location.map(|l| vec![l]).unwrap_or_default(),
                    suggestion: self.extract_suggestion(&lines, i + consumed),
                    raw: line.to_string(),
                });
                i += 1 + consumed;
                continue;
            }

            // Try simple error without code: error: message
            if let Some(caps) = self.simple_error_re.captures(line) {
                // Skip if this is a note or help line
                if line.trim_start().starts_with("error: ") {
                    let message = caps.get(1).map(|m| m.as_str().trim().to_string())
                        .unwrap_or_else(|| "unknown error".to_string());

                    // Check if it's a warning-as-error
                    let kind = if self.warning_error_re.is_match(line) {
                        ErrorKind::LintError
                    } else {
                        ErrorKind::CompileError
                    };

                    // Look ahead for location
                    let (location, consumed) = self.find_location(&lines, i + 1);

                    errors.push(ErrorRecord {
                        command: "cargo".to_string(),
                        exit_code,
                        kind,
                        message,
                        error_code: None,
                        locations: location.map(|l| vec![l]).unwrap_or_default(),
                        suggestion: self.extract_suggestion(&lines, i + consumed),
                        raw: line.to_string(),
                    });
                    i += 1 + consumed;
                    continue;
                }
            }

            i += 1;
        }

        errors
    }

    /// Find location information after an error line
    fn find_location(&self, lines: &[&str], start_idx: usize) -> (Option<Location>, usize) {
        let mut consumed = 0;

        for (idx, line) in lines.iter().enumerate().skip(start_idx).take(5) {
            if let Some(caps) = self.location_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let col_num = caps.get(3)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);

                // Try to get code snippet from next line
                let code_snippet = lines.get(idx + 1).map(|s| s.trim().to_string());

                let mut loc = Location::new(&path, line_num)
                    .with_column(col_num)
                    .set_raw(format!("{}:{}:{}", path, line_num, col_num));

                if let Some(snippet) = code_snippet {
                    loc = loc.with_snippet(snippet);
                }

                return (Some(loc), consumed + 1);
            }
            consumed += 1;

            // Stop if we see another error or empty line
            if line.starts_with("error") || line.trim().is_empty() && consumed > 2 {
                break;
            }
        }

        (None, consumed)
    }

    /// Extract suggestion/help text after an error
    fn extract_suggestion(&self, lines: &[&str], start_idx: usize) -> Option<String> {
        let mut suggestion = String::new();
        let mut in_suggestion = false;

        for line in lines.iter().skip(start_idx).take(10) {
            let trimmed = line.trim();

            // Start of suggestion/help
            if trimmed.starts_with("help:") || trimmed.starts_with("= help:") {
                in_suggestion = true;
                suggestion.push_str(&trimmed[trimmed.find(':').unwrap_or(0) + 1..].trim());
                suggestion.push(' ');
            } else if trimmed.starts_with("suggestion:") {
                in_suggestion = true;
                suggestion.push_str("Suggestion: ");
            } else if in_suggestion {
                // Continue collecting suggestion lines
                if trimmed.starts_with("|") || trimmed.starts_with("help:") {
                    suggestion.push_str(trimmed);
                    suggestion.push(' ');
                } else if trimmed.is_empty() || trimmed.starts_with("error[") {
                    // End of suggestion
                    break;
                }
            }

            // Limit suggestion length
            if suggestion.len() > 500 {
                suggestion.truncate(497);
                suggestion.push_str("...");
                break;
            }
        }

        let result = suggestion.trim().to_string();
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}

impl BuildErrorParser for CargoErrorParser {
    fn parse(&self, output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
        self.parse_cargo_output(output, exit_code)
    }

    fn supports(&self, command: &str) -> bool {
        let cmd = command.to_lowercase();
        cmd.starts_with("cargo") || cmd.contains("rustc")
    }

    fn description(&self) -> &str {
        "Parser for cargo build and cargo test output"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_with_code() {
        let parser = CargoErrorParser::new();
        let output = r#"error[E0425]: cannot find value `x` in this scope
 --> src/main.rs:10:5
  |
10 |     println!("{}", x);
  |                    ^ not found in this scope
  |
help: consider importing this function
  |
1 | use std::io::Read;
  |"#;

        let errors = parser.parse(output, Some(101));
        assert_eq!(errors.len(), 1);

        let err = &errors[0];
        assert_eq!(err.error_code, Some("E0425".to_string()));
        assert_eq!(err.message, "cannot find value `x` in this scope");
        assert_eq!(err.kind, ErrorKind::CompileError);
        assert_eq!(err.locations.len(), 1);
        assert_eq!(err.locations[0].path, "src/main.rs");
        assert_eq!(err.locations[0].line, 10);
        assert_eq!(err.locations[0].column, 5);
        assert!(err.suggestion.is_some());
    }

    #[test]
    fn test_parse_short_format() {
        let parser = CargoErrorParser::new();
        let output = "src/lib.rs:42:10: error[E0308]: mismatched types";

        let errors = parser.parse(output, Some(101));
        assert_eq!(errors.len(), 1);

        let err = &errors[0];
        assert_eq!(err.error_code, Some("E0308".to_string()));
        assert_eq!(err.message, "mismatched types");
        assert_eq!(err.locations[0].path, "src/lib.rs");
        assert_eq!(err.locations[0].line, 42);
        assert_eq!(err.locations[0].column, 10);
    }

    #[test]
    fn test_parse_simple_error() {
        let parser = CargoErrorParser::new();
        let output = r#"error: expected `;`, found `let`
 --> src/main.rs:5:12
5 |     let x = 1
  |            ^ expected `;` here"#;

        let errors = parser.parse(output, Some(101));
        assert!(!errors.is_empty());
        assert_eq!(errors[0].message, "expected `;`, found `let`");
        assert_eq!(errors[0].error_code, None);
    }

    #[test]
    fn test_supports() {
        let parser = CargoErrorParser::new();
        assert!(parser.supports("cargo build"));
        assert!(parser.supports("cargo test"));
        assert!(parser.supports("cargo check"));
        assert!(parser.supports("rustc main.rs"));
        assert!(!parser.supports("npm test"));
        assert!(!parser.supports("pytest"));
    }

    #[test]
    fn test_multiple_errors() {
        let parser = CargoErrorParser::new();
        let output = r#"error[E0425]: cannot find value `x` in this scope
 --> src/main.rs:10:5
10 |     println!("{}", x);
  |                    ^

error[E0425]: cannot find value `y` in this scope
 --> src/main.rs:20:5
20 |     println!("{}", y);
  |                    ^"#;

        let errors = parser.parse(output, Some(101));
        assert_eq!(errors.len(), 2);

        assert_eq!(errors[0].message, "cannot find value `x` in this scope");
        assert_eq!(errors[0].locations[0].line, 10);

        assert_eq!(errors[1].message, "cannot find value `y` in this scope");
        assert_eq!(errors[1].locations[0].line, 20);
    }

    #[test]
    fn test_parse_real_cargo_output() {
        let parser = CargoErrorParser::new();
        // This is extracted from real cargo build output, containing various whitespace variations
        let output = r#"error[E0425]: cannot find value `x` in this scope
 --> src/main.rs:5:32
  |
5 |     let result = calculate_sum(x, y);
  |                                ^ not found in this scope

error[E0308]: mismatched types
 --> src/main.rs:9:23
  |
9 |     let number: i32 = "42";
  |                 ---   ^^^^ expected `i32`, found `&str`

error[E0308]: mismatched types
  --> src/main.rs:18:37
   |
18 | fn calculate_sum(a: i32, b: i32) -> i32 {
   |    -------------                    ^^^ expected `i32`, found `()`"#;

        let errors = parser.parse(output, Some(101));
        assert_eq!(errors.len(), 3, "Should parse 3 errors");

        // First error
        assert_eq!(errors[0].error_code, Some("E0425".to_string()));
        assert_eq!(errors[0].locations.len(), 1);
        assert_eq!(errors[0].locations[0].path, "src/main.rs");
        assert_eq!(errors[0].locations[0].line, 5);

        // Second error - there may be an issue here
        assert_eq!(errors[1].error_code, Some("E0308".to_string()));
        assert_eq!(errors[1].locations.len(), 1, "E0308 at line 9 should have location");
        assert_eq!(errors[1].locations[0].path, "src/main.rs");
        assert_eq!(errors[1].locations[0].line, 9);

        // Third error - note that ` -->` has two spaces here
        assert_eq!(errors[2].error_code, Some("E0308".to_string()));
        assert_eq!(errors[2].locations.len(), 1, "E0308 at line 18 should have location");
        assert_eq!(errors[2].locations[0].path, "src/main.rs");
        assert_eq!(errors[2].locations[0].line, 18);
    }
}
