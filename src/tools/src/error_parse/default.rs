//! Default heuristic error parser
//!
//! Uses regex patterns to extract file:line:message from any tool output.
//! This is the fallback parser when no specific parser is available.

use super::{BuildErrorParser, ErrorKind, ErrorRecord, Location};
use regex::Regex;

/// Heuristic parser for generic error output
///
/// Uses regex to extract common patterns like:
/// - file:line: error: message
/// - file(line): error: message
/// - At file:line: message
pub struct DefaultErrorParser {
    /// Pattern: path/to/file.ext:123: error message
    file_line_col_re: Regex,
    /// Pattern: path/to/file.ext(123): error message
    paren_line_re: Regex,
    /// Pattern: At path/to/file.ext:123: message
    at_line_re: Regex,
    /// Pattern: path\to\file.ext(123): message (Windows style)
    windows_path_re: Regex,
    /// Generic error marker
    error_marker_re: Regex,
}

impl Default for DefaultErrorParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultErrorParser {
    /// Create a new default parser with compiled regexes
    pub fn new() -> Self {
        Self {
            // Match: src/main.rs:42:10: error: something
            //        src/main.rs:42: error: something
            file_line_col_re: Regex::new(
                r"^\s*([\w\-/\\.]+\.[a-zA-Z0-9]+):(\d+):(?:(\d+):)?\s*(error|warning)(?:\s*:)?\s*(.+)"
            ).expect("valid regex"),

            // Match: src/main.rs(42): error: something
            paren_line_re: Regex::new(
                r"^\s*([\w\-/\\.]+\.[a-zA-Z0-9]+)\((\d+)\)\s*:?\s*(error|warning)?(?:\s*:)?\s*(.+)"
            ).expect("valid regex"),

            // Match: At src/main.rs:42: something
            at_line_re: Regex::new(
                r"(?i)^\s*at\s+([\w\-/\\.]+\.[a-zA-Z0-9]+):(\d+)(?::\d+)?\s*:?\s*(.+)"
            ).expect("valid regex"),

            // Windows: src\main.rs(42): error C1234: message
            windows_path_re: Regex::new(
                r"^\s*([\w\\-\\.]+\.[a-zA-Z0-9]+)\((\d+)\)\s*:?\s*(error|warning)\s+(\w+)\s*:?\s*(.+)"
            ).expect("valid regex"),

            // Generic error marker line
            error_marker_re: Regex::new(
                r"(?i)^\s*(?:error|fail|failed|failure)[\s:]*(.+)"
            ).expect("valid regex"),
        }
    }

    /// Parse any output using heuristics
    pub fn parse_generic_output(&self, output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
        let mut errors = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];

            // Try standard format: file:line:col: error: message
            if let Some(caps) = self.file_line_col_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let col_num = caps.get(3)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(0);
                let severity = caps.get(4).map(|m| m.as_str().to_lowercase());
                let message = caps.get(5).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());

                let kind = match severity.as_deref() {
                    Some("warning") => ErrorKind::LintError,
                    _ => ErrorKind::Unknown,
                };

                errors.push(ErrorRecord {
                    command: "unknown".to_string(),
                    exit_code,
                    kind,
                    message,
                    error_code: None,
                    locations: vec![Location::new(&path, line_num).with_column(col_num)],
                    suggestion: None,
                    raw: line.to_string(),
                });
                i += 1;
                continue;
            }

            // Try paren format: file(123): error: message
            if let Some(caps) = self.paren_line_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let severity = caps.get(3).map(|m| m.as_str().to_lowercase());
                let message = caps.get(4).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());

                let kind = match severity.as_deref() {
                    Some("warning") => ErrorKind::LintError,
                    _ => ErrorKind::Unknown,
                };

                errors.push(ErrorRecord {
                    command: "unknown".to_string(),
                    exit_code,
                    kind,
                    message,
                    error_code: None,
                    locations: vec![Location::new(&path, line_num)],
                    suggestion: None,
                    raw: line.to_string(),
                });
                i += 1;
                continue;
            }

            // Try "At" format: At file:line: message
            if let Some(caps) = self.at_line_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let message = caps.get(3).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "error at this location".to_string());

                errors.push(ErrorRecord {
                    command: "unknown".to_string(),
                    exit_code,
                    kind: ErrorKind::Unknown,
                    message,
                    error_code: None,
                    locations: vec![Location::new(&path, line_num)],
                    suggestion: None,
                    raw: line.to_string(),
                });
                i += 1;
                continue;
            }

            // Try Windows format
            if let Some(caps) = self.windows_path_re.captures(line) {
                let path = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let line_num = caps.get(2)
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(1);
                let severity = caps.get(3).map(|m| m.as_str().to_lowercase());
                let code = caps.get(4).map(|m| m.as_str().to_string());
                let message = caps.get(5).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown error".to_string());

                let kind = match severity.as_deref() {
                    Some("warning") => ErrorKind::LintError,
                    _ => ErrorKind::CompileError,
                };

                errors.push(ErrorRecord {
                    command: "unknown".to_string(),
                    exit_code,
                    kind,
                    message,
                    error_code: code,
                    locations: vec![Location::new(&path, line_num)],
                    suggestion: None,
                    raw: line.to_string(),
                });
                i += 1;
                continue;
            }

            // Generic error marker - use previous location or "unknown"
            if let Some(caps) = self.error_marker_re.captures(line) {
                let message = caps.get(1).map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "error occurred".to_string());

                // Only create if we have a message and no recent error
                if !message.is_empty() {
                    errors.push(ErrorRecord {
                        command: "unknown".to_string(),
                        exit_code,
                        kind: ErrorKind::Unknown,
                        message,
                        error_code: None,
                        locations: vec![],
                        suggestion: None,
                        raw: line.to_string(),
                    });
                }
            }

            i += 1;
        }

        errors
    }
}

impl BuildErrorParser for DefaultErrorParser {
    fn parse(&self, output: &str, exit_code: Option<i32>) -> Vec<ErrorRecord> {
        self.parse_generic_output(output, exit_code)
    }

    fn supports(&self, _command: &str) -> bool {
        // Default parser supports everything as fallback
        true
    }

    fn description(&self) -> &str {
        "Heuristic parser for generic tool output using regex patterns"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_file_line_col() {
        let parser = DefaultErrorParser::new();
        let output = "src/main.rs:42:10: error: something went wrong";

        let errors = parser.parse(output, Some(1));
        assert_eq!(errors.len(), 1);

        let err = &errors[0];
        assert_eq!(err.message, "something went wrong");
        assert_eq!(err.locations[0].path, "src/main.rs");
        assert_eq!(err.locations[0].line, 42);
        assert_eq!(err.locations[0].column, 10);
    }

    #[test]
    fn test_parse_simple_file_line() {
        let parser = DefaultErrorParser::new();
        let output = "src/lib.rs:100: error: undefined variable";

        let errors = parser.parse(output, Some(1));
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].locations[0].line, 100);
        assert_eq!(errors[0].kind, ErrorKind::Unknown);
    }

    #[test]
    fn test_parse_paren_format() {
        let parser = DefaultErrorParser::new();
        let output = "src/main.js(42): error: unexpected token";

        let errors = parser.parse(output, Some(1));
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].locations[0].path, "src/main.js");
        assert_eq!(errors[0].locations[0].line, 42);
    }

    #[test]
    fn test_parse_at_format() {
        let parser = DefaultErrorParser::new();
        let output = "At src/parser.py:55: invalid syntax";

        let errors = parser.parse(output, Some(1));
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].locations[0].path, "src/parser.py");
        assert_eq!(errors[0].locations[0].line, 55);
        assert_eq!(errors[0].message, "invalid syntax");
    }

    #[test]
    fn test_parse_multiple_errors() {
        let parser = DefaultErrorParser::new();
        let output = r#"src/a.rs:10: error: first error
some context here
src/b.rs:20:5: error: second error"#;

        let errors = parser.parse(output, Some(1));
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].locations[0].path, "src/a.rs");
        assert_eq!(errors[1].locations[0].path, "src/b.rs");
    }

    #[test]
    fn test_supports_any_command() {
        let parser = DefaultErrorParser::new();
        assert!(parser.supports("anything"));
        assert!(parser.supports(""));
        assert!(parser.supports("random command"));
    }
}
