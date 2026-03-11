use crate::core_local::text_range::TextRange;

/// A snippet of text around a highlight range.
#[derive(Debug, Clone)]
pub struct Snippet {
    /// 0-based line index.
    pub start_line: usize,
    /// 0-based line index.
    pub end_line: usize,
    /// Rendered text (may include line numbers / highlight markers).
    pub text: String,
}

/// Build snippets with best-effort highlighting and optional line numbers.
#[derive(Debug, Clone, Copy)]
pub struct SnippetBuilder {
    pub context_lines: usize,
    pub with_line_numbers: bool,
    pub with_highlight: bool,
    /// Marker used to wrap highlighted text.
    pub highlight_marker: (&'static str, &'static str),
}

impl Default for SnippetBuilder {
    fn default() -> Self {
        Self {
            context_lines: 3,
            with_line_numbers: true,
            with_highlight: true,
            highlight_marker: ("§", "§"),
        }
    }
}

impl SnippetBuilder {
    /// Expand a `TextRange` into a rendered snippet.
    ///
    /// `line_end_indices` should contain the byte index of each `\n`.
    pub fn build(&self, content: &str, line_end_indices: &[usize], range: TextRange) -> Snippet {
        let total_lines = line_end_indices.len().saturating_add(1);
        if total_lines == 0 {
            return Snippet {
                start_line: 0,
                end_line: 0,
                text: String::new(),
            };
        }

        let focus_line = range.start.line.min(total_lines.saturating_sub(1));
        let start_line = focus_line.saturating_sub(self.context_lines);
        let end_line = (focus_line + self.context_lines).min(total_lines.saturating_sub(1));

        let mut rendered = String::new();
        for line in start_line..=end_line {
            if line != start_line {
                rendered.push('\n');
            }
            let (line_start, line_end) = line_bounds(content.len(), line_end_indices, line);
            let mut line_str = &content[line_start..line_end];
            // strip trailing '\r' for Windows files
            if let Some(stripped) = line_str.strip_suffix('\r') {
                line_str = stripped;
            }

            let mut body = if self.with_highlight {
                highlight_line(
                    content,
                    line_start,
                    line_end,
                    line_str,
                    range,
                    self.highlight_marker,
                )
            } else {
                line_str.to_owned()
            };

            if self.with_line_numbers {
                // 1-based line numbers, align to 4 columns.
                body = format!("{:>4} {body}", line + 1);
            }
            rendered.push_str(&body);
        }

        Snippet {
            start_line,
            end_line,
            text: rendered,
        }
    }
}

fn line_bounds(content_len: usize, line_end_indices: &[usize], line: usize) -> (usize, usize) {
    let start = if line == 0 {
        0
    } else {
        line_end_indices
            .get(line.saturating_sub(1))
            .copied()
            .unwrap_or(0)
            .saturating_add(1)
    };

    let end = line_end_indices.get(line).copied().unwrap_or(content_len);

    (start.min(content_len), end.min(content_len))
}

fn highlight_line(
    content: &str,
    line_start: usize,
    line_end: usize,
    line_str: &str,
    range: TextRange,
    marker: (&str, &str),
) -> String {
    let hl_start = range.start.byte.max(line_start);
    let hl_end = range.end.byte.min(line_end);
    if hl_start >= hl_end {
        return line_str.to_owned();
    }

    let rel_start = hl_start.saturating_sub(line_start);
    let rel_end = hl_end.saturating_sub(line_start);

    // Safety: tree-sitter ranges should align to UTF-8 boundaries.
    let before = &content[line_start..line_start + rel_start];
    let mid = &content[line_start + rel_start..line_start + rel_end];
    let after = &content[line_start + rel_end..line_end];
    format!("{before}{}{mid}{}{after}", marker.0, marker.1)
}
