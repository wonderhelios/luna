/// Intent classification and entity extraction.
///
/// Phase-1 implementation is rule-based and intentionally conservative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// User is asking for symbol navigation (where is it defined, go-to-definition, etc.).
    SymbolNavigation,
    /// User is asking for explanation/meaning/usage of one or more symbols.
    ExplainSymbol,
    /// Anything else.
    Other,
}

#[must_use]
pub fn classify_intent(input: &str) -> Intent {
    // Prefer explanation queries over pure navigation queries.
    if is_explain_symbol_query(input) {
        return Intent::ExplainSymbol;
    }
    if is_symbol_navigation_query(input) {
        return Intent::SymbolNavigation;
    }
    Intent::Other
}

#[must_use]
pub fn is_symbol_navigation_query(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    input.contains("定义")
        || input.contains("哪里")
        || input.contains("在哪")
        || input.contains("在哪里")
        || contains_file_position(input)
        || lower.contains("definition")
        || lower.contains("defined")
        || lower.contains("goto")
        || lower.contains("go to")
}

#[must_use]
pub fn is_explain_symbol_query(input: &str) -> bool {
    // Heuristic: explanation-like phrasing AND presence of at least one identifier.
    let has_ident = extract_best_identifier(input).is_some();
    if !has_ident {
        return false;
    }

    let lower = input.to_ascii_lowercase();
    input.contains("含义")
        || input.contains("是什么")
        || input.contains("啥")
        || input.contains("作用")
        || input.contains("解释")
        || input.contains("怎么用")
        || input.contains("如何用")
        || lower.contains("meaning")
        || lower.contains("what is")
        || lower.contains("explain")
}

/// Extract identifiers from free-form user input.
///
/// This targets ASCII identifiers only: `[A-Za-z_][A-Za-z0-9_]*`.
pub fn extract_identifiers<'a>(input: &'a str) -> Vec<&'a str> {
    let bytes = input.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if !is_ident_start(bytes[i]) {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() && is_ident_continue(bytes[i]) {
            i += 1;
        }

        // safe: identifiers are ASCII, slicing at byte boundaries is valid UTF-8.
        out.push(&input[start..i]);
    }

    out
}

/// Extract identifiers and de-duplicate them while preserving the first-seen order.
pub fn extract_identifiers_dedup<'a>(input: &'a str) -> Vec<&'a str> {
    use std::collections::HashSet;

    let tokens = extract_identifiers(input);
    let mut seen = HashSet::<&'a str>::new();
    let mut out = Vec::new();
    for t in tokens {
        if seen.insert(t) {
            out.push(t);
        }
    }
    out
}

/// Heuristic: prefer snake_case tokens (often functions), otherwise the longest token.
#[must_use]
pub fn extract_best_identifier<'a>(input: &'a str) -> Option<&'a str> {
    let tokens = extract_identifiers(input);
    if let Some(t) = tokens.iter().copied().find(|t| t.contains('_')) {
        return Some(t);
    }
    tokens.into_iter().max_by_key(|t| t.len())
}

/// Try to extract a `<path>:<line>[:<col>]` token from input.
///
/// Returns `(path, line_0_based, col_0_based)`.
pub fn extract_file_position(input: &str) -> Option<(std::path::PathBuf, usize, usize)> {
    for raw in input.split_whitespace() {
        let token = raw.trim_matches(|c: char| matches!(c, '(' | ')' | ',' | ';' | '"' | '\''));
        if token.is_empty() {
            continue;
        }
        if let Some(v) = parse_file_position_token(token) {
            return Some(v);
        }
    }
    None
}

fn contains_file_position(input: &str) -> bool {
    extract_file_position(input).is_some()
}

fn parse_file_position_token(token: &str) -> Option<(std::path::PathBuf, usize, usize)> {
    // Support:
    // - path:line:col
    // - path:line
    let mut parts = token.rsplitn(3, ':');
    let last = parts.next()?;
    let mid = parts.next()?;
    let rest = parts.next();

    let (path_str, line_str, col_str) = match rest {
        Some(path_str) => (path_str, mid, Some(last)),
        None => (mid, last, None),
    };

    let line_1 = line_str.parse::<usize>().ok()?;
    let col_1 = col_str.and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    if line_1 == 0 || col_1 == 0 {
        return None;
    }

    Some((
        std::path::PathBuf::from(path_str),
        line_1.saturating_sub(1),
        col_1.saturating_sub(1),
    ))
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || (b'a'..=b'z').contains(&b) || (b'A'..=b'Z').contains(&b)
}

fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || (b'0'..=b'9').contains(&b)
}
