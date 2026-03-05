use intelligence::{SymbolContext, SymbolLocation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    /// Navigation-style output (Phase-1 MVP default).
    Navigation,
    /// Explanation-style output: show kind/doc comments (best-effort) in addition to snippets.
    Explain,
}

/// Render a successful symbol lookup result.
pub fn render_symbol_navigation_success(
    name: &str,
    primary: &SymbolLocation,
    context: Result<SymbolContext, anyhow::Error>,
    other_candidates: &[SymbolLocation],
    references: &[SymbolLocation],
) -> String {
    render_symbol_result(
        RenderStyle::Navigation,
        name,
        primary,
        context,
        other_candidates,
        references,
    )
}

pub fn render_symbol_explain_success(
    name: &str,
    primary: &SymbolLocation,
    context: Result<SymbolContext, anyhow::Error>,
    other_candidates: &[SymbolLocation],
    references: &[SymbolLocation],
) -> String {
    render_symbol_result(
        RenderStyle::Explain,
        name,
        primary,
        context,
        other_candidates,
        references,
    )
}

fn render_symbol_result(
    style: RenderStyle,
    name: &str,
    primary: &SymbolLocation,
    context: Result<SymbolContext, anyhow::Error>,
    other_candidates: &[SymbolLocation],
    references: &[SymbolLocation],
) -> String {
    let line_1_based = primary.range.start.line + 1;

    let mut out = String::new();
    out.push_str(&format!(
        "✅ Found in {}:{}\n",
        primary.rel_path.display(),
        line_1_based
    ));
    out.push_str(&format!(
        "{name} defined at {}:{}\n",
        primary.rel_path.display(),
        line_1_based
    ));

    match context {
        Ok(ctx) => {
            if let Some(sig) = ctx.signature_line.clone() {
                let kind = guess_symbol_kind(&sig);
                out.push_str(&format!("{} (approximate): {sig}\n", signature_label(kind)));

                if style == RenderStyle::Explain {
                    if let Some(kind) = kind {
                        out.push_str(&format!("Type determined: {kind}\n"));
                    }
                    let doc = extract_doc_comment(&ctx.snippet, &sig);
                    if !doc.is_empty() {
                        out.push_str("Documentation comments (excerpt):\n");
                        out.push_str(&doc);
                        out.push_str("\n");
                    }
                }
            }

            let snippet = if style == RenderStyle::Explain {
                trim_snippet_to_definition(&ctx.snippet, ctx.signature_line.as_deref())
            } else {
                ctx.snippet
            };

            let snippet = apply_highlight_markup(&snippet);

            out.push_str("\nCode snippet:\n");
            out.push_str(&snippet);
            out.push_str("\n");
        }
        Err(err) => {
            out.push_str(&format!("⚠️ Failed to get context: {err}\n"));
        }
    }

    if !other_candidates.is_empty() {
        out.push_str("\nOther candidate definitions:\n");
        for loc in other_candidates.iter().take(5) {
            out.push_str(&format!(
                "- {}:{}\n",
                loc.rel_path.display(),
                loc.range.start.line + 1
            ));
        }
    }

    if !references.is_empty() {
        out.push_str("\nReferenced by:\n");
        for r in references.iter().take(10) {
            out.push_str(&format!(
                "- {}:{}\n",
                r.rel_path.display(),
                r.range.start.line + 1
            ));
        }
    }

    out
}

fn guess_symbol_kind(sig: &str) -> Option<&'static str> {
    let first_line = sig.lines().next().unwrap_or(sig);
    let s = first_line.trim_start();
    if s.starts_with("pub struct ") || s.starts_with("struct ") {
        return Some("struct");
    }
    if s.starts_with("pub enum ") || s.starts_with("enum ") {
        return Some("enum");
    }
    if s.starts_with("pub trait ") || s.starts_with("trait ") {
        return Some("trait");
    }
    if s.starts_with("pub type ") || s.starts_with("type ") {
        return Some("type alias");
    }
    if is_function_like(s) {
        return Some("function");
    }
    if s.starts_with("pub mod ") || s.starts_with("mod ") {
        return Some("module");
    }
    None
}

fn is_function_like(line: &str) -> bool {
    let mut s = line.trim_start();

    // Strip visibility.
    if let Some(rest) = s.strip_prefix("pub ") {
        s = rest.trim_start();
    } else if let Some(idx) = s.find("pub(") {
        let _ = idx;
        if s.starts_with("pub(") {
            if let Some(rp) = s.find(')') {
                s = s[rp + 1..].trim_start();
            }
        }
    }

    // Strip common modifiers.
    for kw in ["async", "unsafe", "const", "extern"] {
        if let Some(rest) = s.strip_prefix(kw) {
            s = rest.trim_start();
        }
    }

    s.starts_with("fn ")
}

fn signature_label(kind: Option<&'static str>) -> &'static str {
    match kind {
        Some("function") => "Function signature",
        Some("struct") | Some("enum") | Some("trait") | Some("type alias") | Some("module") => {
            "Definition header"
        }
        _ => "Definition line",
    }
}

fn extract_doc_comment(snippet: &str, signature_line: &str) -> String {
    // Best-effort: find the signature (first line) in the snippet, then walk upwards to collect `///` lines.
    let signature_line = signature_line.lines().next().unwrap_or(signature_line);
    let lines: Vec<&str> = snippet.lines().collect();
    let sig_idx = lines
        .iter()
        .position(|l| normalize_line_for_match(l) == signature_line.trim())
        .or_else(|| {
            lines
                .iter()
                .position(|l| normalize_line_for_match(l).starts_with(signature_line.trim()))
        });

    let Some(sig_idx) = sig_idx else {
        return String::new();
    };

    let mut out = Vec::new();
    let mut i = sig_idx;
    while i > 0 {
        i -= 1;
        let l = lines[i].trim_start();
        if let Some(rest) = l.strip_prefix("///") {
            out.push(rest.trim_start());
            continue;
        }
        break;
    }

    out.reverse();
    out.join("\n")
}

fn trim_snippet_to_definition(snippet: &str, signature: Option<&str>) -> String {
    let Some(signature) = signature else {
        return snippet.to_owned();
    };
    let sig_first_line = signature.lines().next().unwrap_or(signature).trim();
    if sig_first_line.is_empty() {
        return snippet.to_owned();
    }

    let lines: Vec<&str> = snippet.lines().collect();
    let sig_idx = lines
        .iter()
        .position(|l| normalize_line_for_match(l) == sig_first_line)
        .or_else(|| {
            lines
                .iter()
                .position(|l| normalize_line_for_match(l).starts_with(sig_first_line))
        });

    let Some(sig_idx) = sig_idx else {
        return snippet.to_owned();
    };

    // Walk upwards to include contiguous `///` doc lines or `#[..]` attributes.
    let mut start = sig_idx;
    while start > 0 {
        let prev = strip_line_no_prefix(lines[start - 1]).trim_start();
        if prev.starts_with("///") || prev.starts_with("#[") {
            start -= 1;
            continue;
        }
        break;
    }

    lines[start..].join("\n")
}

fn strip_line_no_prefix(line: &str) -> &str {
    // Matches the formatting from SnippetBuilder: "{:>4} <code>".
    // We also accept arbitrary leading spaces + digits + space.
    let mut s = line;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start_digits = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > start_digits {
        // consume one separating space if present
        if i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        s = &s[i..];
    }
    s
}

fn normalize_line_for_match(line: &str) -> String {
    // 1) strip line number prefix
    // 2) remove highlight markers (e.g. `§`)
    // 3) trim
    let s = strip_line_no_prefix(line);
    s.chars()
        .filter(|&c| c != '§')
        .collect::<String>()
        .trim()
        .to_owned()
}

fn apply_highlight_markup(snippet: &str) -> String {
    // Intelligence emits lightweight highlight markers: `§...§`.
    // In terminal output we convert them to ANSI, or strip them if `NO_COLOR` is set.
    let no_color = std::env::var_os("NO_COLOR").is_some();
    if no_color {
        return snippet.replace('§', "");
    }

    // Reverse-video highlight is generally readable across themes.
    const HL_ON: &str = "\u{001b}[7m";
    const HL_OFF: &str = "\u{001b}[0m";

    let mut out = String::with_capacity(snippet.len());
    let mut highlighted = false;
    for ch in snippet.chars() {
        if ch == '§' {
            if highlighted {
                out.push_str(HL_OFF);
            } else {
                out.push_str(HL_ON);
            }
            highlighted = !highlighted;
            continue;
        }
        out.push(ch);
    }
    if highlighted {
        // Make sure we reset styles even on uneven markers.
        out.push_str(HL_OFF);
    }
    out
}

pub fn render_multi_header(found: &[&str]) -> String {
    let mut out = String::new();
    out.push_str("🤔 Thinking...\n");
    if found.is_empty() {
        out.push_str("[No identifier detected]\n");
    } else if found.len() == 1 {
        out.push_str("[Found identifier: ");
        out.push_str(found[0]);
        out.push_str("]\n");
    } else {
        out.push_str("[Found identifiers: ");
        out.push_str(&found.join(", "));
        out.push_str("]\n");
    }

    // Only show the "Searching" line when we are actually going to search.
    if !found.is_empty() {
        out.push_str("[Searching with ScopeGraph...]\n\n");
    } else {
        out.push_str("\n");
    }
    out
}

pub fn render_symbol_navigation_missing_identifier() -> String {
    "❌ No identifier detected.\n\nPlease ask in a format like: `Where is LunaRuntime defined?`".to_owned()
}

pub fn render_symbol_navigation_missing_repo_root(name: &str) -> String {
    let _ = name;
    "⚠️ Unable to resolve repo_root: needs to be run in a git repository directory (cwd should trace back to `.git`).".to_owned()
}

pub fn render_symbol_navigation_search_failed(name: &str, err: &anyhow::Error) -> String {
    let _ = name;
    format!("❌ Search failed: {err}")
}

pub fn render_symbol_navigation_not_found(name: &str) -> String {
    let _ = name;
    "❌ Definition not found (current MVP only scans top-level definitions).".to_owned()
}
