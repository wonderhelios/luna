use std::path::{Path, PathBuf};

use crate::{
    repo_scan::{FsRepoFileProvider, RepoFileProvider, RepoScanError, RepoScanOptions},
    TreeSitterFile, TreeSitterFileError,
};

use crate::{document::build_line_end_indices, snippet::SnippetBuilder};

use core::text_range::TextRange;

/// Controls how context snippets are extracted.
#[derive(Debug, Clone)]
pub struct SnippetOptions {
    /// Number of lines shown before/after the definition line.
    pub context_lines: usize,

    /// Whether to include line numbers in the snippet.
    pub with_line_numbers: bool,

    /// Whether to highlight the symbol range in the snippet.
    pub with_highlight: bool,
}

impl Default for SnippetOptions {
    fn default() -> Self {
        Self {
            context_lines: 5,
            with_line_numbers: true,
            with_highlight: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolLocation {
    pub rel_path: PathBuf,
    pub range: TextRange,
}

#[derive(Debug, Clone)]
pub struct SymbolContext {
    pub location: SymbolLocation,
    pub signature_line: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Clone, Default)]
pub struct SearchResult {
    pub definitions: Vec<SymbolLocation>,
    pub references: Vec<SymbolLocation>,
}

#[derive(Debug)]
pub enum NavigationError {
    RepoScan(RepoScanError),

    TreeSitter {
        rel_path: PathBuf,
        source: TreeSitterFileError,
    },

    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl From<RepoScanError> for NavigationError {
    fn from(value: RepoScanError) -> Self {
        Self::RepoScan(value)
    }
}

impl std::fmt::Display for NavigationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepoScan(err) => write!(f, "{err}"),
            Self::TreeSitter { rel_path, source } => {
                write!(f, "tree-sitter failed for {}: {source}", rel_path.display())
            }
            Self::Io { path, source } => {
                write!(f, "failed to read file {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for NavigationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RepoScan(err) => Some(err),
            Self::TreeSitter { source, .. } => Some(source),
            Self::Io { source, .. } => Some(source),
        }
    }
}

/// A pluggable navigation facade.
///
/// MVP supports identifier-based queries (symbol name). Future phases can extend this
/// to position-based queries and richer cross-file resolution.
pub trait Navigator {
    fn search_symbol(&self, repo_root: &Path, name: &str) -> Result<SearchResult, NavigationError>;

    fn goto_definition(
        &self,
        repo_root: &Path,
        name: &str,
    ) -> Result<Vec<SymbolLocation>, NavigationError>;

    fn get_symbol_context(
        &self,
        repo_root: &Path,
        location: &SymbolLocation,
        opt: &SnippetOptions,
    ) -> Result<SymbolContext, NavigationError>;

    /// Find references for a symbol name using a text-based fallback.
    ///
    /// This is intentionally conservative and language-agnostic.
    fn find_references(
        &self,
        repo_root: &Path,
        name: &str,
        max: usize,
    ) -> Result<Vec<SymbolLocation>, NavigationError>;

    /// Position-based go-to-definition within a single file.
    ///
    /// `line`/`column` are 0-based.
    fn goto_definition_at(
        &self,
        repo_root: &Path,
        rel_path: &Path,
        line: usize,
        column: usize,
    ) -> Result<Vec<SymbolLocation>, NavigationError>;
}

/// Default implementation based on Tree-sitter + ScopeGraph.
#[derive(Debug, Clone)]
pub struct TreeSitterNavigator<P: RepoFileProvider> {
    provider: P,
    scan_opt: RepoScanOptions,
}

impl Default for TreeSitterNavigator<FsRepoFileProvider> {
    fn default() -> Self {
        Self {
            provider: FsRepoFileProvider,
            scan_opt: RepoScanOptions::default(),
        }
    }
}

impl<P: RepoFileProvider> TreeSitterNavigator<P> {
    #[must_use]
    pub fn new(provider: P, scan_opt: RepoScanOptions) -> Self {
        Self { provider, scan_opt }
    }

    fn extract_signature_and_snippet(
        content: &str,
        range: &TextRange,
        opt: &SnippetOptions,
    ) -> (Option<String>, String) {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return (None, String::new());
        }

        let line = range.start.line.min(lines.len().saturating_sub(1));
        let signature_line = Self::extract_definition_signature(&lines, line);
        let line_end_indices = build_line_end_indices(content);
        let snippet = SnippetBuilder {
            context_lines: opt.context_lines,
            with_line_numbers: opt.with_line_numbers,
            with_highlight: opt.with_highlight,
            ..SnippetBuilder::default()
        }
        .build(content, &line_end_indices, *range)
        .text;

        (signature_line, snippet)
    }

    fn extract_definition_signature(lines: &[&str], line_idx: usize) -> Option<String> {
        let line = lines.get(line_idx)?.trim();
        if line.is_empty() {
            return None;
        }

        // If it's a function-like line, try to capture a multi-line signature up to `{` or `;`.
        if is_function_like_line(line) {
            let mut out = Vec::new();
            for l in &lines[line_idx..] {
                let t = l.trim_end();
                if t.is_empty() {
                    break;
                }
                out.push(t);
                if t.contains('{') || t.contains(';') {
                    break;
                }
            }
            if !out.is_empty() {
                let joined = out.join("\n");
                return Some(sanitize_function_signature(&joined));
            }
        }

        // Fallback: single-line definition header.
        Some(sanitize_definition_header(line))
    }

    fn lang_id_for_path(path: &Path) -> Option<&'static str> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => Some("rust"),
            Some("go") => Some("go"),
            Some("py") => Some("python"),
            Some("js") => Some("javascript"),
            Some("ts") => Some("typescript"),
            Some("tsx") => Some("tsx"),
            Some("java") => Some("java"),
            Some("c") => Some("c"),
            // NOTE: Language identifiers must match `language_ids` (case-insensitive).
            // Our C++ config uses `"C++"`, not `"cpp"`.
            Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("h") => Some("C++"),
            Some("rb") => Some("ruby"),
            Some("php") => Some("php"),
            Some("r") => Some("r"),
            _ => None,
        }
    }

    fn find_identifier_occurrences(content: &str, name: &str, max: usize) -> Vec<TextRange> {
        if name.is_empty() || max == 0 {
            return Vec::new();
        }

        let bytes = content.as_bytes();
        let needle = name.as_bytes();
        if needle
            .iter()
            .any(|b| !b.is_ascii_alphanumeric() && *b != b'_')
        {
            // Fallback only supports ASCII identifiers.
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut line: usize = 0;
        let mut col: usize = 0;
        let mut i: usize = 0;

        while i + needle.len() <= bytes.len() {
            let b = bytes[i];
            if b == b'\n' {
                line += 1;
                col = 0;
                i += 1;
                continue;
            }

            if bytes[i..].starts_with(needle) {
                let start_byte = i;
                let start_line = line;
                let start_col = col;
                let end_byte = i + needle.len();
                let end_line = line;
                let end_col = col + needle.len();

                let prev = if start_byte == 0 {
                    None
                } else {
                    Some(bytes[start_byte - 1])
                };
                let next = bytes.get(end_byte).copied();

                let prev_ok = prev.is_none_or(|c| !is_ident_continue(c));
                let next_ok = next.is_none_or(|c| !is_ident_continue(c));

                if prev_ok && next_ok {
                    out.push(TextRange {
                        start: core::text_range::Position {
                            byte: start_byte,
                            line: start_line,
                            column: start_col,
                        },
                        end: core::text_range::Position {
                            byte: end_byte,
                            line: end_line,
                            column: end_col,
                        },
                    });
                    if out.len() >= max {
                        break;
                    }
                }
            }

            // advance by one byte
            i += 1;
            col += 1;
        }

        out
    }
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_'
        || b.is_ascii_lowercase()
        || b.is_ascii_uppercase()
        || b.is_ascii_digit()
}

fn is_function_like_line(line: &str) -> bool {
    // Best-effort Rust-ish detection. We intentionally keep this lightweight.
    // Examples:
    // - fn foo() -> T {
    // - pub fn foo(
    // - pub async fn foo(
    // - async fn main() {
    let mut s = line.trim_start();

    // Strip visibility prefixes.
    if let Some(rest) = s.strip_prefix("pub ") {
        s = rest.trim_start();
    } else if let Some(_rest) = s.strip_prefix("pub(") {
        // pub(crate) / pub(super)
        if let Some(idx) = s.find(')') {
            s = s[idx + 1..].trim_start();
        }
    }

    // Strip common modifiers.
    for kw in ["async", "unsafe", "const", "extern"] {
        if let Some(rest) = s.strip_prefix(kw) {
            // ensure it's a keyword boundary
            let rest = rest.trim_start();
            s = rest;
        }
    }

    s.starts_with("fn ")
}

fn sanitize_function_signature(sig: &str) -> String {
    // Strip trailing function body start `{` and any following content.
    // Also strip trailing `;` for declaration-style signatures.
    let mut s = sig;
    if let Some(idx) = s.find('{') {
        s = &s[..idx];
    }
    let s = s.trim_end();
    let s = s.strip_suffix(';').unwrap_or(s).trim_end();
    s.to_owned()
}

fn sanitize_definition_header(line: &str) -> String {
    // For non-function definitions, keep the header readable by trimming the trailing body start
    // `{` or terminal `;`.
    let mut s = line.trim_end();
    if let Some(rest) = s.strip_suffix('{') {
        s = rest.trim_end();
    }
    if let Some(rest) = s.strip_suffix(';') {
        s = rest.trim_end();
    }
    s.to_owned()
}

impl<P: RepoFileProvider> Navigator for TreeSitterNavigator<P> {
    fn search_symbol(&self, repo_root: &Path, name: &str) -> Result<SearchResult, NavigationError> {
        let definitions = self.goto_definition(repo_root, name)?;
        Ok(SearchResult {
            definitions,
            references: Vec::new(),
        })
    }

    fn goto_definition(
        &self,
        repo_root: &Path,
        name: &str,
    ) -> Result<Vec<SymbolLocation>, NavigationError> {
        let files = self.provider.list_files(repo_root, &self.scan_opt)?;
        let mut out = Vec::new();

        for file in files {
            let src = file.content.as_bytes();
            let Some(lang_id) = Self::lang_id_for_path(&file.rel_path) else {
                continue;
            };
            let ts = match TreeSitterFile::try_build(src, lang_id) {
                Ok(ts) => ts,
                Err(err) => {
                    // Parsing/query mismatch should not fail the entire repo scan.
                    tracing::warn!("skip unparsable file: {:?}, err={err}", file.rel_path);
                    continue;
                }
            };
            let sg = ts.scope_graph().map_err(|e| NavigationError::TreeSitter {
                rel_path: file.rel_path.clone(),
                source: e,
            })?;

            // Collect definitions with their symbol kind priority
            // Priority: class/struct/enum/union > typedef/alias > function > others
            let mut file_defs: Vec<(SymbolLocation, u8)> = Vec::new();

            // Get language namespaces for symbol kind lookup
            let lang_config = crate::ALL_LANGUAGES
                .iter()
                .find(|l| l.language_ids.contains(&lang_id));

            for idx in sg.graph.node_indices() {
                // Phase-1 default was to only scan Rust top-level defs to avoid returning locals.
                // For other languages (e.g. C++), many important symbols live under namespace/class
                // scopes, so we intentionally relax the filter.
                if lang_id == "rust" && !sg.is_top_level(idx) {
                    continue;
                }
                let Some(crate::NodeKind::Def(d)) = sg.get_node(idx) else {
                    continue;
                };
                if d.name(src) != name.as_bytes() {
                    continue;
                }

                // Determine priority based on symbol kind
                let sym_name = if let (Some(sym_id), Some(config)) = (d.symbol_id, lang_config) {
                    Some(sym_id.name(config.namespaces))
                } else {
                    None
                };

                let mut priority = match sym_name {
                    Some("class" | "struct" | "enum" | "union") => 0,
                    Some("typedef" | "alias") => 1,
                    Some("function") => 3,
                    Some("variable") => 4,
                    Some(_) => 5,
                    None => 5,
                };

                // Workaround for tree-sitter-cpp limitation: class definitions with macro modifiers
                // like `class LEVELDB_EXPORT Status` are parsed as function definitions.
                // Check if the source context contains 'class' or 'struct' keyword.
                let lang_id_lower = lang_id.to_lowercase();
                if priority == 3 && (lang_id_lower == "c++" || lang_id_lower == "c") {
                    // Get the line containing this definition
                    let _line_num = d.range.start.line;
                    // Find the start of this line in the source
                    let line_start_byte = src
                        .iter()
                        .take(d.range.start.byte)
                        .enumerate()
                        .filter(|(_, &b)| b == b'\n')
                        .next_back()
                        .map(|(i, _)| i + 1)
                        .unwrap_or(0);
                    // Find the end of this line
                    let line_end_byte = src
                        .iter()
                        .skip(d.range.start.byte)
                        .position(|&b| b == b'\n')
                        .map(|pos| d.range.start.byte + pos)
                        .unwrap_or(src.len());
                    // Get the line content
                    let line_src = &src[line_start_byte..line_end_byte];
                    let line_str = String::from_utf8_lossy(line_src);
                    // Check if the line contains 'class' or 'struct'
                    if line_str.contains("class") || line_str.contains("struct") {
                        priority = 0; // Treat as class/struct definition
                    }
                }

                file_defs.push((
                    SymbolLocation {
                        rel_path: file.rel_path.clone(),
                        range: d.range,
                    },
                    priority,
                ));
            }

            // Sort by priority (lower is better), then by line number
            file_defs.sort_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| a.0.range.start.line.cmp(&b.0.range.start.line))
            });

            for (loc, _) in file_defs {
                out.push(loc);
            }
        }

        Ok(out)
    }

    fn get_symbol_context(
        &self,
        repo_root: &Path,
        location: &SymbolLocation,
        opt: &SnippetOptions,
    ) -> Result<SymbolContext, NavigationError> {
        let abs_path = repo_root.join(&location.rel_path);
        let content = std::fs::read_to_string(&abs_path).map_err(|e| NavigationError::Io {
            path: abs_path,
            source: e,
        })?;

        let (signature_line, snippet) =
            Self::extract_signature_and_snippet(&content, &location.range, opt);

        Ok(SymbolContext {
            location: location.clone(),
            signature_line,
            snippet,
        })
    }

    fn find_references(
        &self,
        repo_root: &Path,
        name: &str,
        max: usize,
    ) -> Result<Vec<SymbolLocation>, NavigationError> {
        let files = self.provider.list_files(repo_root, &self.scan_opt)?;
        let mut out = Vec::new();

        for file in files {
            if out.len() >= max {
                break;
            }
            let remain = max.saturating_sub(out.len());

            // Semantic-first: count only parsed reference nodes.
            let semantic = Self::semantic_references_in_file(&file, name, remain);
            if !semantic.is_empty() {
                out.extend(semantic);
                continue;
            }

            // Fallback: text-based occurrence scan.
            let ranges = Self::find_identifier_occurrences(&file.content, name, remain);
            for r in ranges {
                out.push(SymbolLocation {
                    rel_path: file.rel_path.clone(),
                    range: r,
                });
                if out.len() >= max {
                    break;
                }
            }
        }

        Ok(out)
    }

    fn goto_definition_at(
        &self,
        repo_root: &Path,
        rel_path: &Path,
        line: usize,
        column: usize,
    ) -> Result<Vec<SymbolLocation>, NavigationError> {
        let lang_id = Self::lang_id_for_path(rel_path).unwrap_or("rust");
        let abs_path = repo_root.join(rel_path);
        let content = std::fs::read(&abs_path).map_err(|e| NavigationError::Io {
            path: abs_path.clone(),
            source: e,
        })?;
        let ts = TreeSitterFile::try_build(&content, lang_id).map_err(|e| {
            NavigationError::TreeSitter {
                rel_path: rel_path.to_path_buf(),
                source: e,
            }
        })?;
        let sg = ts.scope_graph().map_err(|e| NavigationError::TreeSitter {
            rel_path: rel_path.to_path_buf(),
            source: e,
        })?;

        let Some(node_idx) = sg.node_by_position(line, column) else {
            return Ok(Vec::new());
        };

        // If on a definition, return itself.
        if let Some(crate::NodeKind::Def(d)) = sg.get_node(node_idx) {
            return Ok(vec![SymbolLocation {
                rel_path: rel_path.to_path_buf(),
                range: d.range,
            }]);
        }

        // If on a reference, return resolved definitions.
        let defs = sg
            .definitions(node_idx)
            .filter_map(|def_idx| match sg.get_node(def_idx) {
                Some(crate::NodeKind::Def(d)) => Some(SymbolLocation {
                    rel_path: rel_path.to_path_buf(),
                    range: d.range,
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        Ok(defs)
    }
}

impl<P: RepoFileProvider> TreeSitterNavigator<P> {
    fn semantic_references_in_file(
        file: &crate::repo_scan::RepoFile,
        name: &str,
        max: usize,
    ) -> Vec<SymbolLocation> {
        if max == 0 {
            return Vec::new();
        }
        let Some(lang_id) = Self::lang_id_for_path(&file.rel_path) else {
            return Vec::new();
        };

        let src = file.content.as_bytes();
        let ts = match TreeSitterFile::try_build(src, lang_id) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let sg = match ts.scope_graph() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let mut out = Vec::new();
        for idx in sg.graph.node_indices() {
            if out.len() >= max {
                break;
            }
            let Some(crate::NodeKind::Ref(r)) = sg.get_node(idx) else {
                continue;
            };
            if r.name(src) != name.as_bytes() {
                continue;
            }
            out.push(SymbolLocation {
                rel_path: file.rel_path.clone(),
                range: r.range,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::SystemTime};

    fn unique_tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("luna-intel-test-{nanos}"))
    }

    #[test]
    fn goto_definition_finds_top_level_rust_symbol() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(
            root.join("src/lib.rs"),
            "pub struct Foo;\n\npub fn bar() {}\n",
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let defs = nav.goto_definition(&root, "bar").unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].rel_path, PathBuf::from("src/lib.rs"));
        assert_eq!(defs[0].range.start.line, 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn get_symbol_context_returns_snippet() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(
            root.join("src/lib.rs"),
            "pub struct Foo;\n\npub fn bar() {}\n",
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let loc = nav.goto_definition(&root, "bar").unwrap().remove(0);
        let ctx = nav
            .get_symbol_context(
                &root,
                &loc,
                &SnippetOptions {
                    context_lines: 1,
                    ..SnippetOptions::default()
                },
            )
            .unwrap();

        assert!(ctx.snippet.contains("pub fn "));
        assert!(ctx.snippet.contains("§bar§"));
        assert!(ctx.snippet.contains("3 "));
        assert_eq!(ctx.signature_line.as_deref(), Some("pub fn bar()"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn extract_definition_signature_strips_trailing_brace_and_semicolon() {
        let lines = vec!["pub trait Navigator {", "    fn foo();", "}"];
        let sig =
            TreeSitterNavigator::<FsRepoFileProvider>::extract_definition_signature(&lines, 0)
                .unwrap();
        assert_eq!(sig, "pub trait Navigator");

        let lines = vec!["pub struct Foo;"];
        let sig =
            TreeSitterNavigator::<FsRepoFileProvider>::extract_definition_signature(&lines, 0)
                .unwrap();
        assert_eq!(sig, "pub struct Foo");
    }

    #[test]
    fn goto_definition_finds_cpp_symbol_in_namespace_scope() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(
            root.join("src/a.cc"),
            r#"
namespace brpc {
class ContentionProfiler {
public:
    void Foo();
};
}
"#,
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let defs = nav.goto_definition(&root, "ContentionProfiler").unwrap();
        assert!(!defs.is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn get_symbol_context_extracts_multiline_function_signature() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(
            root.join("src/lib.rs"),
            "pub fn foo(\n    a: i32,\n) -> i32 {\n    a\n}\n",
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let loc = nav.goto_definition(&root, "foo").unwrap().remove(0);
        let ctx = nav
            .get_symbol_context(
                &root,
                &loc,
                &SnippetOptions {
                    context_lines: 10,
                    ..SnippetOptions::default()
                },
            )
            .unwrap();

        let sig = ctx.signature_line.unwrap();
        assert!(sig.contains("pub fn foo("));
        assert!(sig.contains(") -> i32"));
        assert!(!sig.contains('{'));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn get_symbol_context_signature_does_not_include_body_brace() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(root.join("src/lib.rs"), "pub fn foo() -> i32 { 1 }\n").unwrap();

        let nav = TreeSitterNavigator::default();
        let loc = nav.goto_definition(&root, "foo").unwrap().remove(0);
        let ctx = nav
            .get_symbol_context(
                &root,
                &loc,
                &SnippetOptions {
                    context_lines: 2,
                    ..SnippetOptions::default()
                },
            )
            .unwrap();

        let sig = ctx.signature_line.unwrap();
        assert_eq!(sig, "pub fn foo() -> i32");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn goto_definition_at_resolves_reference_in_same_file() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();

        fs::write(
            root.join("src/lib.rs"),
            "pub fn bar() {}\npub fn foo() { bar(); }\n",
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let defs = nav
            .goto_definition_at(&root, Path::new("src/lib.rs"), 1, 15)
            .unwrap();

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].rel_path, PathBuf::from("src/lib.rs"));
        assert_eq!(defs[0].range.start.line, 0);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn find_references_text_finds_occurrences() {
        let root = unique_tmp_dir();
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn bar() {}\npub fn foo() { bar(); }\n",
        )
        .unwrap();

        let nav = TreeSitterNavigator::default();
        let refs = nav.find_references(&root, "bar", 10).unwrap();
        assert!(!refs.is_empty());

        let _ = fs::remove_dir_all(&root);
    }
}
