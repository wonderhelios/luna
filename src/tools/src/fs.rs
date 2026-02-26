//! File system operations for agents

use crate::{detect_lang_id, LunaError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ============================================================================
// Read File
// ============================================================================

/// Read file content, optionally with line range
pub fn read_file(path: &Path, range: Option<(usize, usize)>) -> Result<String> {
    let s = fs::read_to_string(path)?;
    if let Some((start, end)) = range {
        if start > end {
            return Ok(String::new());
        }
        let mut out = String::new();
        for (i, line) in s.lines().enumerate() {
            if i < start {
                continue;
            }
            if i > end {
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
        Ok(out)
    } else {
        Ok(s)
    }
}

/// Read file by line numbers (0-based)
pub fn read_file_by_lines(
    repo_root: &Path,
    rel_path: &str,
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let full = repo_root.join(rel_path);
    read_file(&full, Some((start_line, end_line)))
}

// ============================================================================
// List Directory
// ============================================================================

/// Directory entry for list_dir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub size: Option<u64>,
}

/// List directory contents
pub fn list_dir(path: &Path) -> Result<Vec<DirEntry>> {
    let mut entries = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let file_type = entry.file_type()?;
        let metadata = entry.metadata().ok();

        entries.push(DirEntry {
            name: name.clone(),
            path: path.join(&name).to_string_lossy().to_string(),
            is_dir: file_type.is_dir(),
            is_file: file_type.is_file(),
            size: metadata.map(|m| m.len()).filter(|_| file_type.is_file()),
        });
    }

    // Sort: directories first, then alphabetically
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.name.cmp(&b.name)));

    Ok(entries)
}

// ============================================================================
// Edit File
// ============================================================================

/// Edit operation for edit_file tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditOp {
    /// Replace entire file content
    ReplaceAll { new_content: String },
    /// Replace specific line range (0-based, inclusive)
    ReplaceLines {
        start_line: usize,
        end_line: usize,
        new_content: String,
    },
    /// Unified diff format (simplified) - TODO
    UnifiedDiff { diff: String },
}

/// Result of edit_file operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditResult {
    pub path: String,
    pub success: bool,
    pub lines_changed: Option<usize>,
    pub error: Option<String>,
    pub backup_path: Option<String>,
}

/// Edit file with automatic backup
pub fn edit_file(path: &Path, op: &EditOp, create_backup: bool) -> Result<EditResult> {
    let path_str = path.to_string_lossy().to_string();

    // Read original content
    let original = fs::read_to_string(path)?;

    // Create backup if requested
    let backup_path = if create_backup {
        let backup = format!("{}.backup", path_str);
        fs::write(&backup, &original)?;
        Some(backup)
    } else {
        None
    };

    let (new_content, lines_changed) = match op {
        EditOp::ReplaceAll { new_content } => {
            (new_content.clone(), Some(new_content.lines().count()))
        }
        EditOp::ReplaceLines {
            start_line,
            end_line,
            new_content,
        } => {
            let start = *start_line;
            let end = *end_line;
            let lines: Vec<&str> = original.lines().collect();

            if start >= lines.len() || end >= lines.len() || start > end {
                return Ok(EditResult {
                    path: path_str,
                    success: false,
                    lines_changed: None,
                    error: Some(format!("Invalid line range: {}..={}", start_line, end_line)),
                    backup_path,
                });
            }

            let replaced_lines = end - start + 1;
            let new_lines: Vec<&str> = new_content.lines().collect();

            // Replace the line range
            let mut new_lines_all = lines[..start].to_vec();
            new_lines_all.extend(new_lines);
            if end + 1 < lines.len() {
                new_lines_all.extend(lines[end + 1..].to_vec());
            }

            (new_lines_all.join("\n") + "\n", Some(replaced_lines))
        }
        EditOp::UnifiedDiff { .. } => {
            return Ok(EditResult {
                path: path_str,
                success: false,
                lines_changed: None,
                error: Some("UnifiedDiff not yet implemented".to_string()),
                backup_path,
            });
        }
    };

    // Write new content
    fs::write(path, &new_content)?;

    Ok(EditResult {
        path: path_str,
        success: true,
        lines_changed,
        error: None,
        backup_path,
    })
}

// ============================================================================
// List Symbols (Enhanced)
// ============================================================================

/// Symbol visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolVisibility {
    Public,
    Private,
    All,
}

/// Symbol sort order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolSortOrder {
    Name,
    Kind,
    Position,
}

/// Options for listing symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolListOptions {
    pub visibility: SymbolVisibility,
    pub sort_by: SymbolSortOrder,
    pub kinds: Vec<String>, // Filter by kinds (empty = all)
}

impl Default for SymbolListOptions {
    fn default() -> Self {
        Self {
            visibility: SymbolVisibility::All,
            sort_by: SymbolSortOrder::Position,
            kinds: Vec::new(),
        }
    }
}

/// Detailed symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDetail {
    pub name: String,
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
    pub visibility: String,
}

/// Detect visibility of a symbol from source code
///
/// For Rust: checks for `pub`, `pub(crate)`, `pub(mod)`, etc.
/// For other languages: checks for common public keywords
fn detect_visibility(src: &str, range: &core::text_range::TextRange, lang_id: &str) -> String {
    // Get the line containing the definition
    let line_start = src[..range.start.byte].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[range.start.byte..]
        .find('\n')
        .map(|i| range.start.byte + i)
        .unwrap_or(src.len());
    let line = &src[line_start..line_end];

    match lang_id {
        "rust" => {
            // Check for pub, pub(crate), pub(super), pub(in path), etc.
            if line.trim_start().starts_with("pub") {
                if line.contains("pub(") {
                    // Extract visibility modifier like pub(crate)
                    if let Some(start) = line.find("pub(") {
                        if let Some(end) = line[start..].find(')') {
                            return format!("pub{}", &line[start + 3..start + end + 1]);
                        }
                    }
                }
                "pub".to_string()
            } else {
                "private".to_string()
            }
        }
        "python" => {
            // Python: no underscore prefix = public, single underscore = protected, double = private
            let name = &src[range.start.byte..range.end.byte];
            if name.starts_with("__") && !name.ends_with("__") {
                "private".to_string()
            } else if name.starts_with('_') {
                "protected".to_string()
            } else {
                "public".to_string()
            }
        }
        "javascript" | "typescript" => {
            // JS/TS: export = public, no export = private
            // Check if we're at module level and have export keyword before this
            let preceding = &src[line_start..range.start.byte];
            if preceding.trim().ends_with("export") || line.contains("export ") {
                "public".to_string()
            } else {
                "private".to_string()
            }
        }
        "go" => {
            // Go: uppercase first letter = exported (public), lowercase = private
            let name = &src[range.start.byte..range.end.byte];
            if let Some(first_char) = name.chars().next() {
                if first_char.is_uppercase() {
                    "public".to_string()
                } else {
                    "private".to_string()
                }
            } else {
                "unknown".to_string()
            }
        }
        "java" | "kotlin" => {
            // Java/Kotlin: public, protected, private, or package-private (default)
            let trimmed = line.trim_start();
            if trimmed.starts_with("public ") {
                "public".to_string()
            } else if trimmed.starts_with("protected ") {
                "protected".to_string()
            } else if trimmed.starts_with("private ") {
                "private".to_string()
            } else {
                "package".to_string()
            }
        }
        "c" | "cpp" | "c++" => {
            // C/C++: no standard visibility, but we can check for static
            let trimmed = line.trim_start();
            if trimmed.starts_with("static ") {
                "internal".to_string()
            } else {
                "public".to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

/// List symbols in a file with enhanced filtering
pub fn list_symbols_enhanced(
    path: &Path,
    options: &SymbolListOptions,
) -> Result<Vec<SymbolDetail>> {
    use intelligence::TreeSitterFile;

    let content = fs::read(path)?;
    let lang_id = detect_lang_id(path).unwrap_or("");

    let ts_file = TreeSitterFile::try_build(&content, lang_id)
        .map_err(|e| LunaError::search(format!("Failed to parse {}: {:?}", path.display(), e)))?;

    let scope_graph = ts_file.scope_graph().map_err(|e| {
        LunaError::search(format!(
            "Failed to build scope graph for {}: {:?}",
            path.display(),
            e
        ))
    })?;

    // Get the language configuration once
    let lang_config = intelligence::ALL_LANGUAGES
        .iter()
        .find(|l| l.language_ids.contains(&lang_id));

    let src_str = String::from_utf8_lossy(&content);

    let mut symbols = Vec::new();

    for idx in scope_graph.graph.node_indices() {
        if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
            let name = String::from_utf8_lossy(def.name(src_str.as_bytes())).to_string();

            // Get symbol kind from namespace
            let kind = def
                .symbol_id
                .and_then(|id| {
                    lang_config.and_then(|l| {
                        l.namespaces
                            .get(id.namespace_idx)
                            .and_then(|ns| ns.get(id.symbol_idx))
                            .copied()
                    })
                })
                .unwrap_or("unknown");

            // Detect visibility from source
            let visibility = detect_visibility(&src_str, &def.range, lang_id);

            // Filter by visibility if requested
            match options.visibility {
                SymbolVisibility::Public if visibility != "public" => continue,
                SymbolVisibility::Private if visibility != "private" => continue,
                _ => {}
            }

            // Filter by kind if requested
            if !options.kinds.is_empty() && !options.kinds.iter().any(|k| k == kind) {
                continue;
            }

            symbols.push(SymbolDetail {
                name,
                kind: kind.to_string(),
                start_line: def.range.start.line + 1,
                end_line: def.range.end.line + 1,
                visibility,
            });
        }
    }

    // Sort symbols according to options
    match options.sort_by {
        SymbolSortOrder::Name => symbols.sort_by(|a, b| a.name.cmp(&b.name)),
        SymbolSortOrder::Kind => symbols.sort_by(|a, b| a.kind.cmp(&b.kind)),
        SymbolSortOrder::Position => {} // Already in position order
    }

    Ok(symbols)
}

/// List symbols filtered by kind
pub fn list_symbols_by_kind(path: &Path, kind: &str) -> Result<Vec<SymbolDetail>> {
    let options = SymbolListOptions {
        kinds: vec![kind.to_string()],
        ..Default::default()
    };
    let symbols = list_symbols_enhanced(path, &options)?;
    Ok(symbols.into_iter().filter(|s| s.kind == kind).collect())
}

/// List only public functions
pub fn list_public_functions(path: &Path) -> Result<Vec<SymbolDetail>> {
    let options = SymbolListOptions {
        visibility: SymbolVisibility::Public,
        kinds: vec!["function".to_string()],
        ..Default::default()
    };
    list_symbols_enhanced(path, &options)
}
