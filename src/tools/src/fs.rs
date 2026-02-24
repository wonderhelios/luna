//! File system operations for agents

use crate::detect_lang_id;
use anyhow::Result;
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
    entries.sort_by(|a, b| {
        b.is_dir.cmp(&a.is_dir)
            .then_with(|| a.name.cmp(&b.name))
    });

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
    /// Replace specific line range (1-based, inclusive)
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

    let new_content = match op {
        EditOp::ReplaceAll { new_content } => new_content.clone(),
        EditOp::ReplaceLines { start_line, end_line, new_content } => {
            let start = start_line.saturating_sub(1);
            let end = end_line.saturating_sub(1);
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

            let _lines_changed = end - start + 1;
            let new_lines: Vec<&str> = new_content.lines().collect();

            // Replace the line range
            let mut new_lines_all = lines[..start].to_vec();
            new_lines_all.extend(new_lines);
            if end + 1 < lines.len() {
                new_lines_all.extend(lines[end + 1..].to_vec());
            }

            new_lines_all.join("\n") + "\n"
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
        lines_changed: Some(new_content.lines().count()),
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

/// List symbols in a file with enhanced filtering
pub fn list_symbols_enhanced(
    path: &Path,
    _options: &SymbolListOptions,
) -> Result<Vec<SymbolDetail>> {
    use intelligence::TreeSitterFile;

    let content = fs::read(path)?;
    let lang_id = detect_lang_id(path).unwrap_or("");

    let ts_file = TreeSitterFile::try_build(&content, lang_id)
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    let scope_graph = ts_file.scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    let src_str = String::from_utf8_lossy(&content);

    let mut symbols = Vec::new();

    for idx in scope_graph.graph.node_indices() {
        if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
            let name = String::from_utf8_lossy(def.name(src_str.as_bytes())).to_string();
            let kind = def.symbol_id
                .and_then(|id| {
                    // Get language from detection instead
                    let lang_id = detect_lang_id(path).unwrap_or("");
                    intelligence::ALL_LANGUAGES
                        .iter()
                        .find(|l| l.language_ids.contains(&lang_id))
                        .and_then(|l| l.namespaces.get(id.namespace_idx)
                            .and_then(|ns| ns.get(id.symbol_idx))
                            .copied())
                })
                .unwrap_or("unknown");

            symbols.push(SymbolDetail {
                name,
                kind: kind.to_string(),
                start_line: def.range.start.line + 1,
                end_line: def.range.end.line + 1,
                visibility: "public".to_string(), // TODO: detect from source
            });
        }
    }

    Ok(symbols)
}

/// List symbols filtered by kind
pub fn list_symbols_by_kind(
    path: &Path,
    kind: &str,
) -> Result<Vec<SymbolDetail>> {
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
