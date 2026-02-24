use crate::types::{ContextPack, EditOp, EditResult, TerminalResult, ToolName, ToolTrace};
use crate::intel_adapter::{ParsedFile, SymbolInfo as IntelSymbolInfo, SymbolLocation as IntelSymbolLocation, SymbolKind};
use anyhow::Result;
use core::code_chunk::{ContextChunk, IndexChunk, IndexChunkOptions, RefillOptions};
use intelligence::ALL_LANGUAGES;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCodeOptions {
    pub max_files: usize,
    pub max_hits: usize,
    pub max_file_bytes: usize,
    pub ignore_dirs: Vec<String>,
}

impl Default for SearchCodeOptions {
    fn default() -> Self {
        Self {
            max_files: 8_000,
            max_hits: 64,
            max_file_bytes: 500 * 1_000,
            ignore_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "dist".to_string(),
                "build".to_string(),
            ],
        }
    }
}

// Read file
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

pub fn read_file_by_lines(
    repo_root: &Path,
    rel_path: &str,
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let full = repo_root.join(rel_path);
    read_file(&full, Some((start_line, end_line)))
}

// Infer language id from file extension (for tree-sitter parsing)
pub fn detect_lang_id(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    ALL_LANGUAGES
        .iter()
        .copied()
        .find(|cfg| cfg.file_extensions.iter().any(|e| e.to_lowercase() == ext))
        .and_then(|cfg| cfg.language_ids.first().copied())
}

// Keyword placeholder for search_code: scan repo files, normalize hits using IndexChunk protocol
// Returns: IndexChunk hits (each chunk's text contains query)
pub fn search_code_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &Tokenizer,
    idx_opt: IndexChunkOptions,
    opt: SearchCodeOptions,
) -> Result<(Vec<IndexChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();

    let q = query.trim();
    if q.is_empty() {
        return Ok((Vec::new(), trace));
    }

    let terms = q
        .split_whitespace()
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Ok((Vec::new(), trace));
    }

    let mut hits = Vec::new();
    let mut scanned_files = 0usize;

    let mut stack = vec![repo_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if scanned_files >= opt.max_files || hits.len() >= opt.max_hits {
            break;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            if scanned_files >= opt.max_files || hits.len() >= opt.max_hits {
                break;
            }
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if let Some(name) = path.file_name().map(|s| s.to_string_lossy().to_string()) {
                    if opt.ignore_dirs.iter().any(|d| d == &name) {
                        continue;
                    }
                }
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.len() as usize > opt.max_file_bytes {
                continue;
            }

            scanned_files += 1;

            let Some(lang_id) = detect_lang_id(&path) else {
                continue;
            };
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Coarse filter: does file content contain query
            let content = String::from_utf8_lossy(&bytes);
            let content_lower = content.to_lowercase();
            if !terms.iter().any(|t| content_lower.contains(t)) {
                continue;
            }

            let rel = path.strip_prefix(repo_root).unwrap_or(&path);
            let rel_str = rel.to_string_lossy().to_string();

            // Normalize using IndexChunk protocol (scope -> token budget)
            let idx_chunks =
                index::index_chunks("", &rel_str, &bytes, lang_id, tokenizer, idx_opt.clone());
            for c in idx_chunks {
                if hits.len() >= opt.max_hits {
                    break;
                }
                let text_lower = c.text.to_lowercase();
                if terms.iter().any(|t| text_lower.contains(t)) {
                    hits.push(c);
                }
            }
        }
    }

    trace.push(ToolTrace {
        tool: ToolName::SearchCode,
        summary: format!(
            "keyword search scanned_files={} hits={} terms={:?}",
            scanned_files,
            hits.len(),
            terms,
        ),
    });
    Ok((hits, trace))
}

// Refill IndexChunk hits into ContextChunk (group by file, parse each file only once)
pub fn refill_hits(
    repo_root: &Path,
    hits: &[IndexChunk],
    opt: RefillOptions,
) -> Result<(Vec<ContextChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();

    let mut hits_by_path: BTreeMap<String, Vec<IndexChunk>> = BTreeMap::new();
    for h in hits {
        hits_by_path
            .entry(h.path.clone())
            .or_default()
            .push(h.clone());
    }
    let file_count = hits_by_path.len();

    let mut out = Vec::new();
    for (rel_path, mut hs) in hits_by_path {
        hs.sort_by_key(|h| h.start_byte);
        let full_path = repo_root.join(&rel_path);
        let bytes = match fs::read(&full_path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let Some(lang_id) = detect_lang_id(&full_path) else {
            continue;
        };
        let ctx = index::refill_chunks(&rel_path, &bytes, lang_id, &hs, opt.clone())?;
        out.extend(ctx);
    }

    // Dedup: multiple IndexChunk hits may fall in the same enclosing scope
    let raw_count = out.len();
    let out = core::code_chunk::dedup_context_chunks(out);

    trace.push(ToolTrace {
        tool: ToolName::RefillChunks,
        summary: format!(
            "refill files={} context_chunks={} (raw={})",
            file_count,
            out.len(),
            raw_count
        ),
    });
    Ok((out, trace))
}

/// Directory entry for list_dir tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub size: Option<u64>,
}

/// List directory contents (non-recursive, one level)
pub fn list_dir(dir: &Path) -> Result<Vec<DirEntry>> {
    let mut entries = Vec::new();

    let rd = fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("failed to read directory {:?}: {}", dir, e))?;

    for entry in rd {
        let entry = entry.map_err(|e| anyhow::anyhow!("failed to read dir entry: {}", e))?;
        let path = entry.path();

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "?".to_string());

        let rel_path = path
            .strip_prefix(dir)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        let ft = entry.file_type().map_err(|e| anyhow::anyhow!("failed to get file type: {}", e))?;
        let is_dir = ft.is_dir();
        let is_file = ft.is_file();

        let size = if is_file {
            fs::metadata(&path).ok().map(|m| m.len())
        } else {
            None
        };

        entries.push(DirEntry {
            name,
            path: rel_path,
            is_dir,
            is_file,
            size,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Edit file with backup support
///
/// # Arguments
/// * `path` - Full path to the file
/// * `op` - Edit operation to perform
/// * `create_backup` - Whether to create .backup file before editing
///
/// # Returns
/// Edit result with success status and optional error message
pub fn edit_file(path: &Path, op: &EditOp, create_backup: bool) -> Result<EditResult> {
    let path_str = path.to_string_lossy().to_string();

    // Create backup if requested
    let backup_path = if create_backup {
        let backup = format!("{}.backup", path_str);
        if path.exists() {
            fs::copy(&path, &backup).map_err(|e| {
                anyhow::anyhow!("failed to create backup {:?}: {}", backup, e)
            })?;
            Some(backup)
        } else {
            None
        }
    } else {
        None
    };

    let lines_changed = match op {
        EditOp::ReplaceAll { new_content } => {
            fs::write(&path, new_content)
                .map_err(|e| anyhow::anyhow!("failed to write {:?}: {}", path, e))?;
            Some(new_content.lines().count())
        }
        EditOp::ReplaceLines {
            start_line,
            end_line,
            new_content,
        } => {
            if !path.exists() {
                return Ok(EditResult {
                    path: path_str,
                    success: false,
                    lines_changed: None,
                    error: Some("file does not exist".to_string()),
                    backup_path,
                });
            }

            let original = fs::read_to_string(&path).map_err(|e| {
                anyhow::anyhow!("failed to read {:?} for editing: {}", path, e)
            })?;

            let mut lines: Vec<&str> = original.lines().collect();

            // Convert to 0-based indices
            let start = start_line.saturating_sub(1);
            let end = end_line.saturating_sub(1).min(lines.len().saturating_sub(1));

            if start > end || end >= lines.len() {
                return Ok(EditResult {
                    path: path_str,
                    success: false,
                    lines_changed: None,
                    error: Some(format!(
                        "invalid line range: {}..={} (file has {} lines)",
                        start_line,
                        end_line,
                        lines.len()
                    )),
                    backup_path,
                });
            }

            let new_lines: Vec<&str> = new_content.lines().collect();
            let _removed = end - start + 1;

            // Replace the range
            lines.drain(start..=end);
            for (i, line) in new_lines.iter().enumerate() {
                lines.insert(start + i, *line);
            }

            // Auto-remove duplicate consecutive lines
            let mut deduped_lines = Vec::new();
            let mut prev_line: Option<&str> = None;
            for line in lines.iter() {
                if Some(*line) != prev_line {
                    deduped_lines.push(*line);
                    prev_line = Some(*line);
                }
            }

            // If deduplication removed lines, update the file
            if deduped_lines.len() != lines.len() {
                let new_content = deduped_lines.join("\n") + "\n";
                fs::write(&path, new_content)
                    .map_err(|e| anyhow::anyhow!("failed to write edited {:?}: {}", path, e))?;

                return Ok(EditResult {
                    path: path_str,
                    success: true,
                    lines_changed: Some(new_lines.len()),
                    error: Some(format!("removed {} duplicate lines", lines.len() - deduped_lines.len())),
                    backup_path,
                });
            }

            let new_content = lines.join("\n") + "\n";
            fs::write(&path, new_content)
                .map_err(|e| anyhow::anyhow!("failed to write edited {:?}: {}", path, e))?;

            Some(new_lines.len())
        }
        EditOp::UnifiedDiff { diff: _ } => {
            // TODO: Implement proper unified diff parsing
            // For now, return error indicating not yet implemented
            return Ok(EditResult {
                path: path_str,
                success: false,
                lines_changed: None,
                error: Some("unified diff parsing not yet implemented".to_string()),
                backup_path,
            });
        }
    };

    Ok(EditResult {
        path: path_str,
        success: true,
        lines_changed,
        error: None,
        backup_path,
    })
}

/// Dangerous commands that should require user confirmation
const DANGEROUS_COMMANDS: &[&str] = &[
    "rm -rf",
    "rm -r",
    "del /f",
    "format",
    "mkfs",
    "shutdown",
    "reboot",
    "dd if=",
    "> /dev/",
    "kill -9",
];

/// Check if a command is dangerous
fn is_dangerous_command(cmd: &str) -> bool {
    let cmd_lower = cmd.to_lowercase();
    DANGEROUS_COMMANDS
        .iter()
        .any(|dangerous| cmd_lower.contains(dangerous))
}

/// Run terminal command with safety checks
///
/// # Arguments
/// * `command` - Command string to execute (e.g., "cargo build" or ["cargo", "build"])
/// * `cwd` - Working directory (None = use current dir)
/// * `allow_dangerous` - If false, dangerous commands will be rejected
///
/// # Returns
/// Terminal execution result
pub fn run_terminal(command: &str, cwd: Option<&Path>, allow_dangerous: bool) -> Result<TerminalResult> {
    // Safety check for dangerous commands
    if !allow_dangerous && is_dangerous_command(command) {
        return Ok(TerminalResult {
            command: command.to_string(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            success: false,
            error: Some(format!(
                "command rejected as dangerous: {}. Use allow_dangerous=true to override.",
                command
            )),
        });
    }

    // Parse command: split by spaces but respect quotes
    let args = parse_command_args(command);
    if args.is_empty() {
        return Ok(TerminalResult {
            command: command.to_string(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            success: false,
            error: Some("empty command".to_string()),
        });
    }

    let program = &args[0];
    let cmd_args = &args[1..];

    let mut cmd = Command::new(program);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.args(cmd_args);

    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) => {
            return Ok(TerminalResult {
                command: command.to_string(),
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                success: false,
                error: Some(format!("failed to execute command: {}", e)),
            });
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();
    let exit_code = output.status.code();

    Ok(TerminalResult {
        command: command.to_string(),
        exit_code,
        stdout,
        stderr,
        success,
        error: if success { None } else { Some("command failed".to_string()) },
    })
}

/// Simple command argument parser (respects quotes)
fn parse_command_args(cmd: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut escape = false;

    for ch in cmd.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }

        match ch {
            '\\' => {
                escape = true;
            }
            '"' => {
                in_quote = !in_quote;
            }
            ' ' | '\t' if !in_quote => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        args.push(current);
    }

    args
}

/// Convenience function: search -> refill -> pack
pub fn build_context_pack_keyword(
    repo_root: &Path,
    query: &str,
    tokenizer: &Tokenizer,
    search_opt: SearchCodeOptions,
    idx_opt: IndexChunkOptions,
    refill_opt: RefillOptions,
) -> Result<ContextPack> {
    let (hits, mut trace) = search_code_keyword(repo_root, query, tokenizer, idx_opt, search_opt)?;
    let (context, mut trace2) = refill_hits(repo_root, &hits, refill_opt)?;
    trace.append(&mut trace2);

    Ok(ContextPack {
        query: query.to_string(),
        hits,
        context,
        trace,
    })
}

// ============================================================================
// Symbol Listing Module
// ============================================================================

/// Enhanced symbol information extracted from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDetail {
    /// Symbol name
    pub name: String,

    /// Symbol kind (from intelligence module)
    pub kind: SymbolKind,

    /// File path containing this symbol
    pub path: String,

    /// Start line (1-based)
    pub start_line: usize,

    /// End line (1-based)
    pub end_line: usize,

    /// Byte position of symbol start
    pub start_byte: usize,

    /// Byte position of symbol end
    pub end_byte: usize,

    /// Function/method signature (if applicable)
    pub signature: Option<String>,

    /// Parent scope name (if nested)
    pub parent_scope: Option<String>,

    /// Visibility modifier (pub, private, etc.)
    pub visibility: SymbolVisibility,
}

/// Symbol visibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SymbolVisibility {
    Public,
    Private,
    Protected,
    Internal,
    Unknown,
}

impl SymbolVisibility {
    /// Create from string identifier (language-agnostic)
    pub fn from_src_line(line: &str) -> Self {
        let line_lower = line.trim().to_lowercase();

        // Multi-language visibility detection
        if line_lower.starts_with("pub")
            || line_lower.starts_with("public")
            || line_lower == "@staticmethod"  // Python
            || line_lower.starts_with("@")  // Python decorators often indicate public
        {
            SymbolVisibility::Public
        } else if line_lower.starts_with("protected") {
            SymbolVisibility::Protected
        } else if line_lower.starts_with("internal") {
            SymbolVisibility::Internal
        } else if line_lower.starts_with("_") {
            // Python: single leading underscore indicates "protected" (internal use)
            SymbolVisibility::Protected
        } else {
            SymbolVisibility::Private
        }
    }
}

/// Filter options for symbol listing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolFilter {
    /// Filter by symbol kind
    pub kinds: Vec<SymbolKind>,

    /// Filter by name pattern (simple contains match)
    pub name_pattern: Option<String>,

    /// Minimum line number (inclusive)
    pub min_line: Option<usize>,

    /// Maximum line number (inclusive)
    pub max_line: Option<usize>,

    /// Only public symbols
    pub public_only: bool,
}

impl SymbolFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a kind filter
    pub fn with_kind(mut self, kind: SymbolKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Add multiple kind filters
    pub fn with_kinds(mut self, kinds: Vec<SymbolKind>) -> Self {
        self.kinds = kinds;
        self
    }

    /// Add name pattern filter
    pub fn with_name_pattern(mut self, pattern: String) -> Self {
        self.name_pattern = Some(pattern);
        self
    }

    /// Add line range filter
    pub fn with_line_range(mut self, min: usize, max: usize) -> Self {
        self.min_line = Some(min);
        self.max_line = Some(max);
        self
    }

    /// Check if a symbol matches this filter
    pub fn matches(&self, symbol: &SymbolDetail) -> bool {
        // Check kind filter
        if !self.kinds.is_empty() && !self.kinds.contains(&symbol.kind) {
            return false;
        }

        // Check name pattern
        if let Some(pattern) = &self.name_pattern {
            if !symbol.name.to_lowercase().contains(&pattern.to_lowercase()) {
                return false;
            }
        }

        // Check line range
        if let Some(min) = self.min_line {
            if symbol.start_line < min {
                return false;
            }
        }
        if let Some(max) = self.max_line {
            if symbol.start_line > max {
                return false;
            }
        }

        // Check visibility
        if self.public_only && symbol.visibility != SymbolVisibility::Public {
            return false;
        }

        true
    }
}

/// Options for listing symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolListOptions {
    /// Filter to apply
    pub filter: SymbolFilter,

    /// Sort order
    pub sort_by: SymbolSortOrder,

    /// Maximum number of results
    pub limit: Option<usize>,
}

impl Default for SymbolListOptions {
    fn default() -> Self {
        Self {
            filter: SymbolFilter::default(),
            sort_by: SymbolSortOrder::Name,
            limit: None,
        }
    }
}

/// Sort order for symbol listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolSortOrder {
    /// Sort by symbol name
    Name,
    /// Sort by kind, then name
    KindThenName,
    /// Sort by line number
    Line,
    /// Sort by visibility (public first), then name
    VisibilityThenName,
}

/// Result of a repository-wide symbol search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositorySymbols {
    /// All found symbols
    pub symbols: Vec<SymbolDetail>,

    /// Files scanned
    pub files_scanned: usize,

    /// Files with parsing errors
    pub errors: Vec<FileParseError>,
}

/// Error from parsing a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileParseError {
    pub path: String,
    pub error: String,
}

// ============================================================================
// Symbol Listing Implementation
// ============================================================================

/// List symbols in a single file with full detail
///
/// This uses the intelligence module to provide rich symbol information
/// including symbol kind, location, and signature when available.
///
/// # Arguments
/// * `path` - Path to the file to analyze
/// * `options` - Optional listing parameters (filter, sort, limit)
///
/// # Returns
/// A vector of symbol details matching the criteria
pub fn list_symbols_enhanced(path: &Path, options: Option<SymbolListOptions>) -> Result<Vec<SymbolDetail>> {
    let parsed = ParsedFile::parse(path)?;
    let path_str = path.to_string_lossy().to_string();
    let src_str = std::str::from_utf8(&parsed.src).unwrap_or("");

    let mut symbols = Vec::new();

    // Extract all symbols
    for sym in parsed.extract_symbols() {
        let kind = classify_symbol_kind(&sym, src_str);
        let visibility = extract_visibility(src_str, &sym.location);
        let signature = extract_signature(src_str, &sym.location, &kind);
        let parent = extract_parent_scope(&parsed, &sym);

        symbols.push(SymbolDetail {
            name: sym.name,
            kind,
            path: path_str.clone(),
            start_line: sym.location.start_line,
            end_line: sym.location.end_line,
            start_byte: sym.location.start_byte,
            end_byte: sym.location.end_byte,
            signature,
            parent_scope: parent,
            visibility,
        });
    }

    // Apply filters
    let opts = options.unwrap_or_default();
    if !matches_filter(&opts.filter, &symbols) {
        symbols.retain(|s| opts.filter.matches(s));
    }

    // Sort
    sort_symbols(&mut symbols, &opts.sort_by);

    // Apply limit
    if let Some(limit) = opts.limit {
        symbols.truncate(limit);
    }

    Ok(symbols)
}

/// Check if any symbols match the filter (optimization)
fn matches_filter(filter: &SymbolFilter, _symbols: &[SymbolDetail]) -> bool {
    if filter.kinds.is_empty()
        && filter.name_pattern.is_none()
        && filter.min_line.is_none()
        && filter.max_line.is_none()
        && !filter.public_only
    {
        return false; // No filter active
    }
    true
}

/// Sort symbols by the specified order
fn sort_symbols(symbols: &mut Vec<SymbolDetail>, order: &SymbolSortOrder) {
    match order {
        SymbolSortOrder::Name => {
            symbols.sort_by(|a, b| {
                a.name.cmp(&b.name)
                    .then_with(|| a.path.cmp(&b.path))
                    .then_with(|| a.start_line.cmp(&b.start_line))
            });
        }
        SymbolSortOrder::KindThenName => {
            symbols.sort_by(|a, b| {
                a.kind.as_str().cmp(b.kind.as_str())
                    .then_with(|| a.name.cmp(&b.name))
            });
        }
        SymbolSortOrder::Line => {
            symbols.sort_by(|a, b| {
                a.start_line.cmp(&b.start_line)
                    .then_with(|| a.name.cmp(&b.name))
            });
        }
        SymbolSortOrder::VisibilityThenName => {
            symbols.sort_by(|a, b| {
                // Public first, then by name
                let a_pub = a.visibility == SymbolVisibility::Public;
                let b_pub = b.visibility == SymbolVisibility::Public;
                match (b_pub, a_pub) {
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    _ => a.name.cmp(&b.name),
                }
            });
        }
    }
}

/// Classify symbol kind - delegates to intelligence module
///
/// The intelligence module already determines symbol types based on:
/// - Language-specific tree-sitter queries (scopes.scm)
/// - Namespace system that maps to SymbolKind
///
/// We simply use the pre-classified kind from intel_adapter.
fn classify_symbol_kind(sym: &IntelSymbolInfo, _src: &str) -> SymbolKind {
    // The intelligence module has already done the heavy lifting
    // of language-aware symbol classification via namespace mapping.
    // We just use its result directly.
    sym.kind
}

/// Extract visibility from source context (language-agnostic)
fn extract_visibility(src: &str, location: &IntelSymbolLocation) -> SymbolVisibility {
    let line = src.lines().nth(location.start_line.saturating_sub(1))
        .unwrap_or("");

    SymbolVisibility::from_src_line(line)
}

/// Extract function/method signature from source
fn extract_signature(
    src: &str,
    location: &IntelSymbolLocation,
    kind: &SymbolKind,
) -> Option<String> {
    // Only extract signatures for callable symbols
    if !kind.is_callable() {
        return None;
    }

    let start_line = location.start_line.saturating_sub(1);
    let lines: Vec<&str> = src.lines().collect();

    // Look for function signature (may span multiple lines)
    let mut signature = String::new();

    for (i, line) in lines.iter().enumerate().skip(start_line).take(10) {
        signature.push_str(line);
        signature.push(' ');

        // Stop at opening brace
        if line.contains('{') {
            break;
        }

        // Stop if we hit another top-level declaration
        let trimmed = line.trim();
        if i > start_line && (trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("struct ") || trimmed.starts_with("impl ")) {
            break;
        }
    }

    // Remove opening brace if present
    let sig = if let Some(brace_pos) = signature.find('{') {
        signature[..brace_pos].trim()
    } else {
        signature.trim()
    };

    if sig.is_empty() || sig.len() < 3 {
        None
    } else {
        Some(sig.to_string())
    }
}

/// Extract parent scope name from parsed file
fn extract_parent_scope(_parsed: &ParsedFile, sym: &IntelSymbolInfo) -> Option<String> {
    sym.parent.clone()
}

/// List symbols across multiple files in a repository
///
/// # Arguments
/// * `repo_root` - Root directory of the repository
/// * `options` - Listing options including filters
///
/// # Returns
/// RepositorySymbols containing all found symbols and any parse errors
pub fn list_repository_symbols(
    repo_root: &Path,
    options: Option<SymbolListOptions>,
) -> Result<RepositorySymbols> {
    use crate::intel_adapter::detect_language;

    let mut all_symbols = Vec::new();
    let mut errors = Vec::new();
    let mut files_scanned = 0usize;

    let opts = options.unwrap_or_default();

    // Walk the repository directory
    let mut stack = vec![repo_root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                // Skip common ignore directories
                if let Some(name) = path.file_name() {
                    let name = name.to_string_lossy();
                    if matches!(name.as_ref(), "target" | "node_modules" | "dist" | "build" | ".git" | ".idea" | "vendor") {
                        continue;
                    }
                }
                stack.push(path);
                continue;
            }

            if !file_type.is_file() {
                continue;
            }

            // Check if file is supported
            let _lang_id = match detect_language(&path) {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Check if file size is reasonable
            let metadata = match fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };

            if metadata.len() > 500_000 {
                // Skip files larger than 500KB
                continue;
            }

            files_scanned += 1;

            // Try to parse and extract symbols
            match list_symbols_enhanced(&path, Some(opts.clone())) {
                Ok(mut symbols) => {
                    // Set relative path
                    let rel_path = path.strip_prefix(repo_root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    for sym in &mut symbols {
                        sym.path = rel_path.clone();
                    }

                    all_symbols.extend(symbols);
                }
                Err(e) => {
                    errors.push(FileParseError {
                        path: path.to_string_lossy().to_string(),
                        error: e.to_string(),
                    });
                }
            }
        }
    }

    // Final sort and limit
    sort_symbols(&mut all_symbols, &opts.sort_by);
    if let Some(limit) = opts.limit {
        all_symbols.truncate(limit);
    }

    Ok(RepositorySymbols {
        symbols: all_symbols,
        files_scanned,
        errors,
    })
}

/// Quick search for a specific symbol by name across the repository
///
/// # Arguments
/// * `repo_root` - Root directory to search
/// * `symbol_name` - Name of the symbol to find (exact or partial match)
/// * `kind_filter` - Optional kind filter
///
/// # Returns
/// Vector of matching symbols
pub fn find_symbol(
    repo_root: &Path,
    symbol_name: &str,
    kind_filter: Option<SymbolKind>,
) -> Result<Vec<SymbolDetail>> {
    let filter = SymbolFilter {
        name_pattern: Some(symbol_name.to_string()),
        kinds: kind_filter.map(|k| vec![k]).unwrap_or_default(),
        ..Default::default()
    };

    let options = SymbolListOptions {
        filter,
        sort_by: SymbolSortOrder::Name,
        limit: Some(100), // Reasonable default limit
    };

    let result = list_repository_symbols(repo_root, Some(options))?;
    Ok(result.symbols)
}

/// Get symbols of a specific kind from a file
///
/// # Arguments
/// * `path` - File to analyze
/// * `kind` - Symbol kind to filter for
///
/// # Returns
/// Filtered and sorted symbols
pub fn list_symbols_by_kind(path: &Path, kind: SymbolKind) -> Result<Vec<SymbolDetail>> {
    let filter = SymbolFilter {
        kinds: vec![kind],
        ..Default::default()
    };

    let options = SymbolListOptions {
        filter,
        sort_by: SymbolSortOrder::Line,
        limit: None,
    };

    list_symbols_enhanced(path, Some(options))
}

/// Get all public functions/methods from a file
///
/// Convenience function for common use case
pub fn list_public_functions(path: &Path) -> Result<Vec<SymbolDetail>> {
    let filter = SymbolFilter {
        kinds: vec![SymbolKind::Function, SymbolKind::Method],
        public_only: true,
        ..Default::default()
    };

    let options = SymbolListOptions {
        filter,
        sort_by: SymbolSortOrder::Name,
        limit: None,
    };

    list_symbols_enhanced(path, Some(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_map_by_range_works() {
        let mut uniq: BTreeMap<(String, usize, usize), ContextChunk> = BTreeMap::new();
        let mk = |s, e| ContextChunk {
            path: "a.rs".to_string(),
            alias: 0,
            snippet: "x".to_string(),
            start_line: s,
            end_line: e,
            reason: "r".to_string(),
        };
        for c in [mk(1, 10), mk(1, 10), mk(5, 20)] {
            let key = (c.path.clone(), c.start_line, c.end_line);
            uniq.entry(key).or_insert(c);
        }
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn test_list_symbols_enhanced() {
        // Test with actual tools.rs file since we know it exists and has symbols
        let tools_path = Path::new("src/agent/src/tools.rs");
        if tools_path.exists() {
            let symbols = list_symbols_enhanced(tools_path, None).unwrap();

            // Should find some symbols
            assert!(!symbols.is_empty());

            // Check that we have at least functions
            let has_function = symbols.iter().any(|s| s.kind == SymbolKind::Function);
            assert!(has_function, "Should find at least one function");
        }
    }

    #[test]
    fn test_symbol_filter() {
        let filter = SymbolFilter::new()
            .with_kind(SymbolKind::Function)
            .with_name_pattern("test".to_string());

        assert!(!filter.kinds.is_empty());
        assert_eq!(filter.kinds.len(), 1);
        assert!(filter.name_pattern.is_some());
    }

    #[test]
    fn test_symbol_filter_matches() {
        let filter = SymbolFilter {
            kinds: vec![SymbolKind::Function],
            name_pattern: Some("test".to_string()),
            min_line: None,
            max_line: None,
            public_only: false,
        };

        let func_symbol = SymbolDetail {
            name: "test_function".to_string(),
            kind: SymbolKind::Function,
            path: "test.rs".to_string(),
            start_line: 5,
            end_line: 10,
            start_byte: 0,
            end_byte: 100,
            signature: Some("fn test_function()".to_string()),
            parent_scope: None,
            visibility: SymbolVisibility::Public,
        };

        assert!(filter.matches(&func_symbol));

        // Test non-matching kind
        let struct_symbol = SymbolDetail {
            kind: SymbolKind::Struct,
            ..func_symbol.clone()
        };
        assert!(!filter.matches(&struct_symbol));

        // Test non-matching name
        let other_func = SymbolDetail {
            name: "other_function".to_string(),
            ..func_symbol.clone()
        };
        assert!(!filter.matches(&other_func));
    }

    #[test]
    fn test_symbol_visibility_from_str() {
        assert_eq!(SymbolVisibility::from_src_line("pub fn foo() {}"), SymbolVisibility::Public);
        assert_eq!(SymbolVisibility::from_src_line("public fn foo() {}"), SymbolVisibility::Public);
        assert_eq!(SymbolVisibility::from_src_line("private fn foo() {}"), SymbolVisibility::Private);
        assert_eq!(SymbolVisibility::from_src_line("protected fn foo() {}"), SymbolVisibility::Protected);
        assert_eq!(SymbolVisibility::from_src_line("internal fn foo() {}"), SymbolVisibility::Internal);
        assert_eq!(SymbolVisibility::from_src_line("_internal fn foo() {}"), SymbolVisibility::Protected);
        assert_eq!(SymbolVisibility::from_src_line("fn foo() {}"), SymbolVisibility::Private);
    }

    #[test]
    fn test_symbol_kind_properties() {
        assert!(SymbolKind::Function.is_callable());
        assert!(SymbolKind::Method.is_callable());
        assert!(!SymbolKind::Struct.is_callable());

        assert!(SymbolKind::Struct.is_type());
        assert!(SymbolKind::Enum.is_type());
        assert!(SymbolKind::Class.is_type());
        assert!(!SymbolKind::Function.is_type());

        assert!(SymbolKind::Variable.is_variable());
        assert!(SymbolKind::Const.is_variable());
        assert!(!SymbolKind::Function.is_variable());
    }

    #[test]
    fn test_list_symbols_by_kind_filter() {
        // Test with actual tools.rs file
        let tools_path = Path::new("src/agent/src/tools.rs");
        if tools_path.exists() {
            // Filter for functions only
            let filter = SymbolFilter {
                kinds: vec![SymbolKind::Function],
                ..Default::default()
            };

            let options = SymbolListOptions {
                filter,
                sort_by: SymbolSortOrder::Name,
                limit: None,
            };

            let symbols = list_symbols_enhanced(tools_path, Some(options)).unwrap();

            // Should only contain functions
            for sym in &symbols {
                assert_eq!(sym.kind, SymbolKind::Function);
            }
        }
    }

    #[test]
    fn test_list_symbols_with_limit() {
        // Test with actual tools.rs file
        let tools_path = Path::new("src/agent/src/tools.rs");
        if tools_path.exists() {
            let options = SymbolListOptions {
                filter: SymbolFilter::default(),
                sort_by: SymbolSortOrder::Name,
                limit: Some(5),
            };

            let symbols = list_symbols_enhanced(tools_path, Some(options)).unwrap();

            // Should be limited to 5 results
            assert!(symbols.len() <= 5);
        }
    }

    #[test]
    fn test_symbol_sort_order() {
        let mut symbols = vec![
            SymbolDetail {
                name: "zebra".to_string(),
                kind: SymbolKind::Function,
                path: "a.rs".to_string(),
                start_line: 10,
                end_line: 15,
                start_byte: 100,
                end_byte: 200,
                signature: None,
                parent_scope: None,
                visibility: SymbolVisibility::Private,
            },
            SymbolDetail {
                name: "apple".to_string(),
                kind: SymbolKind::Struct,
                path: "a.rs".to_string(),
                start_line: 1,
                end_line: 5,
                start_byte: 0,
                end_byte: 100,
                signature: None,
                parent_scope: None,
                visibility: SymbolVisibility::Public,
            },
        ];

        // Test sorting by name
        sort_symbols(&mut symbols, &SymbolSortOrder::Name);
        assert_eq!(symbols[0].name, "apple");
        assert_eq!(symbols[1].name, "zebra");

        // Test sorting by line
        sort_symbols(&mut symbols, &SymbolSortOrder::Line);
        assert_eq!(symbols[0].start_line, 1);
        assert_eq!(symbols[1].start_line, 10);
    }

    #[test]
    fn test_list_public_functions() {
        // Test with actual tools.rs file
        let tools_path = Path::new("src/agent/src/tools.rs");
        if tools_path.exists() {
            let result = list_public_functions(tools_path);
            assert!(result.is_ok());

            let symbols = result.unwrap();
            // Should only return public functions
            for sym in &symbols {
                assert!(sym.kind == SymbolKind::Function || sym.kind == SymbolKind::Method);
                assert_eq!(sym.visibility, SymbolVisibility::Public);
            }
        }
    }

    #[test]
    fn test_list_symbols_by_kind() {
        // Test with actual tools.rs file
        let tools_path = Path::new("src/agent/src/tools.rs");
        if tools_path.exists() {
            let result = list_symbols_by_kind(tools_path, SymbolKind::Function);
            assert!(result.is_ok());

            let symbols = result.unwrap();
            // Should only contain functions
            for sym in &symbols {
                assert_eq!(sym.kind, SymbolKind::Function);
            }
        }
    }
}
