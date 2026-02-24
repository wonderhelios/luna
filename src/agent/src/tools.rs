use crate::types::{ContextPack, EditOp, EditResult, TerminalResult, ToolName, ToolTrace};
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
}
