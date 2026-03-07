//! Tools implementation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

use error::ResultExt as _;

#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Repository root used to resolve relative paths.
    pub repo_root: Option<PathBuf>,
    /// Current working directory.
    pub cwd: Option<PathBuf>,
    /// Hard output limit for commands and file reads.
    pub max_bytes: usize,
}

impl ToolContext {
    #[must_use]
    pub fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            return path.to_path_buf();
        }
        if let Some(repo_root) = &self.repo_root {
            return repo_root.join(path);
        }
        if let Some(cwd) = &self.cwd {
            return cwd.join(path);
        }
        path.to_path_buf()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub ok: bool,
    pub stdout: String,
    pub stderr: String,
}

impl ToolResult {
    #[must_use]
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self {
            ok: true,
            stdout: stdout.into(),
            stderr: String::new(),
        }
    }

    #[must_use]
    pub fn err(stderr: impl Into<String>) -> Self {
        Self {
            ok: false,
            stdout: String::new(),
            stderr: stderr.into(),
        }
    }
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, ctx: &ToolContext, args: &Value) -> error::Result<ToolResult>;
}

#[derive(Default)]
pub struct ToolRegistry {
    read_file: ReadFileTool,
    edit_file: EditFileTool,
    run_terminal: RunTerminalTool,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&self, ctx: &ToolContext, call: &ToolCall) -> error::Result<ToolResult> {
        match call.name.as_str() {
            "read_file" => self.read_file.run(ctx, &call.args),
            "edit_file" => self.edit_file.run(ctx, &call.args),
            "run_terminal" => self.run_terminal.run(ctx, &call.args),
            _ => Ok(ToolResult::err(format!("unknown tool: {}", call.name))),
        }
    }
}

#[derive(Default)]
struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn run(&self, ctx: &ToolContext, args: &Value) -> error::Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| error::LunaError::invalid_input("read_file missing args.path"))?;
        let abs = ctx.resolve_path(Path::new(path));
        let bytes = std::fs::read(&abs)
            .map_err(|e| error::LunaError::io(Some(abs.clone()), e))
            .with_context(|| format!("read file: {}", abs.display()))?;
        let limited = if bytes.len() > ctx.max_bytes {
            &bytes[..ctx.max_bytes]
        } else {
            &bytes
        };
        let s = String::from_utf8_lossy(limited).to_string();
        Ok(ToolResult::ok(s))
    }
}

#[derive(Default)]
struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn run(&self, ctx: &ToolContext, args: &Value) -> error::Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| error::LunaError::invalid_input("edit_file missing args.path"))?;
        let abs = ctx.resolve_path(Path::new(path));

        let mut content = std::fs::read_to_string(&abs)
            .map_err(|e| error::LunaError::io(Some(abs.clone()), e))
            .with_context(|| format!("read file for edit: {}", abs.display()))?;
        let had_trailing_newline = content.ends_with('\n');
        let mut lines = content.lines().map(ToOwned::to_owned).collect::<Vec<_>>();

        // Supported shapes:
        // 1) { path, line_1, new_line }
        // 2) { path, start_line_1, end_line_1, replace_with }
        if let (Some(line_1), Some(new_line)) = (
            args.get("line_1").and_then(|v| v.as_u64()),
            args.get("new_line").and_then(|v| v.as_str()),
        ) {
            let idx = usize::try_from(line_1).ok().and_then(|v| v.checked_sub(1));
            let Some(i) = idx else {
                return Ok(ToolResult::err("edit_file invalid line_1"));
            };
            if i >= lines.len() {
                return Ok(ToolResult::err(format!(
                    "edit_file line out of range: {line_1} > {}",
                    lines.len()
                )));
            }
            lines[i] = new_line.to_owned();
        } else if let (Some(start), Some(end), Some(replace_with)) = (
            args.get("start_line_1").and_then(|v| v.as_u64()),
            args.get("end_line_1").and_then(|v| v.as_u64()),
            args.get("replace_with").and_then(|v| v.as_str()),
        ) {
            let start0 = usize::try_from(start).ok().and_then(|v| v.checked_sub(1));
            let end0 = usize::try_from(end).ok().and_then(|v| v.checked_sub(1));
            let (Some(s0), Some(e0)) = (start0, end0) else {
                return Ok(ToolResult::err("edit_file invalid line range"));
            };
            if s0 > e0 || e0 >= lines.len() {
                return Ok(ToolResult::err("edit_file range out of bounds"));
            }
            let repl_lines = replace_with
                .lines()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            lines.splice(s0..=e0, repl_lines);
        } else {
            return Ok(ToolResult::err(
                "edit_file missing args: provide (line_1,new_line) or (start_line_1,end_line_1,replace_with)",
            ));
        }

        content = lines.join("\n");
        // Preserve trailing newline if the original had it.
        if had_trailing_newline {
            content.push('\n');
        }
        std::fs::write(&abs, content)
            .map_err(|e| error::LunaError::io(Some(abs.clone()), e))
            .with_context(|| format!("write edited file: {}", abs.display()))?;
        Ok(ToolResult::ok(format!("edited: {}", abs.display())))
    }
}

#[derive(Default)]
struct RunTerminalTool;

impl Tool for RunTerminalTool {
    fn name(&self) -> &'static str {
        "run_terminal"
    }

    fn run(&self, ctx: &ToolContext, args: &Value) -> error::Result<ToolResult> {
        let cmd = args
            .get("cmd")
            .and_then(|v| v.as_str())
            .ok_or_else(|| error::LunaError::invalid_input("run_terminal missing args.cmd"))?;
        let cwd = args
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .or_else(|| ctx.cwd.clone())
            .or_else(|| ctx.repo_root.clone());

        let mut command = std::process::Command::new("sh");
        command.arg("-lc").arg(cmd);
        if let Some(dir) = cwd {
            command.current_dir(dir);
        }
        let out = command
            .output()
            .map_err(error::LunaError::from)
            .context("run terminal")?;

        let mut stdout = out.stdout;
        let mut stderr = out.stderr;
        if stdout.len() > ctx.max_bytes {
            stdout.truncate(ctx.max_bytes);
        }
        if stderr.len() > ctx.max_bytes {
            stderr.truncate(ctx.max_bytes);
        }

        Ok(ToolResult {
            ok: out.status.success(),
            stdout: String::from_utf8_lossy(&stdout).to_string(),
            stderr: String::from_utf8_lossy(&stderr).to_string(),
        })
    }
}

// NOTE: `ToolContext::resolve_path` is the canonical helper.
