use std::path::{Path, PathBuf};

use error::{Result, ResultExt as _};

use intelligence::{Navigator, SnippetOptions, TreeSitterNavigator};

use crate::{
    intent, render,
    response::{EventSink, RuntimeEvent},
};

/// Runtime router decides which subsystem should handle the user input.
///
#[derive(Debug, Default)]
pub struct RuntimeRouter {
    symbol_nav: SymbolNavigationRouter,
}

impl RuntimeRouter {
    pub fn maybe_handle(
        &self,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut dyn EventSink,
    ) -> Result<Option<String>> {
        match intent::classify_intent(user_input) {
            intent::Intent::SymbolNavigation => {
                Ok(Some(self.symbol_nav.handle(user_input, cwd, events)?))
            }
            intent::Intent::ExplainSymbol => Ok(Some(
                self.symbol_nav.handle_explain(user_input, cwd, events)?,
            )),
            intent::Intent::Other => Ok(None),
        }
    }
}

#[derive(Debug, Default)]
struct SymbolNavigationRouter {
    navigator: TreeSitterNavigator<intelligence::repo_scan::FsRepoFileProvider>,
    snippet_opt: SnippetOptions,
}

impl SymbolNavigationRouter {
    fn handle(
        &self,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut dyn EventSink,
    ) -> Result<String> {
        self.handle_multi(user_input, cwd, events, render::RenderStyle::Navigation)
    }

    fn handle_explain(
        &self,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut dyn EventSink,
    ) -> Result<String> {
        self.handle_multi(user_input, cwd, events, render::RenderStyle::Explain)
    }

    fn handle_multi(
        &self,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut dyn EventSink,
        style: render::RenderStyle,
    ) -> Result<String> {
        // Position-based go-to-definition: <path>:<line>[:<col>]
        if let Some((path, line, col)) = intent::extract_file_position(user_input) {
            return self.handle_position(path, line, col, cwd, events, style);
        }

        let mut names = intent::extract_identifiers_dedup(user_input);
        if names.is_empty() {
            let header = render::render_multi_header(&[]);
            return Ok(format!(
                "{header}{}",
                render::render_symbol_navigation_missing_identifier()
            ));
        }
        // Keep it bounded to avoid overly long outputs for generic names.
        names.truncate(3);

        let Some(repo_root) = resolve_repo_root(cwd) else {
            // If multiple identifiers exist, show the first in the error to keep message concise.
            let header = render::render_multi_header(&names);
            return Ok(format!(
                "{header}{}",
                render::render_symbol_navigation_missing_repo_root(names[0])
            ));
        };

        let mut out = render::render_multi_header(&names);

        for (idx, name) in names.iter().enumerate() {
            events.emit(&RuntimeEvent::FoundIdentifier {
                name: (*name).to_owned(),
            });
            events.emit(&RuntimeEvent::ScopeGraphSearchStarted {
                repo_root: repo_root.display().to_string(),
            });

            let definitions = match self.navigator.goto_definition(&repo_root, name) {
                Ok(v) => v,
                Err(err) => {
                    let err = error::LunaError::invalid_input(err.to_string());
                    out.push_str(&render::render_symbol_navigation_search_failed(name, &err));
                    out.push('\n');
                    continue;
                }
            };
            events.emit(&RuntimeEvent::ScopeGraphSearchCompleted {
                matches: definitions.len(),
            });

            if definitions.is_empty() {
                out.push_str(&render::render_symbol_navigation_not_found(name));
                out.push('\n');
                continue;
            }

            let primary = &definitions[0];
            let ctx = self
                .navigator
                .get_symbol_context(&repo_root, primary, &self.snippet_opt)
                .map_err(|e| error::LunaError::invalid_input(e.to_string()))
                .with_context(|| {
                    format!(
                        "get symbol context for {}:{}",
                        primary.rel_path.display(),
                        primary.range.start.line + 1
                    )
                });

            let other_candidates = definitions.get(1..).unwrap_or_default();

            let references = self
                .navigator
                .find_references(&repo_root, name, 30)
                .unwrap_or_default();

            // Filter out definition locations themselves.
            let references = references
                .into_iter()
                .filter(|r| {
                    !definitions
                        .iter()
                        .any(|d| d.rel_path == r.rel_path && d.range == r.range)
                })
                .collect::<Vec<_>>();

            let section = match style {
                render::RenderStyle::Navigation => render::render_symbol_navigation_success(
                    name,
                    primary,
                    ctx,
                    other_candidates,
                    &references,
                ),
                render::RenderStyle::Explain => render::render_symbol_explain_success(
                    name,
                    primary,
                    ctx,
                    other_candidates,
                    &references,
                ),
            };

            out.push_str(&section);
            if idx + 1 < names.len() {
                out.push('\n');
            }
        }

        Ok(out)
    }

    fn handle_position(
        &self,
        path: PathBuf,
        line: usize,
        col: usize,
        cwd: Option<&Path>,
        events: &mut dyn EventSink,
        style: render::RenderStyle,
    ) -> Result<String> {
        let Some(repo_root) = resolve_repo_root(cwd) else {
            let header = render::render_multi_header(&["<position>"]);
            return Ok(format!(
                "{header}{}",
                render::render_symbol_navigation_missing_repo_root("<position>")
            ));
        };

        let abs_path = if path.is_absolute() {
            path
        } else {
            repo_root.join(path)
        };

        let rel_path = match abs_path.strip_prefix(&repo_root) {
            Ok(p) => p.to_path_buf(),
            Err(_) => {
                let header = render::render_multi_header(&["<position>"]);
                return Ok(format!(
                    "{header}❌ File is not under repo_root: {}\nrepo_root: {}",
                    abs_path.display(),
                    repo_root.display()
                ));
            }
        };

        events.emit(&RuntimeEvent::ScopeGraphSearchStarted {
            repo_root: repo_root.display().to_string(),
        });

        let definitions = match self
            .navigator
            .goto_definition_at(&repo_root, &rel_path, line, col)
        {
            Ok(v) => v,
            Err(err) => {
                let err = error::LunaError::invalid_input(err.to_string());
                let header = render::render_multi_header(&["<position>"]);
                return Ok(format!(
                    "{header}{}",
                    render::render_symbol_navigation_search_failed("<position>", &err)
                ));
            }
        };

        events.emit(&RuntimeEvent::ScopeGraphSearchCompleted {
            matches: definitions.len(),
        });

        let Some(primary) = definitions.first() else {
            let header = render::render_multi_header(&["<position>"]);
            return Ok(format!(
                "{header}{}",
                render::render_symbol_navigation_not_found("<position>")
            ));
        };

        let ctx = self
            .navigator
            .get_symbol_context(&repo_root, primary, &self.snippet_opt)
            .map_err(|e| error::LunaError::invalid_input(e.to_string()));

        let name = std::fs::read_to_string(repo_root.join(&rel_path))
            .ok()
            .and_then(|c| identifier_at_position(&c, line, col))
            .unwrap_or_else(|| "<position>".to_owned());
        let other_candidates = definitions.get(1..).unwrap_or_default();

        let mut references = self
            .navigator
            .find_references(&repo_root, &name, 30)
            .unwrap_or_default();
        references.retain(|r| {
            !definitions
                .iter()
                .any(|d| d.rel_path == r.rel_path && d.range == r.range)
        });

        events.emit(&RuntimeEvent::FoundIdentifier { name: name.clone() });

        let section = match style {
            render::RenderStyle::Navigation => render::render_symbol_navigation_success(
                &name,
                primary,
                ctx,
                other_candidates,
                &references,
            ),
            render::RenderStyle::Explain => render::render_symbol_explain_success(
                &name,
                primary,
                ctx,
                other_candidates,
                &references,
            ),
        };

        let header = render::render_multi_header(&[name.as_str()]);
        Ok(format!("{header}{section}"))
    }
}

fn identifier_at_position(content: &str, line: usize, column: usize) -> Option<String> {
    let line_str = content.lines().nth(line)?;
    if line_str.is_empty() {
        return None;
    }
    let bytes = line_str.as_bytes();
    let mut idx = column.min(bytes.len().saturating_sub(1));
    // If the cursor is on a non-identifier char, try stepping left once.
    if !is_ident_continue(bytes[idx]) {
        if idx == 0 {
            return None;
        }
        idx -= 1;
        if !is_ident_continue(bytes[idx]) {
            return None;
        }
    }

    let mut start = idx;
    while start > 0 && is_ident_continue(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = idx + 1;
    while end < bytes.len() && is_ident_continue(bytes[end]) {
        end += 1;
    }
    Some(line_str[start..end].to_owned())
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

pub(crate) fn resolve_repo_root(cwd: Option<&Path>) -> Option<PathBuf> {
    let mut cur = cwd?;
    if cur.is_file() {
        cur = cur.parent()?;
    }

    loop {
        if cur.join(".git").exists() {
            return Some(cur.to_path_buf());
        }
        cur = cur.parent()?;
    }
}
