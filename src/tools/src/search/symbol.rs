use crate::{detect_lang_id, Result};
use intelligence::TreeSitterFile;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Symbol location for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolLocation {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: String,
}

/// Find definitions of a symbol name across the repository
pub fn find_symbol_definitions(
    repo_root: &Path,
    symbol_name: &str,
    max_results: usize,
) -> Result<Vec<SymbolLocation>> {
    let mut results = Vec::new();

    for entry in walkdir::WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let path = e.path();
            if path.is_dir() {
                let name = e.file_name().to_string_lossy();
                return !matches!(
                    name.as_ref(),
                    "target" | "node_modules" | ".git" | "dist" | "build"
                );
            }
            path.is_file() && detect_lang_id(path).is_some()
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();

        // Skip directories and non-files
        if !path.is_file() {
            continue;
        }

        if results.len() >= max_results {
            break;
        }

        let src = match std::fs::read(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let lang_id = detect_lang_id(path).unwrap_or("");

        let ts_file = match TreeSitterFile::try_build(&src, lang_id) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let scope_graph = match ts_file.scope_graph() {
            Ok(g) => g,
            Err(_) => continue,
        };

        let src_str = String::from_utf8_lossy(&src);

        for idx in scope_graph.graph.node_indices() {
            if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
                let name = String::from_utf8_lossy(def.name(src_str.as_bytes()));
                if name == symbol_name {
                    results.push(SymbolLocation {
                        path: path
                            .strip_prefix(repo_root)
                            .unwrap_or(path)
                            .to_string_lossy()
                            .to_string(),
                        start_line: def.range.start.line + 1,
                        end_line: def.range.end.line + 1,
                        kind: "definition".to_string(),
                    });
                }
            }
        }
    }

    Ok(results)
}
