use crate::{detect_lang_id, LunaError, Result, ToolTrace};
use core::code_chunk::{ContextChunk, IndexChunk, RefillOptions};
use index;
use intelligence::{NodeKind, TreeSitterFile};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use super::keyword::is_common_keyword;
use super::symbol::find_symbol_definitions;

/// Refill IndexChunk hits into ContextChunks (function/class-level context)
///
/// This function also performs automatic symbol resolution:
/// - Detects references to symbols not defined in the current context
/// - Looks up definitions across the repository
/// - Adds relevant definitions to the context for better code understanding
pub fn refill_hits(
    repo_root: &Path,
    hits: &[IndexChunk],
    opt: RefillOptions,
) -> Result<(Vec<ContextChunk>, Vec<ToolTrace>)> {
    let mut trace = Vec::new();
    let mut context = Vec::new();

    // Group hits by file
    let mut by_file: BTreeMap<String, Vec<IndexChunk>> = BTreeMap::new();
    for h in hits {
        by_file.entry(h.path.clone()).or_default().push(h.clone());
    }

    for (path, file_hits) in by_file {
        let full_path = repo_root.join(&path);

        // Read file
        let src = fs::read(&full_path)?;
        let lang_id = detect_lang_id(&full_path).unwrap_or("");

        // Refill using index module
        let mut file_context = index::refill_chunks(&path, &src, lang_id, &file_hits, opt.clone())
            .map_err(|e| LunaError::search(format!("refill failed for {}: {:?}", path, e)))?;

        // Perform automatic symbol resolution for this file
        let resolved_context =
            resolve_external_symbols(repo_root, &path, &src, lang_id, &file_context)?;

        context.append(&mut file_context);
        context.extend(resolved_context);
    }

    // Deduplicate by (path, start_line, end_line)
    let mut uniq: BTreeMap<(String, usize, usize), ContextChunk> = BTreeMap::new();
    for c in context {
        let key = (c.path.clone(), c.start_line, c.end_line);
        uniq.entry(key).or_insert(c);
    }

    let context: Vec<_> = uniq.into_values().collect();

    trace.push(ToolTrace {
        tool: "refill_hits".to_string(),
        summary: format!(
            "refilled {} hits into {} context chunks",
            hits.len(),
            context.len()
        ),
    });

    Ok((context, trace))
}

/// Resolve external symbols referenced in the context
///
/// This function:
/// 1. Parses the source file using TreeSitter
/// 2. Builds a ScopeGraph to identify references
/// 3. Finds references to symbols not locally defined
/// 4. Searches the repository for definitions of those symbols
/// 5. Returns ContextChunks for the resolved definitions
fn resolve_external_symbols(
    repo_root: &Path,
    path: &str,
    src: &[u8],
    lang_id: &str,
    _existing_context: &[ContextChunk],
) -> Result<Vec<ContextChunk>> {
    if lang_id.is_empty() {
        return Ok(Vec::new());
    }

    // Build TreeSitterFile and ScopeGraph
    let ts_file = match TreeSitterFile::try_build(src, lang_id) {
        Ok(f) => f,
        Err(_) => return Ok(Vec::new()),
    };

    let scope_graph = match ts_file.scope_graph() {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };

    let src_str = String::from_utf8_lossy(src);

    // Collect all symbol names defined in the current file
    let mut local_definitions: HashSet<String> = HashSet::new();
    for idx in scope_graph.graph.node_indices() {
        if let Some(NodeKind::Def(def)) = scope_graph.get_node(idx) {
            let name = String::from_utf8_lossy(def.name(src_str.as_bytes()));
            local_definitions.insert(name.to_string());
        }
    }

    // Find external references (references to symbols not locally defined)
    let mut external_refs: HashSet<String> = HashSet::new();
    for idx in scope_graph.graph.node_indices() {
        if let Some(NodeKind::Ref(ref_node)) = scope_graph.get_node(idx) {
            let name = String::from_utf8_lossy(ref_node.name(src_str.as_bytes()));
            let name_str = name.to_string();

            // Check if this reference resolves to a local definition
            let is_local = scope_graph.definitions(idx).any(|def_idx| {
                if let Some(NodeKind::Def(def)) = scope_graph.get_node(def_idx) {
                    let def_name = String::from_utf8_lossy(def.name(src_str.as_bytes()));
                    def_name == name.as_ref()
                } else {
                    false
                }
            });

            if !is_local && !local_definitions.contains(&name_str) {
                external_refs.insert(name_str);
            }
        }
    }

    // Limit the number of external symbols to resolve
    let max_symbols = 5;
    let external_refs: Vec<_> = external_refs.into_iter().take(max_symbols).collect();

    if external_refs.is_empty() {
        return Ok(Vec::new());
    }

    // Search for definitions of external symbols
    let mut resolved_chunks = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();

    // Add current file to seen paths to avoid circular references
    seen_paths.insert(path.to_string());

    for symbol_name in external_refs {
        // Skip common keywords and short names
        if symbol_name.len() < 3 || is_common_keyword(&symbol_name, Some(lang_id)) {
            continue;
        }

        // Find symbol definitions
        match find_symbol_definitions(repo_root, &symbol_name, 3) {
            Ok(defs) => {
                for def in defs {
                    // Skip if already in context
                    if seen_paths.contains(&def.path) {
                        continue;
                    }

                    // Read the definition file
                    let def_path = repo_root.join(&def.path);
                    if let Ok(def_src) = fs::read(&def_path) {
                        let def_src_str = String::from_utf8_lossy(&def_src);

                        // Extract the definition snippet
                        let start_line = def.start_line.saturating_sub(1);
                        let end_line = def.end_line + 2; // Include a few lines after
                        let lines: Vec<&str> = def_src_str.lines().collect();

                        if start_line < lines.len() {
                            let snippet =
                                lines[start_line..lines.len().min(end_line)].join("\n");

                            resolved_chunks.push(ContextChunk {
                                path: def.path.clone(),
                                alias: 0, // Will be reassigned later
                                snippet,
                                start_line,
                                end_line: lines.len().min(end_line).saturating_sub(1),
                                reason: format!("definition of '{}' (auto-resolved)", symbol_name),
                            });

                            seen_paths.insert(def.path);
                        }
                    }
                }
            }
            Err(_) => continue,
        }
    }

    Ok(resolved_chunks)
}
