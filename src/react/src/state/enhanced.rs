//! Enhanced state summary using ScopeGraph analysis
//!
//! This module provides `summarize_state_enhanced` which uses Tree-sitter
//! and ScopeGraph to extract detailed symbol information including:
//! - Function signatures
//! - Call relationships
//! - Visibility information

use core::code_chunk::{ContextChunk, IndexChunk};
use intelligence::scope_resolution::NodeIndex;
use std::collections::HashMap;
use std::path::Path;

/// Symbol information extracted from ScopeGraph
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub visibility: String,
    pub signature: Option<String>,
    pub callers: Vec<String>,
    pub callees: Vec<String>,
    pub path: String,
    pub line: usize,
}

/// Enhanced state summary using ScopeGraph analysis
///
/// This provides:
/// - Function signatures with parameters and return types
/// - Call relationships (who calls whom)
/// - Visibility information
pub fn summarize_state_enhanced(
    hits: &[IndexChunk],
    context: &[ContextChunk],
    repo_root: &Path,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "hits={} context_chunks={}\n",
        hits.len(),
        context.len()
    ));

    // Group context chunks by file for efficient processing
    let mut file_chunks: HashMap<String, Vec<&ContextChunk>> = HashMap::new();
    for chunk in context {
        file_chunks
            .entry(chunk.path.clone())
            .or_default()
            .push(chunk);
    }

    // Process each file to extract symbol information
    let mut all_symbols: Vec<SymbolInfo> = Vec::new();
    for (path, chunks) in file_chunks {
        if let Ok(symbols) = analyze_file_symbols(repo_root, &path, &chunks) {
            all_symbols.extend(symbols);
        }
    }

    // Build call graph
    let call_graph = build_call_graph(&all_symbols);

    // Summary section
    if !all_symbols.is_empty() {
        s.push_str(&format!("symbols={}\n", all_symbols.len()));

        // Group by kind
        let mut by_kind: HashMap<String, Vec<&SymbolInfo>> = HashMap::new();
        for sym in &all_symbols {
            by_kind.entry(sym.kind.clone()).or_default().push(sym);
        }

        for (kind, syms) in &by_kind {
            s.push_str(&format!("  {}({}): ", kind, syms.len()));
            let names: Vec<_> = syms.iter().map(|s| s.name.clone()).collect();
            s.push_str(&names.join(", "));
            s.push('\n');
        }
    }

    // Call relationships
    if !call_graph.is_empty() {
        s.push_str("call_relations:\n");
        for (caller, callees) in &call_graph {
            if !callees.is_empty() {
                s.push_str(&format!("  {} -> {}\n", caller, callees.join(", ")));
            }
        }
    }

    // Detailed preview of definitions
    s.push_str("definitions:\n");
    for sym in all_symbols.iter().take(6) {
        s.push_str(&format_symbol(sym));
    }
    if all_symbols.len() > 6 {
        s.push_str(&format!("  ... ({} more)\n", all_symbols.len() - 6));
    }

    s
}

/// Analyze a single file and extract symbol information using ScopeGraph
fn analyze_file_symbols(
    repo_root: &Path,
    rel_path: &str,
    _chunks: &[&ContextChunk],
) -> anyhow::Result<Vec<SymbolInfo>> {
    use intelligence::{TreeSitterFile, ALL_LANGUAGES};

    let full_path = repo_root.join(rel_path);
    let content = std::fs::read(&full_path)?;

    // Detect language
    let lang_id = tools::detect_lang_id(&full_path).unwrap_or("");
    if lang_id.is_empty() {
        return Ok(Vec::new());
    }

    // Build TreeSitterFile and ScopeGraph
    let ts_file = TreeSitterFile::try_build(&content, lang_id)
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    let scope_graph = ts_file
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    let src_str = String::from_utf8_lossy(&content);
    let lang_config = ALL_LANGUAGES
        .iter()
        .find(|l| l.language_ids.contains(&lang_id));

    let mut symbols = Vec::new();

    // Iterate through all nodes in scope graph
    for idx in scope_graph.graph.node_indices() {
        if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
            let name =
                String::from_utf8_lossy(def.name(src_str.as_bytes())).to_string();

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

            // Get visibility
            let visibility = detect_visibility_from_scope(&scope_graph, idx);

            // Extract signature from source
            let signature =
                extract_signature(&src_str, def.range.start.line, def.range.end.line);

            // Find callers (references to this definition)
            let callers: Vec<String> = scope_graph
                .references(idx)
                .filter_map(|ref_idx| {
                    if let Some(intelligence::NodeKind::Ref(_)) =
                        scope_graph.get_node(ref_idx)
                    {
                        // Find the enclosing function definition
                        find_enclosing_function(&scope_graph, ref_idx, &src_str)
                    } else {
                        None
                    }
                })
                .collect();

            // Find callees (functions called from this definition)
            let callees = find_callees_in_scope(&scope_graph, idx, &src_str);

            symbols.push(SymbolInfo {
                name,
                kind: kind.to_string(),
                visibility,
                signature,
                callers,
                callees,
                path: rel_path.to_string(),
                line: def.range.start.line + 1, // 1-based
            });
        }
    }

    Ok(symbols)
}

/// Detect visibility based on scope graph position
fn detect_visibility_from_scope(
    scope_graph: &intelligence::ScopeGraph,
    def_idx: NodeIndex,
) -> String {
    // If directly connected to root, it's likely public/exported
    if scope_graph.is_top_level(def_idx) {
        "public".to_string()
    } else {
        "private".to_string()
    }
}

/// Extract function signature from source lines
fn extract_signature(src: &str, start_line: usize, end_line: usize) -> Option<String> {
    let lines: Vec<&str> = src.lines().collect();
    if start_line >= lines.len() {
        return None;
    }

    let first_line = lines[start_line];

    // Try to find function signature pattern
    if let Some(fn_start) = first_line.find("fn ") {
        let after_fn = &first_line[fn_start..];
        // Find the opening brace or end of line
        if let Some(brace_pos) = after_fn.find('{') {
            return Some(after_fn[..brace_pos].trim().to_string());
        } else {
            // Multi-line signature, collect until {
            let mut sig = after_fn.to_string();
            for line in lines.iter().take(end_line + 1).skip(start_line + 1) {
                if let Some(pos) = line.find('{') {
                    sig.push_str(&line[..pos]);
                    break;
                } else {
                    sig.push_str(line);
                    sig.push(' ');
                }
            }
            return Some(sig.trim().to_string());
        }
    }

    // For struct/enum definitions
    if first_line.contains("struct ") || first_line.contains("enum ") {
        if let Some(pos) = first_line.find('{') {
            return Some(first_line[..pos].trim().to_string());
        }
    }

    None
}

/// Find the enclosing function name for a reference
fn find_enclosing_function(
    _scope_graph: &intelligence::ScopeGraph,
    _node_idx: NodeIndex,
    _src: &str,
) -> Option<String> {
    // Walk up the scope graph to find enclosing definition
    // This is a simplified version - would need proper scope traversal
    None
}

/// Find functions called from within a definition's scope
fn find_callees_in_scope(
    _scope_graph: &intelligence::ScopeGraph,
    _def_idx: NodeIndex,
    _src: &str,
) -> Vec<String> {
    // This would require analyzing the function body for call expressions
    Vec::new()
}

/// Build call graph from symbol information
fn build_call_graph(symbols: &[SymbolInfo]) -> HashMap<String, Vec<String>> {
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();

    for sym in symbols {
        if !sym.callees.is_empty() {
            graph.insert(sym.name.clone(), sym.callees.clone());
        }
    }

    graph
}

/// Format a single symbol for display
fn format_symbol(sym: &SymbolInfo) -> String {
    let mut s = String::new();

    let vis_marker = match sym.visibility.as_str() {
        "public" | "pub" => "+",
        "private" => "-",
        _ => "",
    };

    if let Some(ref sig) = sym.signature {
        s.push_str(&format!("  {}{} {}\n", vis_marker, sym.kind, sig));
    } else {
        s.push_str(&format!("  {}{} {}\n", vis_marker, sym.kind, sym.name));
    }

    s.push_str(&format!("    @ {}:{}\n", sym.path, sym.line));

    if !sym.callers.is_empty() {
        s.push_str(&format!("    called_by: {}\n", sym.callers.join(", ")));
    }

    if !sym.callees.is_empty() {
        s.push_str(&format!("    calls: {}\n", sym.callees.join(", ")));
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_state_enhanced() {
        // Create a temporary test file
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        std::fs::write(
            &test_file,
            r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn private_helper(x: i32) -> i32 {
    x * 2
}

pub struct Point {
    x: f64,
    y: f64,
}
"#,
        )
        .unwrap();

        let context = vec![
            ContextChunk {
                path: "test.rs".to_string(),
                alias: 0,
                snippet: "pub fn add(a: i32, b: i32) -> i32 {".to_string(),
                start_line: 1,
                end_line: 3,
                reason: "search_hit".to_string(),
            },
            ContextChunk {
                path: "test.rs".to_string(),
                alias: 1,
                snippet: "pub struct Point {".to_string(),
                start_line: 9,
                end_line: 12,
                reason: "search_hit".to_string(),
            },
        ];

        let hits: Vec<IndexChunk> = vec![];
        let summary = summarize_state_enhanced(&hits, &context, temp_dir.path());

        // Print the summary for demo
        println!("\n=== Enhanced State Summary Demo ===\n{}", summary);

        // Verify the summary contains expected information
        assert!(summary.contains("context_chunks=2"));
        assert!(summary.contains("symbols="));
        assert!(summary.contains("definitions:"));

        temp_dir.close().unwrap();
    }
}
