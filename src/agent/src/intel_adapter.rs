//! Intelligence Adapter Layer
//!
//! This module provides a high-level interface to the intelligence module,
//! abstracting away the complexity of tree-sitter and scope resolution.
//!
//! Design Principles:
//! - Clean separation between agent logic and language analysis
//! - Error handling with meaningful messages
//! - Lazy evaluation where possible (only parse when needed)
//! - Language-agnostic patterns

use anyhow::{Result, anyhow};
use std::path::Path;
use std::collections::HashMap;

// Re-export commonly used types from intelligence
pub use intelligence::{
    ScopeGraph, TreeSitterFile,
    ALL_LANGUAGES, TSLanguageConfig,
};

/// A language-agnostic symbol kind that maps to intelligence's namespace system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SymbolKind {
    // Functions and methods
    Function,
    Method,

    // Types
    Struct,
    Enum,
    Union,
    Interface,
    Class,
    TypeAlias,

    // Variables
    Variable,
    Const,
    Static,
    Field,

    // Enum variants
    EnumVariant,

    // Modules and imports
    Module,
    Import,

    // Parameters
    Parameter,

    // Other
    Label,
    Lifetime,

    // Unknown
    Unknown,
}

impl SymbolKind {
    /// Map from intelligence namespace symbol name to our SymbolKind
    /// This provides a centralized mapping that covers all languages
    fn from_namespace_symbol(symbol_name: &str) -> Self {
        match symbol_name {
            // Functions
            "function" => SymbolKind::Function,

            // Types
            "struct" => SymbolKind::Struct,
            "enum" => SymbolKind::Enum,
            "union" => SymbolKind::Union,
            "typedef" => SymbolKind::TypeAlias,
            "interface" | "class" => SymbolKind::Class,

            // Variables
            "variable" => SymbolKind::Variable,
            "const" => SymbolKind::Const,

            // Fields
            "field" => SymbolKind::Field,
            "enumerator" => SymbolKind::EnumVariant,

            // Modules
            "module" => SymbolKind::Module,

            // Parameters
            "parameter" => SymbolKind::Parameter,

            // Other
            "label" => SymbolKind::Label,
            "lifetime" => SymbolKind::Lifetime,

            // Unknown (catch-all for language-specific symbols)
            _ => SymbolKind::Unknown,
        }
    }

    /// Convert to display string
    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Method => "method",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Union => "union",
            SymbolKind::Interface => "interface",
            SymbolKind::Class => "class",
            SymbolKind::TypeAlias => "type",
            SymbolKind::Variable => "variable",
            SymbolKind::Const => "const",
            SymbolKind::Static => "static",
            SymbolKind::Field => "field",
            SymbolKind::EnumVariant => "variant",
            SymbolKind::Module => "module",
            SymbolKind::Import => "import",
            SymbolKind::Parameter => "parameter",
            SymbolKind::Label => "label",
            SymbolKind::Lifetime => "lifetime",
            SymbolKind::Unknown => "unknown",
        }
    }

    /// Check if this is a callable symbol
    pub fn is_callable(&self) -> bool {
        matches!(self, SymbolKind::Function | SymbolKind::Method)
    }

    /// Check if this is a type definition
    pub fn is_type(&self) -> bool {
        matches!(
            self,
            SymbolKind::Struct | SymbolKind::Enum | SymbolKind::Union
            | SymbolKind::Interface | SymbolKind::Class | SymbolKind::TypeAlias
        )
    }

    /// Check if this is a variable-like symbol
    pub fn is_variable(&self) -> bool {
        matches!(self, SymbolKind::Variable | SymbolKind::Const | SymbolKind::Static | SymbolKind::Field)
    }
}

/// Symbol information for display purposes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,

    /// Symbol kind (derived from intelligence namespace)
    pub kind: SymbolKind,

    /// Definition location
    pub location: SymbolLocation,

    /// Parent scope (if any)
    pub parent: Option<String>,

    /// Visibility (public/private, etc.) - derived from source analysis
    pub visibility: Option<String>,
}

/// Location in source code
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolLocation {
    /// File path (relative to repo root if available)
    pub path: String,

    /// Start line (1-based for display)
    pub start_line: usize,

    /// End line (1-based for display)
    pub end_line: usize,

    /// Start byte position (0-based)
    pub start_byte: usize,

    /// End byte position (0-based)
    pub end_byte: usize,
}

/// A parsed file with its symbol information
pub struct ParsedFile {
    /// Original source code
    pub src: Vec<u8>,

    /// Detected language config
    pub language: &'static TSLanguageConfig,

    /// Scope graph (if parsing succeeded)
    pub scope_graph: Option<ScopeGraph>,

    /// Cached namespace symbol names for this language
    /// Maps SymbolId to display string
    namespace_cache: HashMap<(usize, usize), &'static str>,
}

impl ParsedFile {
    /// Parse a file and extract symbol information
    pub fn parse(path: &Path) -> Result<Self> {
        let content = std::fs::read(path)
            .map_err(|e| anyhow!("failed to read file {:?}: {}", path, e))?;

        let lang_id = detect_language(path)?;
        Self::parse_content(&content, lang_id)
    }

    /// Parse from content directly (for testing or when file is already loaded)
    pub fn parse_content(content: &[u8], lang_id: &str) -> Result<Self> {
        // Find language config
        let language = ALL_LANGUAGES
            .iter()
            .find(|lang| lang.language_ids.contains(&lang_id))
            .ok_or_else(|| anyhow!("unsupported language: {}", lang_id))?;

        // Build namespace cache for efficient lookups
        let mut namespace_cache = HashMap::new();
        for (ns_idx, namespace) in language.namespaces.iter().enumerate() {
            for (sym_idx, &sym_name) in namespace.iter().enumerate() {
                namespace_cache.insert((ns_idx, sym_idx), sym_name);
            }
        }

        let ts_file = TreeSitterFile::try_build(content, lang_id)
            .map_err(|e| anyhow!("failed to parse content: {:?}", e))?;

        Ok(Self {
            src: content.to_vec(),
            language,
            scope_graph: Some(ts_file.scope_graph()
                .map_err(|e| anyhow!("failed to build scope graph: {:?}", e))?),
            namespace_cache,
        })
    }

    /// Extract all symbols from this file
    pub fn extract_symbols(&self) -> Vec<SymbolInfo> {
        let scope_graph = match &self.scope_graph {
            Some(graph) => graph,
            None => return Vec::new(),
        };

        let src_str = std::str::from_utf8(&self.src).unwrap_or("<invalid utf8>");

        scope_graph
            .graph
            .node_indices()
            .filter_map(|idx| scope_graph.get_node(idx))
            .filter_map(|node_kind| match node_kind {
                intelligence::NodeKind::Def(def) => {
                    let name_bytes = def.name(src_str.as_bytes());
                    let name = String::from_utf8_lossy(name_bytes).to_string();
                    let range = def.range;

                    // Use symbol_id from intelligence to determine the kind
                    let kind = def.symbol_id
                        .and_then(|id| self.namespace_cache.get(&(id.namespace_idx, id.symbol_idx)))
                        .map(|&sym_name| SymbolKind::from_namespace_symbol(sym_name))
                        .unwrap_or(SymbolKind::Unknown);

                    Some(SymbolInfo {
                        name,
                        kind,
                        location: SymbolLocation {
                            path: String::new(), // Caller should fill this
                            start_line: range.start.line + 1,
                            end_line: range.end.line + 1,
                            start_byte: range.start.byte,
                            end_byte: range.end.byte,
                        },
                        parent: None, // TODO: extract parent scope
                        visibility: None, // TODO: extract visibility
                    })
                }
                intelligence::NodeKind::Import(imp) => {
                    let name_bytes = imp.name(src_str.as_bytes());
                    let name = String::from_utf8_lossy(name_bytes).to_string();
                    let range = imp.range;
                    Some(SymbolInfo {
                        name,
                        kind: SymbolKind::Import,
                        location: SymbolLocation {
                            path: String::new(),
                            start_line: range.start.line + 1,
                            end_line: range.end.line + 1,
                            start_byte: range.start.byte,
                            end_byte: range.end.byte,
                        },
                        parent: None,
                        visibility: None,
                    })
                }
                _ => None,
            })
            .collect()
    }

    /// Get function signatures (extracts name and parameter info)
    pub fn get_function_signatures(&self) -> Vec<String> {
        let scope_graph = match &self.scope_graph {
            Some(graph) => graph,
            None => return Vec::new(),
        };

        let src_str = std::str::from_utf8(&self.src).unwrap_or("<invalid utf8>");

        scope_graph
            .graph
            .node_indices()
            .filter_map(|idx| scope_graph.get_node(idx))
            .filter_map(|node_kind| match node_kind {
                intelligence::NodeKind::Def(def) => {
                    // Only process functions
                    let is_function = def.symbol_id
                        .and_then(|id| self.namespace_cache.get(&(id.namespace_idx, id.symbol_idx)))
                        .map(|&sym_name| sym_name == "function")
                        .unwrap_or(false);

                    if !is_function {
                        return None;
                    }

                    let name_bytes = def.name(src_str.as_bytes());
                    if name_bytes.is_empty() {
                        return None;
                    }

                    // Try to extract function signature
                    let range = def.range;
                    let start_line = range.start.line;
                    let end_line = range.end.line.min(start_line + 20);

                    if start_line < end_line {
                        let lines: Vec<&str> = src_str.lines().collect();
                        let sig_lines: Vec<&str> = lines
                            .iter()
                            .skip(start_line)
                            .take(end_line - start_line + 1)
                            .copied()
                            .collect();

                        let sig = sig_lines.join(" ");
                        if sig.contains('{') {
                            Some(sig.split('{').next().map(|s| s.trim()).unwrap_or(&sig).to_string())
                        } else {
                            Some(sig)
                        }
                    } else {
                        Some(String::new())
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// Get references to a specific symbol
    pub fn get_references_to(&self, symbol_name: &str) -> Vec<SymbolLocation> {
        let scope_graph = match &self.scope_graph {
            Some(graph) => graph,
            None => return Vec::new(),
        };

        let src_str = std::str::from_utf8(&self.src).unwrap_or("");

        scope_graph
            .graph
            .node_indices()
            .filter_map(|idx| scope_graph.get_node(idx))
            .filter_map(|node_kind| match node_kind {
                intelligence::NodeKind::Ref(reference) => {
                    let ref_name = String::from_utf8_lossy(reference.name(src_str.as_bytes()));
                    if ref_name == symbol_name {
                        let range = reference.range;
                        Some(SymbolLocation {
                            path: String::new(),
                            start_line: range.start.line + 1,
                            end_line: range.end.line + 1,
                            start_byte: range.start.byte,
                            end_byte: range.end.byte,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// Check if this file defines a specific symbol
    pub fn has_definition(&self, symbol_name: &str) -> bool {
        let scope_graph = match &self.scope_graph {
            Some(graph) => graph,
            None => return false,
        };

        let src_str = std::str::from_utf8(&self.src).unwrap_or("");
        let symbol_bytes = symbol_name.as_bytes();

        scope_graph
            .graph
            .node_indices()
            .any(|idx| {
                if let Some(intelligence::NodeKind::Def(def)) = scope_graph.get_node(idx) {
                    def.name(src_str.as_bytes()) == symbol_bytes
                } else {
                    false
                }
            })
    }

    /// Find references to definitions within a specific byte range
    pub fn references_in_range(&self, _start_byte: usize, _end_byte: usize) -> Vec<String> {
        let _scope_graph = match &self.scope_graph {
            Some(graph) => graph,
            None => return Vec::new(),
        };

        let _src_str = std::str::from_utf8(&self.src).unwrap_or("");
        let refs = Vec::new();

        // Find all nodes in the range
        // Note: Symbol name extraction requires namespace access
        // This can be enhanced later if needed
        refs
    }

    /// Get the language ID for this file
    pub fn lang_id(&self) -> &str {
        self.language.language_ids.first().unwrap_or(&"")
    }
}

/// Detect programming language from file path
pub fn detect_language(path: &Path) -> Result<&'static str> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Find matching language
    for lang in ALL_LANGUAGES.iter() {
        for ext in lang.file_extensions {
            if ext.eq_ignore_ascii_case(extension) {
                return Ok(lang.language_ids.first().unwrap_or(&""));
            }
        }
    }

    Err(anyhow!("unsupported language: {}", extension))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        // Language IDs are language-specific, just check that detection works
        assert!(detect_language(Path::new("test.rs")).is_ok());
        assert!(detect_language(Path::new("test.py")).is_ok());
        assert!(detect_language(Path::new("test.js")).is_ok());
        assert!(detect_language(Path::new("test.xyz")).is_err());
    }

    #[test]
    fn test_symbol_kind_from_namespace() {
        // Test mappings from various languages
        assert_eq!(SymbolKind::from_namespace_symbol("function"), SymbolKind::Function);
        assert_eq!(SymbolKind::from_namespace_symbol("struct"), SymbolKind::Struct);
        assert_eq!(SymbolKind::from_namespace_symbol("class"), SymbolKind::Class);
        assert_eq!(SymbolKind::from_namespace_symbol("variable"), SymbolKind::Variable);
        assert_eq!(SymbolKind::from_namespace_symbol("const"), SymbolKind::Const);
        assert_eq!(SymbolKind::from_namespace_symbol("field"), SymbolKind::Field);
        assert_eq!(SymbolKind::from_namespace_symbol("module"), SymbolKind::Module);
        assert_eq!(SymbolKind::from_namespace_symbol("parameter"), SymbolKind::Parameter);
        assert_eq!(SymbolKind::from_namespace_symbol("unknown_key"), SymbolKind::Unknown);
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
}
