pub mod core_local;
pub mod document;
pub mod language;
pub mod namespace;
pub mod navigation;
pub mod repo_scan;
pub mod scope_resolution;
pub mod snippet;

// Re-exports for internal use
pub use language::{MemoizedQuery, TSLanguageConfig, ALL_LANGUAGES};
pub use namespace::{NameSpaces, NameSpaceMethods};

use scope_resolution::{ResolutionMethod, ScopeGraph};

/// Errors that can occur when working with TreeSitter files
#[derive(Debug)]
pub enum TreeSitterFileError {
    ParseError(String),
    LanguageNotSupported(String),
    ScopeGraphError(String),
}

impl std::fmt::Display for TreeSitterFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::LanguageNotSupported(lang) => write!(f, "language not supported: {lang}"),
            Self::ScopeGraphError(msg) => write!(f, "scope graph error: {msg}"),
        }
    }
}

impl std::error::Error for TreeSitterFileError {}

/// A parsed source file with tree-sitter and scope graph
pub struct TreeSitterFile {
    src: Vec<u8>,
    language: tree_sitter::Language,
    scope_graph: ScopeGraph,
}

impl TreeSitterFile {
    /// Try to build a TreeSitterFile from source bytes
    pub fn try_build(src: &[u8], lang_id: &str) -> Result<Self, TreeSitterFileError> {
        // Find language config
        let lang_config = ALL_LANGUAGES
            .iter()
            .find(|l| l.language_ids.contains(&lang_id))
            .ok_or_else(|| TreeSitterFileError::LanguageNotSupported(lang_id.to_string()))?;

        let language = (lang_config.grammar)();

        // Parse the source
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(language)
            .map_err(|e| TreeSitterFileError::ParseError(e.to_string()))?;

        let tree = parser
            .parse(src, None)
            .ok_or_else(|| TreeSitterFileError::ParseError("failed to parse".to_string()))?;

        // Build scope graph
        let query = lang_config
            .scope_query
            .query(lang_config.grammar)
            .map_err(|e| TreeSitterFileError::ScopeGraphError(e.to_string()))?;

        let scope_graph = ResolutionMethod::Generic.build_scope(
            query,
            tree.root_node(),
            src,
            lang_config,
        );

        Ok(Self {
            src: src.to_vec(),
            language,
            scope_graph,
        })
    }

    /// Get the scope graph
    pub fn scope_graph(&self) -> Result<&ScopeGraph, TreeSitterFileError> {
        Ok(&self.scope_graph)
    }

    /// Get the source bytes
    pub fn src(&self) -> &[u8] {
        &self.src
    }

    /// Get the language
    pub fn language(&self) -> tree_sitter::Language {
        self.language
    }
}
