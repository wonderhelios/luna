//! ContextQuery: Query types for retrieving context

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Query for retrieving context from repository
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextQuery {
    /// Exact symbol query (uses ScopeGraph)
    Symbol { name: String },

    /// Position-based query (file:line)
    Position { path: PathBuf, line: usize },

    /// Semantic concept query (will use vector search in Phase 4.2)
    Concept { description: String },

    /// File-level query
    File { path: PathBuf },

    /// Task-driven comprehensive query
    TaskDriven {
        /// Keywords extracted from task
        keywords: Vec<String>,
        /// File paths mentioned
        paths: Vec<PathBuf>,
        /// Symbols mentioned
        symbols: Vec<String>,
    },

    /// Related symbols query (find callers/callees)
    Related {
        base_symbol: String,
        relation: SymbolRelation,
    },
}

/// Type of symbol relationship to query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolRelation {
    /// Symbols that call this symbol
    Callers,
    /// Symbols that this symbol calls
    Callees,
    /// Symbols defined in the same scope
    Siblings,
    /// Parent scope symbols
    Parents,
    /// Child scope symbols
    Children,
}

impl ContextQuery {
    /// Create a symbol query
    #[must_use]
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol { name: name.into() }
    }

    /// Create a position query
    #[must_use]
    pub fn position(path: impl Into<PathBuf>, line: usize) -> Self {
        Self::Position {
            path: path.into(),
            line,
        }
    }

    /// Create a file query
    #[must_use]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File { path: path.into() }
    }

    /// Create a concept query
    #[must_use]
    pub fn concept(description: impl Into<String>) -> Self {
        Self::Concept {
            description: description.into(),
        }
    }

    /// Create a task-driven query
    #[must_use]
    pub fn task_driven(
        keywords: Vec<String>,
        paths: Vec<PathBuf>,
        symbols: Vec<String>,
    ) -> Self {
        Self::TaskDriven {
            keywords,
            paths,
            symbols,
        }
    }

    /// Create a related symbols query
    #[must_use]
    pub fn related(base_symbol: impl Into<String>, relation: SymbolRelation) -> Self {
        Self::Related {
            base_symbol: base_symbol.into(),
            relation,
        }
    }

    /// Check if this query is for a specific symbol
    #[must_use]
    pub fn is_symbol_query(&self) -> bool {
        matches!(self, Self::Symbol { .. })
    }

    /// Get the symbol name if this is a symbol query
    #[must_use]
    pub fn symbol_name(&self) -> Option<&str> {
        match self {
            Self::Symbol { name } => Some(name),
            _ => None,
        }
    }

    /// Get all file paths mentioned in this query
    #[must_use]
    pub fn mentioned_paths(&self) -> Vec<&PathBuf> {
        match self {
            Self::Position { path, .. } => vec![path],
            Self::File { path } => vec![path],
            Self::TaskDriven { paths, .. } => paths.iter().collect(),
            _ => vec![],
        }
    }

    /// Estimate complexity of this query
    #[must_use]
    pub fn complexity(&self) -> QueryComplexity {
        match self {
            // Symbol queries are fastest (ScopeGraph indexed)
            Self::Symbol { .. } => QueryComplexity::Fast,
            // Position queries require file lookup
            Self::Position { .. } => QueryComplexity::Fast,
            // File queries are simple
            Self::File { .. } => QueryComplexity::Fast,
            // Concept queries are slowest (vector search)
            Self::Concept { .. } => QueryComplexity::Slow,
            // Task-driven depends on components
            Self::TaskDriven { symbols, paths, .. } => {
                let component_count = symbols.len() + paths.len();
                if component_count <= 2 {
                    QueryComplexity::Medium
                } else {
                    QueryComplexity::Slow
                }
            }
            // Related queries require graph traversal
            Self::Related { .. } => QueryComplexity::Medium,
        }
    }
}

/// Query complexity hint for planning
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryComplexity {
    /// < 10ms expected
    Fast,
    /// 10-100ms expected
    Medium,
    /// 100ms+ expected
    Slow,
}

/// Query result with metadata
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The query that was executed
    pub query: ContextQuery,
    /// Number of chunks found
    pub chunk_count: usize,
    /// Execution time in milliseconds
    pub elapsed_ms: u64,
    /// Whether the query was served from cache
    pub from_cache: bool,
}

impl QueryResult {
    #[must_use]
    pub fn new(query: ContextQuery, chunk_count: usize, elapsed_ms: u64) -> Self {
        Self {
            query,
            chunk_count,
            elapsed_ms,
            from_cache: false,
        }
    }

    pub fn cached(mut self) -> Self {
        self.from_cache = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_query() {
        let query = ContextQuery::symbol("foo");
        assert!(query.is_symbol_query());
        assert_eq!(query.symbol_name(), Some("foo"));
        assert_eq!(query.complexity(), QueryComplexity::Fast);
    }

    #[test]
    fn test_position_query() {
        let query = ContextQuery::position("src/lib.rs", 42);
        assert!(!query.is_symbol_query());
        assert_eq!(query.mentioned_paths().len(), 1);
        assert_eq!(query.complexity(), QueryComplexity::Fast);
    }

    #[test]
    fn test_task_driven_query() {
        let query = ContextQuery::task_driven(
            vec!["auth".to_string(), "login".to_string()],
            vec![PathBuf::from("src/auth.rs")],
            vec!["authenticate".to_string()],
        );

        assert_eq!(query.mentioned_paths().len(), 1);
        assert_eq!(query.complexity(), QueryComplexity::Medium);
    }

    #[test]
    fn test_complex_task_driven() {
        let query = ContextQuery::task_driven(
            vec!["a".to_string(), "b".to_string()],
            vec![
                PathBuf::from("src/a.rs"),
                PathBuf::from("src/b.rs"),
                PathBuf::from("src/c.rs"),
            ],
            vec!["x".to_string(), "y".to_string()],
        );

        assert_eq!(query.complexity(), QueryComplexity::Slow);
    }
}
