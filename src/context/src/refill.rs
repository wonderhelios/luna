//! RefillPipeline: IndexChunk → ContextChunk conversion engine
//!
//! Core responsibilities:
//! 1. Retrieve: Fast recall of candidate chunks using ScopeGraph + file scanning
//! 2. Refine: Deduplication, sorting, truncation, symbol signature injection
//! 3. Refill: Dynamic context supplementation based on LLM feedback
//!
//! ## Refill Pipeline Flow
//!
//! ```text
//! Query
//!   │
//!   ▼
//! ┌──────────────┐
//! │  Retrieve    │ ◄── Uses ScopeGraph for symbol queries
//! │              │ ◄── Uses file provider for path queries
//! └──────┬───────┘
//!        │ Vec<IndexChunk>
//!        ▼
//! ┌──────────────┐
//! │   Refine     │ ◄── Deduplicate by symbol
//! │              │ ◄── Sort by relevance
//! │              │ ◄── Truncate to token budget
//! │              │ ◄── Inject signatures
//! └──────┬───────┘
//!        │ Vec<ContextChunk>
//!        ▼
//! ┌──────────────┐
//! │ Build String │ ◄── Format for LLM prompt
//! └──────────────┘
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use error::ResultExt;

use crate::{
    cache::ContextCache,
    context_chunk::{ContextChunk, ContextType},
    index_chunk::IndexChunk,
    query::{ContextQuery, SymbolRelation},
    ChunkId, LanguageId, SourceLocation, SymbolId, TextRange, TokenBudget,
};

/// File provider trait for reading repository files
///
/// This mirrors intelligence::RepoFileProvider but is defined here
/// to avoid tight coupling.
pub trait FileProvider: Send + Sync {
    /// List all source files in the repository
    fn list_files(&self, repo_root: &Path) -> error::Result<Vec<PathBuf>>;

    /// Read file content
    fn read_file(&self, path: &Path) -> error::Result<String>;

    /// Get file modification time
    fn modified_time(&self, path: &Path) -> error::Result<u64>;
}

/// Symbol resolver trait for ScopeGraph-based navigation
///
/// Abstracts over intelligence::Navigator to avoid direct dependency.
pub trait SymbolResolver: Send + Sync {
    /// Find symbol definition locations
    fn find_definition(&self, repo_root: &Path, name: &str) -> error::Result<Vec<SourceLocation>>;

    /// Find symbol references
    fn find_references(
        &self,
        repo_root: &Path,
        name: &str,
        max: usize,
    ) -> error::Result<Vec<SourceLocation>>;

    /// Get symbol signature at location
    fn get_signature(&self, repo_root: &Path, location: &SourceLocation)
        -> error::Result<Option<String>>;

    /// Get snippet around location
    fn get_snippet(
        &self,
        repo_root: &Path,
        location: &SourceLocation,
        context_lines: usize,
    ) -> error::Result<String>;
}

/// RefillPipeline: The core context transformation engine
pub struct RefillPipeline {
    repo_root: PathBuf,
    file_provider: Arc<dyn FileProvider>,
    symbol_resolver: Arc<dyn SymbolResolver>,
    budget: TokenBudget,
    cache: ContextCache,
}

impl RefillPipeline {
    /// Create a new RefillPipeline
    pub fn new(
        repo_root: PathBuf,
        file_provider: Arc<dyn FileProvider>,
        symbol_resolver: Arc<dyn SymbolResolver>,
        budget: TokenBudget,
    ) -> Self {
        Self {
            repo_root,
            file_provider,
            symbol_resolver,
            budget,
            cache: ContextCache::with_default_size(),
        }
    }

    /// Get repository root
    #[must_use]
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }
}

/// Retrieve phase: Fast recall of candidate chunks
impl RefillPipeline {
    /// Retrieve IndexChunks based on query
    ///
    /// This is the "coarse" retrieval phase - aim for high recall,
    /// even if precision is lower. We'll refine in the next phase.
    pub fn retrieve(&self, query: &ContextQuery, top_k: usize) -> error::Result<Vec<IndexChunk>> {
        // Check cache first
        if let Some(cached) = self.cache.get_cached_query(query) {
            return Ok(cached);
        }

        let start = std::time::Instant::now();
        let chunks = match query {
            ContextQuery::Symbol { name } => self.retrieve_symbol(name, top_k),
            ContextQuery::Position { path, line } => self.retrieve_position(path, *line, top_k),
            ContextQuery::Concept { description } => self.retrieve_concept(description, top_k),
            ContextQuery::File { path } => self.retrieve_file(path),
            ContextQuery::TaskDriven {
                keywords,
                paths,
                symbols,
            } => self.retrieve_task_driven(keywords, paths, symbols, top_k),
            ContextQuery::Related {
                base_symbol,
                relation,
            } => self.retrieve_related(base_symbol, *relation, top_k),
        }?;

        // Cache results
        let chunk_ids: Vec<ChunkId> = chunks.iter().map(|c| c.id).collect();
        self.cache.store_batch(chunks.clone());
        self.cache.cache_query_result(query, chunk_ids);

        let elapsed = start.elapsed();
        tracing::debug!(
            "Retrieved {} chunks for query {:?} in {:?}",
            chunks.len(),
            query,
            elapsed
        );

        Ok(chunks)
    }

    /// Retrieve chunks for a symbol query (uses ScopeGraph)
    fn retrieve_symbol(&self, name: &str, top_k: usize) -> error::Result<Vec<IndexChunk>> {
        let mut chunks = Vec::new();

        // 1. Get definition
        let defs = self
            .symbol_resolver
            .find_definition(&self.repo_root, name)
            .with_context(|| format!("find definition for {}", name))?;

        for loc in defs {
            let signature = self
                .symbol_resolver
                .get_signature(&self.repo_root, &loc)
                .ok()
                .flatten();

            let snippet = self
                .symbol_resolver
                .get_snippet(&self.repo_root, &loc, 5)
                .unwrap_or_default();

            let content = if let Some(sig) = signature {
                format!("{}\n{}", sig, snippet)
            } else {
                snippet
            };

            let lang = detect_language(&loc.rel_path);
            let mut chunk =
                IndexChunk::symbol_definition(content, loc, SymbolId::new(name, ""));
            chunk.language = lang;
            chunks.push(chunk);
        }

        // 2. Get references (if we have room)
        if chunks.len() < top_k {
            let refs = self
                .symbol_resolver
                .find_references(&self.repo_root, name, top_k - chunks.len())
                .unwrap_or_default();

            for loc in refs {
                let snippet = self
                    .symbol_resolver
                    .get_snippet(&self.repo_root, &loc, 3)
                    .unwrap_or_default();

                let _lang = detect_language(&loc.rel_path);
                let chunk =
                    IndexChunk::new(snippet, loc, crate::IndexChunkType::SymbolReference);
                chunks.push(chunk);
            }
        }

        Ok(chunks.into_iter().take(top_k).collect())
    }

    /// Retrieve chunks for a position query
    fn retrieve_position(
        &self,
        path: &Path,
        line: usize,
        top_k: usize,
    ) -> error::Result<Vec<IndexChunk>> {
        let abs_path = self.repo_root.join(path);
        let content = self.file_provider.read_file(&abs_path)?;

        // Extract snippet around the line
        let lines: Vec<&str> = content.lines().collect();
        let start = line.saturating_sub(5);
        let end = (line + 5).min(lines.len());

        let snippet = lines[start..end].join("\n");

        let source = SourceLocation {
            repo_root: self.repo_root.clone(),
            rel_path: path.to_path_buf(),
            range: TextRange::new(start + 1, end),
        };

        let lang = detect_language(path);
        let chunk = IndexChunk::new(snippet, source, crate::IndexChunkType::CodeBlock);

        // Also get file summary if it's a small file
        let mut chunks = vec![chunk];
        if lines.len() < 100 && top_k > 1 {
            let summary = format!(
                "// File: {} ({} lines)\n{}",
                path.display(),
                lines.len(),
                lines[..lines.len().min(20)].join("\n")
            );
            let source = SourceLocation {
                repo_root: self.repo_root.clone(),
                rel_path: path.to_path_buf(),
                range: TextRange::new(1, lines.len().min(20)),
            };
            let summary_chunk =
                IndexChunk::file_summary(summary, source, lang);
            chunks.push(summary_chunk);
        }

        Ok(chunks)
    }

    /// Retrieve chunks for a concept query (placeholder for Phase 4.2)
    fn retrieve_concept(
        &self,
        _description: &str,
        top_k: usize,
    ) -> error::Result<Vec<IndexChunk>> {
        // Phase 4.2: Use vector search
        // For now, return empty (will trigger fallback behavior)
        tracing::warn!("Concept queries not yet implemented (Phase 4.2)");
        Ok(Vec::with_capacity(top_k))
    }

    /// Retrieve chunks for a file query
    fn retrieve_file(&self, path: &Path) -> error::Result<Vec<IndexChunk>> {
        let abs_path = self.repo_root.join(path);
        let content = self.file_provider.read_file(&abs_path)?;
        let lines: Vec<&str> = content.lines().collect();

        let lang = detect_language(path);
        let source = SourceLocation {
            repo_root: self.repo_root.clone(),
            rel_path: path.to_path_buf(),
            range: TextRange::new(1, lines.len()),
        };

        // For large files, just take the first N lines as summary
        let summary_lines = lines.len().min(50);
        let summary = lines[..summary_lines].join("\n");

        let chunk = IndexChunk::file_summary(summary, source, lang);
        Ok(vec![chunk])
    }

    /// Retrieve chunks for a task-driven query
    fn retrieve_task_driven(
        &self,
        _keywords: &[String],
        paths: &[PathBuf],
        symbols: &[String],
        top_k: usize,
    ) -> error::Result<Vec<IndexChunk>> {
        let mut chunks = Vec::new();
        let mut seen_symbols: HashSet<String> = HashSet::new();

        // 1. Retrieve mentioned symbols
        for symbol in symbols {
            if seen_symbols.contains(symbol) {
                continue;
            }
            seen_symbols.insert(symbol.clone());

            match self.retrieve_symbol(symbol, 3) {
                Ok(mut sym_chunks) => chunks.append(&mut sym_chunks),
                Err(e) => tracing::warn!("Failed to retrieve symbol {}: {}", symbol, e),
            }
        }

        // 2. Retrieve mentioned files
        for path in paths {
            match self.retrieve_file(path) {
                Ok(mut file_chunks) => chunks.append(&mut file_chunks),
                Err(e) => tracing::warn!("Failed to retrieve file {:?}: {}", path, e),
            }
        }

        Ok(chunks.into_iter().take(top_k).collect())
    }

    /// Retrieve related symbols (callers, callees, etc.)
    fn retrieve_related(
        &self,
        base_symbol: &str,
        relation: SymbolRelation,
        top_k: usize,
    ) -> error::Result<Vec<IndexChunk>> {
        // Get base symbol definition first
        let base_locs = self
            .symbol_resolver
            .find_definition(&self.repo_root, base_symbol)?;

        if base_locs.is_empty() {
            return Ok(Vec::new());
        }

        // Handle different relation types
        match relation {
            SymbolRelation::Callers => {
                // Find symbols that reference the base symbol
                let refs = self
                    .symbol_resolver
                    .find_references(&self.repo_root, base_symbol, top_k * 2)?;

                // Return reference locations as chunks
                Ok(refs
                    .into_iter()
                    .take(top_k)
                    .filter_map(|loc| {
                        let snippet = self
                            .symbol_resolver
                            .get_snippet(&self.repo_root, &loc, 3)
                            .ok()?;
                        Some(IndexChunk::new(
                            snippet,
                            loc,
                            crate::index_chunk::IndexChunkType::SymbolReference,
                        ))
                    })
                    .collect())
            }
            _ => {
                // Other relations require full ScopeGraph traversal
                // For now, return empty
                tracing::warn!("Relation {:?} not fully implemented", relation);
                Ok(Vec::new())
            }
        }
    }
}

/// Refine phase: Convert IndexChunk to ContextChunk
impl RefillPipeline {
    /// Refine IndexChunks into ContextChunks
    ///
    /// Processing steps:
    /// 1. Deduplicate by symbol (same symbol in multiple places)
    /// 2. Sort by relevance score
    /// 3. Truncate to token budget (keep highest relevance)
    /// 4. Inject symbol signatures
    pub fn refine(&self, chunks: &[IndexChunk]) -> Vec<ContextChunk> {
        // 1. Deduplicate by primary symbol
        let mut seen_symbols: HashSet<SymbolId> = HashSet::new();
        let mut unique_chunks: Vec<&IndexChunk> = Vec::new();

        for chunk in chunks {
            if let Some(symbol) = chunk.primary_symbol() {
                if seen_symbols.contains(symbol) {
                    continue;
                }
                seen_symbols.insert(symbol.clone());
            }
            unique_chunks.push(chunk);
        }

        // 2. Convert to ContextChunks with relevance scores
        let mut context_chunks: Vec<ContextChunk> = unique_chunks
            .into_iter()
            .map(|ic| self.index_to_context(ic))
            .collect();

        // 3. Sort by relevance (highest first)
        context_chunks.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Truncate to token budget
        self.truncate_to_budget(&mut context_chunks);

        context_chunks
    }

    /// Convert IndexChunk to ContextChunk
    fn index_to_context(&self, index: &IndexChunk) -> ContextChunk {
        let context_type = match index.chunk_type {
            crate::IndexChunkType::FileSummary => ContextType::FileOverview,
            crate::IndexChunkType::SymbolDefinition => ContextType::NavigationResult,
            crate::IndexChunkType::SymbolReference => ContextType::RelatedSymbol,
            crate::IndexChunkType::CodeBlock => ContextType::CodeSnippet,
            crate::IndexChunkType::Documentation => ContextType::Documentation,
        };

        // Calculate relevance score
        let relevance = calculate_relevance(index);

        let mut chunk = ContextChunk::new(index.content.clone(), index.source.clone(), context_type);
        chunk.set_relevance(relevance);

        // Inject symbol signatures
        for symbol in &index.symbols {
            chunk.add_signature(symbol.full_name());
        }

        // Try to get signature from symbol resolver for definitions
        if index.chunk_type == crate::IndexChunkType::SymbolDefinition {
            if let Ok(Some(sig)) = self
                .symbol_resolver
                .get_signature(&self.repo_root, &index.source)
            {
                chunk.add_signature(sig);
            }
        }

        chunk
    }

    /// Truncate chunks to fit token budget
    fn truncate_to_budget(&self, chunks: &mut Vec<ContextChunk>) {
        let mut total_tokens: usize = 0;
        let mut keep_count = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            total_tokens += chunk.token_count;
            if total_tokens > self.budget.max_context_tokens {
                keep_count = i;
                break;
            }
        }

        chunks.truncate(keep_count);

        // Check if we need to truncate the last chunk
        if let Some(last) = chunks.last() {
            let total_used: usize = chunks.iter().map(|c| c.token_count).sum();

            if total_used > self.budget.max_context_tokens {
                // Calculate how much we can keep of the last chunk
                let tokens_before_last = total_used - last.token_count;
                let remaining = self.budget.max_context_tokens.saturating_sub(tokens_before_last);

                if remaining > 0 {
                    // Get mutable reference to last and truncate it
                    if let Some(last_mut) = chunks.last_mut() {
                        last_mut.truncate_to_tokens(remaining);
                    }
                } else {
                    // Remove the last chunk entirely
                    chunks.pop();
                }
            }
        }
    }
}

/// Refill phase: Dynamic context supplementation
impl RefillPipeline {
    /// Refill context with additional chunks based on missing symbols
    ///
    /// Called when LLM indicates it needs more context (e.g., "see also X").
    pub fn refill(
        &self,
        current: &[ContextChunk],
        missing_symbols: &[SymbolId],
    ) -> error::Result<Vec<ContextChunk>> {
        let mut new_chunks = Vec::new();

        for symbol in missing_symbols {
            // Check if already in current
            if current.iter().any(|c| {
                c.symbol_signatures
                    .iter()
                    .any(|sig| sig.contains(&symbol.name))
            }) {
                continue;
            }

            // Retrieve symbol
            match self.retrieve(&ContextQuery::symbol(&symbol.name), 1) {
                Ok(index_chunks) => {
                    let context_chunks = self.refine(&index_chunks);
                    new_chunks.extend(context_chunks);
                }
                Err(e) => {
                    tracing::warn!("Failed to refill symbol {}: {}", symbol.name, e);
                }
            }
        }

        Ok(new_chunks)
    }

    /// Build context string for LLM prompt
    pub fn build_context_string(&self, chunks: &[ContextChunk]) -> String {
        if chunks.is_empty() {
            return String::new();
        }

        let mut output = String::from("## Relevant Code Context\n\n");

        for chunk in chunks {
            output.push_str(&chunk.format_for_prompt());
            output.push('\n');
        }

        output.push_str("## End Context\n");
        output
    }
}

/// Helper functions
fn detect_language(path: &Path) -> LanguageId {
    path.extension()
        .and_then(|e| e.to_str())
        .map(LanguageId::from_extension)
        .unwrap_or(LanguageId::Unknown)
}

fn calculate_relevance(index: &IndexChunk) -> f32 {
    let mut score = 0.5; // Base score

    // Definitions are more relevant than references
    match index.chunk_type {
        crate::IndexChunkType::SymbolDefinition => score += 0.3,
        crate::IndexChunkType::FileSummary => score += 0.1,
        crate::IndexChunkType::Documentation => score += 0.1,
        _ => {}
    }

    // Chunks with symbols are more relevant
    if !index.symbols.is_empty() {
        score += 0.1 * index.symbols.len().min(3) as f32;
    }

    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Mock FileProvider for testing
    struct MockFileProvider {
        files: Mutex<HashMap<PathBuf, String>>,
    }

    impl MockFileProvider {
        fn new() -> Self {
            Self {
                files: Mutex::new(HashMap::new()),
            }
        }

        fn add_file(&self, path: PathBuf, content: String) {
            self.files.lock().unwrap().insert(path, content);
        }
    }

    impl FileProvider for MockFileProvider {
        fn list_files(&self, _repo_root: &Path) -> error::Result<Vec<PathBuf>> {
            Ok(self.files.lock().unwrap().keys().cloned().collect())
        }

        fn read_file(&self, path: &Path) -> error::Result<String> {
            self.files
                .lock()
                .unwrap()
                .get(path)
                .cloned()
                .ok_or_else(|| error::LunaError::not_found(format!("file not found: {:?}", path)))
        }

        fn modified_time(&self, _path: &Path) -> error::Result<u64> {
            Ok(0)
        }
    }

    // Mock SymbolResolver for testing
    struct MockSymbolResolver;

    impl SymbolResolver for MockSymbolResolver {
        fn find_definition(&self, _repo_root: &Path, name: &str) -> error::Result<Vec<SourceLocation>> {
            Ok(vec![SourceLocation {
                repo_root: PathBuf::from("/repo"),
                rel_path: PathBuf::from(format!("src/{}.rs", name)),
                range: TextRange::new(1, 10),
            }])
        }

        fn find_references(
            &self,
            _repo_root: &Path,
            _name: &str,
            max: usize,
        ) -> error::Result<Vec<SourceLocation>> {
            Ok(Vec::new()) // Simplified
        }

        fn get_signature(
            &self,
            _repo_root: &Path,
            _location: &SourceLocation,
        ) -> error::Result<Option<String>> {
            Ok(Some("fn mock() -> i32".to_string()))
        }

        fn get_snippet(
            &self,
            _repo_root: &Path,
            _location: &SourceLocation,
            _context_lines: usize,
        ) -> error::Result<String> {
            Ok("fn mock() -> i32 { 42 }".to_string())
        }
    }

    fn create_test_pipeline() -> RefillPipeline {
        let file_provider = Arc::new(MockFileProvider::new());
        let symbol_resolver = Arc::new(MockSymbolResolver);

        RefillPipeline::new(
            PathBuf::from("/repo"),
            file_provider,
            symbol_resolver,
            TokenBudget {
                max_context_tokens: 1000,
            },
        )
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("foo.rs")), LanguageId::Rust);
        assert_eq!(detect_language(Path::new("foo.py")), LanguageId::Python);
        assert_eq!(detect_language(Path::new("foo.unknown")), LanguageId::Unknown);
    }

    #[test]
    fn test_calculate_relevance() {
        let source = SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(1, 5),
        };

        let def_chunk = IndexChunk::symbol_definition("fn foo() {}", source.clone(), SymbolId::new("foo", ""));
        assert!(calculate_relevance(&def_chunk) > 0.5);

        let ref_chunk = IndexChunk::new("foo()", source.clone(), crate::IndexChunkType::SymbolReference);
        assert_eq!(calculate_relevance(&ref_chunk), 0.5);
    }

    #[test]
    fn test_refine_deduplication() {
        let pipeline = create_test_pipeline();

        let source = SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(1, 5),
        };

        // Create chunks with same symbol
        let symbol = SymbolId::new("foo", "");
        let chunk1 = IndexChunk::symbol_definition("fn foo() {}", source.clone(), symbol.clone());
        let chunk2 = IndexChunk::symbol_definition("fn foo() { }", source.clone(), symbol.clone());

        let refined = pipeline.refine(&[chunk1, chunk2]);
        assert_eq!(refined.len(), 1); // Deduplicated
    }

    #[test]
    fn test_truncate_to_budget() {
        let pipeline = create_test_pipeline();

        let source = SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(1, 5),
        };

        // Create many chunks with larger content to exceed budget
        let mut chunks = Vec::new();
        for i in 0..100 {
            // Each chunk is ~100 chars = 25 tokens, 100 chunks = 2500 tokens
            let chunk = ContextChunk::new(
                format!("fn func{}() {{}} // {} long comment to increase token count", i, "x".repeat(80)),
                source.clone(),
                ContextType::CodeSnippet,
            );
            chunks.push(chunk);
        }

        pipeline.truncate_to_budget(&mut chunks);
        assert!(chunks.len() < 100, "Expected truncation, but got {} chunks", chunks.len());

        // Verify total tokens within budget
        let total: usize = chunks.iter().map(|c| c.token_count).sum();
        assert!(total <= 1000);
    }

    #[test]
    fn test_build_context_string() {
        let pipeline = create_test_pipeline();

        let source = SourceLocation {
            repo_root: PathBuf::from("/repo"),
            rel_path: PathBuf::from("src/lib.rs"),
            range: TextRange::new(10, 15),
        };

        let chunk = ContextChunk::navigation_result(
            "pub fn find_main() {}",
            source,
            "fn find_main()",
        );

        let context_str = pipeline.build_context_string(&[chunk]);
        assert!(context_str.contains("Relevant Code Context"));
        assert!(context_str.contains("src/lib.rs:10-15"));
        assert!(context_str.contains("fn find_main()"));
    }
}
