//! Bridge between intelligence crate and context crate
//!
//! Implements FileProvider and SymbolResolver traits for RefillPipeline
//! using the existing intelligence::Navigator infrastructure.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use context::{
    refill::{FileProvider, SymbolResolver},
    SourceLocation, TextRange,
};
use intelligence::{
    repo_scan::{FsRepoFileProvider, RepoFileProvider, RepoScanOptions},
    Navigator, SnippetOptions, SymbolLocation as IntelSymbolLocation,
};
use _core::text_range::{Position, TextRange as CoreTextRange};

/// Adapter for intelligence crate to implement context::FileProvider
pub struct IntelligenceFileProvider<N: Navigator> {
    navigator: Arc<N>,
}

impl<N: Navigator> IntelligenceFileProvider<N> {
    pub fn new(navigator: Arc<N>) -> Self {
        Self { navigator }
    }
}

impl<N: Navigator + Send + Sync> FileProvider for IntelligenceFileProvider<N> {
    fn list_files(&self, repo_root: &Path) -> error::Result<Vec<PathBuf>> {
        // Use intelligence's default file listing
        let scan_opt = RepoScanOptions::default();
        let provider = FsRepoFileProvider;
        let files = provider
            .list_files(repo_root, &scan_opt)
            .map_err(|e| error::LunaError::io(Some(repo_root.to_path_buf()), std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(files.into_iter().map(|f| f.rel_path).collect())
    }

    fn read_file(&self, path: &Path) -> error::Result<String> {
        std::fs::read_to_string(path)
            .map_err(|e| error::LunaError::io(Some(path.to_path_buf()), e))
    }

    fn modified_time(&self, path: &Path) -> error::Result<u64> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| error::LunaError::io(Some(path.to_path_buf()), e))?;
        let modified = metadata
            .modified()
            .map_err(|e| error::LunaError::io(Some(path.to_path_buf()), e))?;
        let duration = modified
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Ok(duration.as_millis() as u64)
    }
}

/// Adapter for intelligence crate to implement context::SymbolResolver
pub struct IntelligenceSymbolResolver<N: Navigator> {
    navigator: Arc<N>,
}

impl<N: Navigator> IntelligenceSymbolResolver<N> {
    pub fn new(navigator: Arc<N>) -> Self {
        Self { navigator }
    }
}

impl<N: Navigator + Send + Sync> SymbolResolver for IntelligenceSymbolResolver<N> {
    fn find_definition(
        &self,
        repo_root: &Path,
        name: &str,
    ) -> error::Result<Vec<SourceLocation>> {
        let locations = self
            .navigator
            .goto_definition(repo_root, name)
            .map_err(|e| error::LunaError::invalid_input(format!("navigation error: {e}")))?;

        Ok(locations
            .into_iter()
            .map(|loc| intel_location_to_source_location(repo_root, loc))
            .collect())
    }

    fn find_references(
        &self,
        repo_root: &Path,
        name: &str,
        max: usize,
    ) -> error::Result<Vec<SourceLocation>> {
        let locations = self
            .navigator
            .find_references(repo_root, name, max)
            .map_err(|e| error::LunaError::invalid_input(format!("navigation error: {e}")))?;

        Ok(locations
            .into_iter()
            .map(|loc| intel_location_to_source_location(repo_root, loc))
            .collect())
    }

    fn get_signature(
        &self,
        repo_root: &Path,
        location: &SourceLocation,
    ) -> error::Result<Option<String>> {
        let intel_loc = source_location_to_intel_location(location);
        let opt = SnippetOptions::default();

        let ctx = self
            .navigator
            .get_symbol_context(repo_root, &intel_loc, &opt)
            .map_err(|e| error::LunaError::invalid_input(format!("navigation error: {e}")))?;

        Ok(ctx.signature_line)
    }

    fn get_snippet(
        &self,
        repo_root: &Path,
        location: &SourceLocation,
        context_lines: usize,
    ) -> error::Result<String> {
        let intel_loc = source_location_to_intel_location(location);
        let opt = SnippetOptions {
            context_lines,
            ..SnippetOptions::default()
        };

        let ctx = self
            .navigator
            .get_symbol_context(repo_root, &intel_loc, &opt)
            .map_err(|e| error::LunaError::invalid_input(format!("navigation error: {e}")))?;

        Ok(ctx.snippet)
    }
}

/// Convert intelligence::SymbolLocation to context::SourceLocation
fn intel_location_to_source_location(
    repo_root: &Path,
    loc: IntelSymbolLocation,
) -> SourceLocation {
    SourceLocation {
        repo_root: repo_root.to_path_buf(),
        rel_path: loc.rel_path,
        range: TextRange::with_cols(
            loc.range.start.line + 1, // 0-based to 1-based
            loc.range.start.column,
            loc.range.end.line + 1,
            loc.range.end.column,
        ),
    }
}

/// Convert context::SourceLocation to intelligence::SymbolLocation
fn source_location_to_intel_location(loc: &SourceLocation) -> IntelSymbolLocation {
    IntelSymbolLocation {
        rel_path: loc.rel_path.clone(),
        range: CoreTextRange {
            start: Position {
                line: loc.range.start_line.saturating_sub(1), // 1-based to 0-based
                column: loc.range.start_col,
                byte: 0,
            },
            end: Position {
                line: loc.range.end_line.saturating_sub(1),
                column: loc.range.end_col,
                byte: 0,
            },
        },
    }
}

/// Factory function to create a fully configured RefillPipeline
pub fn create_refill_pipeline(
    repo_root: PathBuf,
) -> Option<context::RefillPipeline> {
    tracing::debug!("Creating RefillPipeline for: {}", repo_root.display());

    // Check if repo_root exists and is a directory
    if !repo_root.exists() {
        tracing::warn!("Repo root does not exist: {}", repo_root.display());
        return None;
    }
    if !repo_root.is_dir() {
        tracing::warn!("Repo root is not a directory: {}", repo_root.display());
        return None;
    }

    tracing::debug!("Repo root validated, creating navigator...");

    // Create navigator
    let navigator: Arc<intelligence::TreeSitterNavigator<FsRepoFileProvider>> =
        Arc::new(intelligence::TreeSitterNavigator::default());

    // Create adapters
    let file_provider: Arc<dyn FileProvider> =
        Arc::new(IntelligenceFileProvider::new(Arc::clone(&navigator)));
    let symbol_resolver: Arc<dyn SymbolResolver> =
        Arc::new(IntelligenceSymbolResolver::new(navigator));

    let budget = context::TokenBudget {
        max_context_tokens: 4000,
    };

    Some(context::RefillPipeline::new(
        repo_root,
        file_provider,
        symbol_resolver,
        budget,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_conversion() {
        let repo_root = PathBuf::from("/repo");
        let intel_loc = IntelSymbolLocation {
            rel_path: PathBuf::from("src/lib.rs"),
            range: CoreTextRange {
                start: Position {
                    line: 9,    // 0-based
                    column: 4,
                    byte: 100,
                },
                end: Position {
                    line: 9,
                    column: 20,
                    byte: 116,
                },
            },
        };

        let ctx_loc = intel_location_to_source_location(&repo_root, intel_loc);
        assert_eq!(ctx_loc.rel_path, PathBuf::from("src/lib.rs"));
        assert_eq!(ctx_loc.range.start_line, 10); // 1-based
        assert_eq!(ctx_loc.range.start_col, 4);

        // Convert back
        let back_to_intel = source_location_to_intel_location(&ctx_loc);
        assert_eq!(back_to_intel.range.start.line, 9); // Back to 0-based
    }
}
