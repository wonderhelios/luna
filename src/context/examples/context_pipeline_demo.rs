//! Context Pipeline Demo
//!
//! Run with: cargo run --example context_pipeline_demo
//!
//! This demo shows how the Context Pipeline retrieves and refines code context
//! using ScopeGraph-based symbol navigation.

use std::path::PathBuf;
use std::sync::Arc;

use context::{
    query::ContextQuery,
    refill::{FileProvider, SymbolResolver},
    ContextChunk, SourceLocation, TextRange, TokenBudget,
};

// Mock implementations for demo
struct MockFileProvider;

impl FileProvider for MockFileProvider {
    fn list_files(
        &self,
        _repo_root: &std::path::Path,
    ) -> error::Result<Vec<std::path::PathBuf>> {
        Ok(vec![
            PathBuf::from("src/main.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/utils.rs"),
        ])
    }

    fn read_file(&self, path: &std::path::Path) -> error::Result<String> {
        let content = match path.file_name().and_then(|s| s.to_str()) {
            Some("main.rs") => r#"fn main() {
    println!("Hello, world!");
    let result = find_main_function("test");
    println!("{:?}", result);
}
"#,
            Some("lib.rs") => r#"pub fn find_main_function(query: &str) -> Vec<String> {
    // Search for main functions in the codebase
    vec![format!("Found: {}", query)]
}

pub struct Config {
    pub name: String,
}
"#,
            Some("utils.rs") => r#"pub fn helper() {
    find_main_function("helper");
}
"#,
            _ => "// Unknown file",
        };
        Ok(content.to_string())
    }

    fn modified_time(&self, _path: &std::path::Path) -> error::Result<u64> {
        Ok(0)
    }
}

struct MockSymbolResolver;

impl SymbolResolver for MockSymbolResolver {
    fn find_definition(
        &self,
        repo_root: &std::path::Path,
        name: &str,
    ) -> error::Result<Vec<SourceLocation>> {
        println!("  🔍 Finding definition for: {}", name);

        let locations = if name == "find_main_function" {
            vec![SourceLocation {
                repo_root: repo_root.to_path_buf(),
                rel_path: PathBuf::from("src/lib.rs"),
                range: TextRange::with_cols(1, 1, 4, 1),
            }]
        } else {
            vec![]
        };

        println!("  ✓ Found {} definition(s)", locations.len());
        Ok(locations)
    }

    fn find_references(
        &self,
        repo_root: &std::path::Path,
        name: &str,
        max: usize,
    ) -> error::Result<Vec<SourceLocation>> {
        println!("  🔍 Finding references for: {} (max: {})", name, max);

        let locations = if name == "find_main_function" {
            vec![
                SourceLocation {
                    repo_root: repo_root.to_path_buf(),
                    rel_path: PathBuf::from("src/main.rs"),
                    range: TextRange::new(3, 3),
                },
                SourceLocation {
                    repo_root: repo_root.to_path_buf(),
                    rel_path: PathBuf::from("src/utils.rs"),
                    range: TextRange::new(2, 2),
                },
            ]
        } else {
            vec![]
        };

        let count = locations.len().min(max);
        println!("  ✓ Found {} reference(s)", count);
        Ok(locations.into_iter().take(max).collect())
    }

    fn get_signature(
        &self,
        _repo_root: &std::path::Path,
        location: &SourceLocation,
    ) -> error::Result<Option<String>> {
        let sig = if location.rel_path.ends_with("lib.rs") {
            Some("pub fn find_main_function(query: &str) -> Vec<String>".to_string())
        } else {
            None
        };
        Ok(sig)
    }

    fn get_snippet(
        &self,
        _repo_root: &std::path::Path,
        location: &SourceLocation,
        context_lines: usize,
    ) -> error::Result<String> {
        let snippet = format!(
            "// File: {} (lines {}-{})\n// Context: {} lines\npub fn find_main_function(query: &str) -> Vec<String> {{\n    // ... implementation ...\n}}",
            location.rel_path.display(),
            location.range.start_line,
            location.range.end_line,
            context_lines
        );
        Ok(snippet)
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Context Pipeline Demo                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create pipeline
    let repo_root = PathBuf::from("/demo/repo");
    let file_provider: Arc<dyn FileProvider> = Arc::new(MockFileProvider);
    let symbol_resolver: Arc<dyn SymbolResolver> = Arc::new(MockSymbolResolver);

    let pipeline = context::RefillPipeline::new(
        repo_root.clone(),
        file_provider,
        symbol_resolver,
        TokenBudget {
            max_context_tokens: 2000,
        },
    );

    // Demo 1: Symbol-based query
    println!("Demo 1: Symbol Query ─────────────────────────────────────────");
    println!("Query: ContextQuery::Symbol {{ name: \"find_main_function\" }}");
    println!();

    let query = ContextQuery::symbol("find_main_function");
    match pipeline.retrieve(&query, 5) {
        Ok(index_chunks) => {
            println!("\n  Retrieved {} IndexChunk(s)", index_chunks.len());
            for chunk in &index_chunks {
                println!(
                    "    - {:?}: {} chars",
                    chunk.chunk_type,
                    chunk.content.len()
                );
            }

            println!("\n  Refining to ContextChunks...");
            let context_chunks = pipeline.refine(&index_chunks);
            print_context_chunks(&context_chunks);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Demo 2: Task-driven query
    println!("\n\nDemo 2: Task-Driven Query ────────────────────────────────────");
    println!("Query: TaskDriven with symbols and paths");
    println!();

    let task_query = ContextQuery::TaskDriven {
        keywords: vec!["find".to_string(), "main".to_string()],
        paths: vec![PathBuf::from("src/main.rs")],
        symbols: vec!["find_main_function".to_string()],
    };

    match pipeline.retrieve(&task_query, 10) {
        Ok(index_chunks) => {
            println!("  Retrieved {} IndexChunk(s)", index_chunks.len());
            let context_chunks = pipeline.refine(&index_chunks);
            print_context_chunks(&context_chunks);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Demo 3: Build context string
    println!("\n\nDemo 3: Build Context String for LLM ─────────────────────────");
    let query = ContextQuery::symbol("find_main_function");
    if let Ok(index_chunks) = pipeline.retrieve(&query, 5) {
        let context_chunks = pipeline.refine(&index_chunks);
        let context_str = pipeline.build_context_string(&context_chunks);

        println!("Generated context string:");
        println!("─────────────────────────────────────────────────────────────");
        println!("{}", context_str);
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "Total tokens: {}",
            context_chunks.iter().map(|c| c.token_count).sum::<usize>()
        );
    }

    println!("\n\n✅ Demo complete!");
}

fn print_context_chunks(chunks: &[ContextChunk]) {
    println!("\n  ContextChunks ({} total):", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n  [{}] Type: {:?}", i + 1, chunk.context_type);
        println!(
            "      Source: {}:{}-{}",
            chunk.source.rel_path.display(),
            chunk.source.range.start_line,
            chunk.source.range.end_line
        );
        println!("      Relevance: {:.2}", chunk.relevance_score);
        println!("      Tokens: {}", chunk.token_count);
        if !chunk.symbol_signatures.is_empty() {
            println!("      Signatures: {:?}", chunk.symbol_signatures);
        }

        // Print first few lines of content
        let preview: String = chunk
            .content
            .lines()
            .take(3)
            .collect::<Vec<_>>()
            .join("\n");
        println!("      Content preview:\n{}", preview);
        if chunk.content.lines().count() > 3 {
            println!("      ... ({} more lines)", chunk.content.lines().count() - 3);
        }
    }
}
