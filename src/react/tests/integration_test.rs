//! Integration tests for the ReAct agent
//!
//! These tests verify the full flow of the ReAct loop including:
//! - Tool execution
//! - State tracking
//! - Context building

use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use toolkit::Tool;

// Helper function to create a temporary test repository
fn setup_test_repo() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();

    // Create a simple Rust file
    let rust_file = repo_path.join("src/lib.rs");
    fs::create_dir_all(repo_path.join("src")).unwrap();
    fs::write(
        &rust_file,
        r#"pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#,
    )
    .unwrap();

    // Create a README
    fs::write(
        repo_path.join("README.md"),
        "# Test Repository\n\nThis is a test repository for ReAct agent integration tests.",
    )
    .unwrap();

    temp_dir
}

#[test]
fn test_tool_registry() {
    use toolkit::{ListDirTool, ReadFileTool};
    use toolkit::{Tool, ToolInput, ToolOutput, ToolRegistry};

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(ReadFileTool::new()));
    registry.register(Box::new(ListDirTool::new()));

    assert_eq!(registry.len(), 2);
    assert!(registry.has("read_file"));
    assert!(registry.has("list_dir"));
}

#[test]
fn test_read_file_tool() {
    use std::path::PathBuf;
    use toolkit::{ReadFileTool, ToolInput};

    let temp_dir = setup_test_repo();
    let test_file = temp_dir.path().join("src/lib.rs");

    let tool = ReadFileTool::new();
    let input = ToolInput {
        args: serde_json::json!({
            "path": "src/lib.rs",
        }),
        repo_root: temp_dir.path().to_path_buf(),
        policy: None,
    };

    let output = tool.execute(&input);

    assert!(output.success);
    assert!(output.data.is_object());
    let content = output.data.get("content").and_then(|v| v.as_str());
    assert!(content.is_some());
    assert!(content.unwrap().contains("pub fn greet"));
}

#[test]
fn test_list_dir_tool() {
    use toolkit::{ListDirTool, ToolInput};

    let temp_dir = setup_test_repo();

    let tool = ListDirTool::new();
    let input = ToolInput {
        args: serde_json::json!({
            "path": "src",
        }),
        repo_root: temp_dir.path().to_path_buf(),
        policy: None,
    };

    let output = tool.execute(&input);

    assert!(output.success);
    let entries = output.data.get("entries").and_then(|v| v.as_array());
    assert!(entries.is_some());
    assert!(!entries.unwrap().is_empty());
}

#[test]
fn test_context_pack_building() {
    use core::code_chunk::{IndexChunkOptions, RefillOptions};
    use tokenizers::Tokenizer;
    use tools::{build_context_pack_keyword, SearchCodeOptions};

    let temp_dir = setup_test_repo();

    // Try to load tokenizer, skip test if not available
    let tokenizer = match Tokenizer::from_file("data/tokenizer.json") {
        Ok(t) => t,
        Err(_) => {
            println!("Skipping test: tokenizer not found");
            return;
        }
    };

    let result = build_context_pack_keyword(
        temp_dir.path(),
        "greet",
        &tokenizer,
        SearchCodeOptions::default(),
        IndexChunkOptions::default(),
        RefillOptions::default(),
    );

    assert!(result.is_ok());
    let pack = result.unwrap();
    assert!(!pack.context.is_empty());
}

#[test]
fn test_search_functionality() {
    use core::code_chunk::IndexChunkOptions;
    use tokenizers::Tokenizer;
    use tools::{search_code_keyword, SearchCodeOptions};

    let temp_dir = setup_test_repo();

    // Try to load tokenizer, skip test if not available
    let tokenizer = match Tokenizer::from_file("data/tokenizer.json") {
        Ok(t) => t,
        Err(_) => {
            println!("Skipping test: tokenizer not found");
            return;
        }
    };

    let result = search_code_keyword(
        temp_dir.path(),
        "greet",
        &tokenizer,
        IndexChunkOptions::default(),
        SearchCodeOptions {
            max_files: 100,
            max_hits: 10,
            ..Default::default()
        },
    );

    assert!(result.is_ok());
    let (hits, _trace) = result.unwrap();
    assert!(!hits.is_empty());
}

#[test]
fn test_file_edit_operations() {
    use tools::{edit_file, EditOp};

    let temp_dir = setup_test_repo();
    let test_file = temp_dir.path().join("test.txt");

    // Create test file
    fs::write(&test_file, "line 1\nline 2\nline 3\n").unwrap();

    // Edit file - ReplaceLines uses 0-based line numbers (inclusive)
    // Replace line 1 (0-based, the second line) with new content
    let op = EditOp::ReplaceLines {
        start_line: 1,
        end_line: 1,
        new_content: "edited line".to_string(),
    };

    let result = edit_file(&test_file, &op, false);

    assert!(result.is_ok());
    let edit_result = result.unwrap();
    assert!(edit_result.success);
    assert_eq!(edit_result.lines_changed, Some(1));

    // Verify the edit
    let content = fs::read_to_string(&test_file).unwrap();
    assert!(content.contains("edited line"));
    assert_eq!(content, "line 1\nedited line\nline 3\n");
}

#[test]
fn test_merge_hits() {
    use core::code_chunk::IndexChunk;
    use react::state::merge_hits;

    let base = vec![IndexChunk {
        path: "test.rs".to_string(),
        start_byte: 0,
        end_byte: 10,
        start_line: 0,
        end_line: 1,
        text: "hello".to_string(),
    }];

    let more = vec![
        IndexChunk {
            path: "test.rs".to_string(),
            start_byte: 0,
            end_byte: 10,
            start_line: 0,
            end_line: 1,
            text: "hello".to_string(),
        },
        IndexChunk {
            path: "other.rs".to_string(),
            start_byte: 20,
            end_byte: 30,
            start_line: 2,
            end_line: 3,
            text: "world".to_string(),
        },
    ];

    let merged = merge_hits(base, more);

    // Should deduplicate by (path, start_byte, end_byte)
    assert_eq!(merged.len(), 2);
    assert!(merged.iter().any(|h| h.path == "test.rs"));
    assert!(merged.iter().any(|h| h.path == "other.rs"));
}

#[test]
fn test_summarize_state() {
    use core::code_chunk::ContextChunk;
    use react::summarize_state;

    let context = vec![ContextChunk {
        path: "test.rs".to_string(),
        alias: 0,
        snippet: "pub fn test() {}".to_string(),
        start_line: 0,
        end_line: 1,
        reason: "test".to_string(),
    }];

    let hits = vec![];
    let summary = summarize_state(&hits, &context);

    assert!(summary.contains("hits=0"));
    assert!(summary.contains("context_chunks=1"));
}

#[test]
fn test_goto_definition_tool() {
    use toolkit::{GotoDefinitionTool, ToolInput};

    let temp_dir = setup_test_repo();

    let tool = GotoDefinitionTool::new();
    let input = ToolInput {
        args: serde_json::json!({
            "symbol_name": "greet",
            "max_results": 5,
        }),
        repo_root: temp_dir.path().to_path_buf(),
        policy: None,
    };

    let output = tool.execute(&input);

    assert!(output.success);
    let defs = output.data.get("definitions").and_then(|v| v.as_array());
    assert!(defs.is_some());
    // Should find the greet function in lib.rs
    let defs = defs.unwrap();
    assert!(!defs.is_empty());

    let first_def = defs.first().unwrap();
    let path = first_def.get("path").and_then(|v| v.as_str());
    assert!(path.is_some());
    assert!(path.unwrap().contains("lib.rs"));
}

#[test]
fn test_enhanced_state_summary() {
    use core::code_chunk::ContextChunk;
    use react::state::summarize_state_enhanced;

    let temp_dir = setup_test_repo();

    let context = vec![ContextChunk {
        path: "src/lib.rs".to_string(),
        alias: 0,
        snippet: "pub fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}\n\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string(),
        start_line: 0,
        end_line: 6,
        reason: "test".to_string(),
    }];

    let hits = vec![];
    let summary = summarize_state_enhanced(&hits, &context, temp_dir.path());

    assert!(summary.contains("hits=0"));
    assert!(summary.contains("context_chunks=1"));
    // Enhanced summary includes symbol information
    assert!(summary.contains("symbols=") || summary.contains("definitions:"));
}

#[test]
fn test_symbol_resolution_in_refill() {
    use core::code_chunk::{IndexChunkOptions, RefillOptions};
    use tokenizers::Tokenizer;
    use tools::{find_symbol_definitions, refill_hits, search_code_keyword, SearchCodeOptions};

    let temp_dir = setup_test_repo();

    // Try to load tokenizer, skip test if not available
    let tokenizer = match Tokenizer::from_file("data/tokenizer.json") {
        Ok(t) => t,
        Err(_) => {
            println!("Skipping test: tokenizer not found");
            return;
        }
    };

    // Search for "add" function
    let result = search_code_keyword(
        temp_dir.path(),
        "add",
        &tokenizer,
        IndexChunkOptions::default(),
        SearchCodeOptions {
            max_files: 100,
            max_hits: 10,
            ..Default::default()
        },
    );

    assert!(result.is_ok());
    let (hits, _) = result.unwrap();

    // Refill hits - this should also perform automatic symbol resolution
    let result = refill_hits(temp_dir.path(), &hits, RefillOptions::default());
    assert!(result.is_ok());

    let (context, _) = result.unwrap();
    // Should have context chunks
    assert!(!context.is_empty());
}

#[test]
fn test_find_symbol_definitions() {
    use tools::find_symbol_definitions;

    let temp_dir = setup_test_repo();

    // Find definition of "greet" function
    let result = find_symbol_definitions(temp_dir.path(), "greet", 5);

    // Print error if failed for debugging
    if let Err(ref e) = result {
        println!("Error finding symbol definitions: {}", e);
    }

    assert!(result.is_ok(), "find_symbol_definitions failed: {:?}", result);
    let defs = result.unwrap();
    assert!(!defs.is_empty(), "No definitions found for 'greet'");

    let def = &defs[0];
    assert_eq!(def.path, "src/lib.rs");
    assert_eq!(def.kind, "definition");
}
