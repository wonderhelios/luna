//! Smart Code Chunking Strategies
//!
//! This module provides various strategies for splitting code into chunks
//! suitable for embedding and semantic search.
//!
//! ## Chunking Strategies
//!
//! 1. **Fixed Size**: Simple byte/token-based splitting
//! 2. **Semantic**: Split at logical boundaries (functions, structs, etc.)
//! 3. **Hierarchical**: Multi-level chunking (file -> class -> method)

use std::collections::HashMap;
use std::path::Path;

use crate::{LanguageId, SourceLocation, TextRange};

/// A chunk of code with metadata
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// Unique identifier for this chunk
    pub id: String,
    /// The code content
    pub content: String,
    /// Source location
    pub source: SourceLocation,
    /// Language
    pub language: LanguageId,
    /// Chunk type
    pub chunk_type: ChunkType,
    /// Estimated token count
    pub token_count: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CodeChunk {
    /// Create a new code chunk
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        source: SourceLocation,
        language: LanguageId,
        chunk_type: ChunkType,
    ) -> Self {
        let content = content.into();
        let token_count = estimate_tokens(&content);

        Self {
            id: id.into(),
            content,
            source,
            language,
            chunk_type,
            token_count,
            metadata: HashMap::new(),
        }
    }

    /// Get text suitable for embedding
    /// Includes context like language and chunk type
    pub fn embedding_text(&self) -> String {
        let context = match self.chunk_type {
            ChunkType::Function { ref name, .. } => format!("Function {}: ", name),
            ChunkType::Struct { ref name } => format!("Struct {}: ", name),
            ChunkType::Trait { ref name } => format!("Trait {}: ", name),
            ChunkType::Impl { ref type_name } => format!("Implementation for {}: ", type_name),
            ChunkType::Module { ref name } => format!("Module {}: ", name),
            ChunkType::Comment => "Documentation: ".to_string(),
            ChunkType::Import => "Imports: ".to_string(),
            ChunkType::FileSummary => "File overview: ".to_string(),
            ChunkType::CodeBlock => "Code: ".to_string(),
        };

        format!("{}{}", context, self.content)
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Type of code chunk
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    /// Function or method
    Function {
        name: String,
        signature: String,
        is_async: bool,
        is_unsafe: bool,
    },
    /// Struct or class definition
    Struct {
        name: String,
    },
    /// Trait or interface
    Trait {
        name: String,
    },
    /// Implementation block
    Impl {
        type_name: String,
    },
    /// Module or namespace
    Module {
        name: String,
    },
    /// Documentation comment
    Comment,
    /// Import statements
    Import,
    /// File summary
    FileSummary,
    /// Generic code block
    CodeBlock,
}

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Simple fixed-size chunking with overlap
    FixedSize,
    /// Semantic-aware chunking at logical boundaries
    Semantic,
    /// Hierarchical multi-level chunking
    Hierarchical,
}

/// Code chunker - splits code into chunks
pub struct CodeChunker {
    strategy: ChunkingStrategy,
    max_tokens: usize,
    overlap_tokens: usize,
}

impl CodeChunker {
    /// Create a new code chunker
    pub fn new(strategy: ChunkingStrategy, max_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            strategy,
            max_tokens,
            overlap_tokens,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ChunkingStrategy::Semantic, 256, 32)
    }

    /// Chunk a file into pieces
    pub fn chunk_file(
        &self,
        content: &str,
        path: &Path,
        repo_root: &Path,
    ) -> Vec<CodeChunk> {
        let language = detect_language(path);
        let rel_path = path.strip_prefix(repo_root).unwrap_or(path);

        match self.strategy {
            ChunkingStrategy::FixedSize => {
                self.fixed_size_chunks(content, rel_path, repo_root, language)
            }
            ChunkingStrategy::Semantic => {
                self.semantic_chunks(content, rel_path, repo_root, language)
            }
            ChunkingStrategy::Hierarchical => {
                self.hierarchical_chunks(content, rel_path, repo_root, language)
            }
        }
    }

    /// Fixed-size chunking with overlap
    fn fixed_size_chunks(
        &self,
        content: &str,
        rel_path: &Path,
        repo_root: &Path,
        language: LanguageId,
    ) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let approx_tokens_per_line = 10; // Rough estimate
        let lines_per_chunk = self.max_tokens / approx_tokens_per_line;
        let overlap_lines = self.overlap_tokens / approx_tokens_per_line;

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_idx = 0;

        while start < total_lines {
            let end = (start + lines_per_chunk).min(total_lines);
            let chunk_lines = &lines[start..end];
            let chunk_content = chunk_lines.join("\n");

            let source = SourceLocation {
                repo_root: repo_root.to_path_buf(),
                rel_path: rel_path.to_path_buf(),
                range: TextRange::new(start + 1, end),
            };

            let chunk = CodeChunk::new(
                format!("{}#chunk{}", rel_path.display(), chunk_idx),
                chunk_content,
                source,
                language,
                ChunkType::CodeBlock,
            );

            chunks.push(chunk);
            chunk_idx += 1;

            // Move forward with overlap
            if end >= total_lines {
                break;
            }
            start = end - overlap_lines;
        }

        chunks
    }

    /// Semantic chunking - split at logical boundaries
    fn semantic_chunks(
        &self,
        content: &str,
        rel_path: &Path,
        repo_root: &Path,
        language: LanguageId,
    ) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();

        // First, try to extract top-level items
        let items = extract_top_level_items(content, language);

        if items.is_empty() {
            // Fall back to fixed-size chunking
            return self.fixed_size_chunks(content, rel_path, repo_root, language);
        }

        let lines: Vec<&str> = content.lines().collect();

        for (idx, item) in items.iter().enumerate() {
            let start_line = item.start_line;
            let end_line = item.end_line.min(lines.len());

            if start_line >= end_line || start_line == 0 {
                continue;
            }

            let chunk_lines = &lines[start_line - 1..end_line];
            let chunk_content = chunk_lines.join("\n");

            // Skip if too large, split it
            if estimate_tokens(&chunk_content) > self.max_tokens * 2 {
                chunks.extend(self.split_large_item(
                    &chunk_content,
                    item,
                    rel_path,
                    repo_root,
                    idx,
                ));
                continue;
            }

            let source = SourceLocation {
                repo_root: repo_root.to_path_buf(),
                rel_path: rel_path.to_path_buf(),
                range: TextRange::new(start_line, end_line),
            };

            let chunk_type = match item.kind {
                ItemKind::Function => ChunkType::Function {
                    name: item.name.clone(),
                    signature: item.signature.clone(),
                    is_async: item.is_async,
                    is_unsafe: item.is_unsafe,
                },
                ItemKind::Struct => ChunkType::Struct {
                    name: item.name.clone(),
                },
                ItemKind::Trait => ChunkType::Trait {
                    name: item.name.clone(),
                },
                ItemKind::Impl => ChunkType::Impl {
                    type_name: item.name.clone(),
                },
                ItemKind::Module => ChunkType::Module {
                    name: item.name.clone(),
                },
                ItemKind::Comment => ChunkType::Comment,
                ItemKind::Import => ChunkType::Import,
                ItemKind::CodeBlock => ChunkType::CodeBlock,
            };

            let chunk = CodeChunk::new(
                format!("{}#{}", rel_path.display(), item.name),
                chunk_content,
                source,
                language,
                chunk_type,
            );

            chunks.push(chunk);
        }

        // Add file summary chunk
        let summary = create_file_summary(content, rel_path, repo_root, language);
        chunks.insert(0, summary);

        chunks
    }

    /// Split a large item into smaller chunks
    fn split_large_item(
        &self,
        content: &str,
        item: &TopLevelItem,
        rel_path: &Path,
        repo_root: &Path,
        _idx: usize,
    ) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut sub_idx = 0;
        let mut current_chunk = String::new();
        let mut current_tokens = 0;
        let mut start_line = item.start_line;

        for (line_idx, line) in lines.iter().enumerate() {
            let line_tokens = estimate_tokens(line);

            if current_tokens + line_tokens > self.max_tokens && !current_chunk.is_empty() {
                // Save current chunk
                let source = SourceLocation {
                    repo_root: repo_root.to_path_buf(),
                    rel_path: rel_path.to_path_buf(),
                    range: TextRange::new(start_line, item.start_line + line_idx),
                };

                let chunk = CodeChunk::new(
                    format!("{}#{}#part{}", rel_path.display(), item.name, sub_idx),
                    current_chunk.clone(),
                    source,
                    detect_language(rel_path),
                    ChunkType::CodeBlock,
                );

                chunks.push(chunk);
                sub_idx += 1;

                // Start new chunk with overlap
                current_chunk = lines
                    [line_idx.saturating_sub(self.overlap_tokens / 5)..line_idx]
                    .join("\n");
                current_tokens = estimate_tokens(&current_chunk);
                start_line = item.start_line + line_idx.saturating_sub(self.overlap_tokens / 5);
            }

            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
            current_tokens += line_tokens;
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            let source = SourceLocation {
                repo_root: repo_root.to_path_buf(),
                rel_path: rel_path.to_path_buf(),
                range: TextRange::new(start_line, item.end_line),
            };

            let chunk = CodeChunk::new(
                format!("{}#{}#part{}", rel_path.display(), item.name, sub_idx),
                current_chunk,
                source,
                detect_language(rel_path),
                ChunkType::CodeBlock,
            );

            chunks.push(chunk);
        }

        chunks
    }

    /// Hierarchical chunking - multi-level representation
    fn hierarchical_chunks(
        &self,
        content: &str,
        rel_path: &Path,
        repo_root: &Path,
        language: LanguageId,
    ) -> Vec<CodeChunk> {
        // Start with semantic chunks
        let chunks = self.semantic_chunks(content, rel_path, repo_root, language);

        // Add a high-level overview chunk
        let overview = create_file_summary(content, rel_path, repo_root, language);

        // Add import chunk if present
        let import_chunk = extract_imports_chunk(content, rel_path, repo_root, language);

        // Combine: overview first, then imports, then rest
        let mut result = vec![overview];
        if let Some(imports) = import_chunk {
            result.push(imports);
        }
        result.extend(chunks.into_iter().filter(|c| c.chunk_type != ChunkType::FileSummary));

        result
    }
}

/// Item kind for top-level extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Function,
    Struct,
    Trait,
    Impl,
    Module,
    Comment,
    Import,
    CodeBlock,
}

/// A top-level item in source code
#[derive(Debug, Clone)]
struct TopLevelItem {
    kind: ItemKind,
    name: String,
    signature: String,
    start_line: usize,
    end_line: usize,
    is_async: bool,
    is_unsafe: bool,
}

/// Extract top-level items from source code
fn extract_top_level_items(content: &str, language: LanguageId) -> Vec<TopLevelItem> {
    match language {
        LanguageId::Rust => extract_rust_items(content),
        LanguageId::Python => extract_python_items(content),
        LanguageId::JavaScript | LanguageId::TypeScript => extract_js_items(content),
        LanguageId::Go => extract_go_items(content),
        _ => extract_generic_items(content),
    }
}

/// Extract Rust-specific items
fn extract_rust_items(content: &str) -> Vec<TopLevelItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut brace_count = 0;
    let mut current_item: Option<TopLevelItem> = None;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        // Skip empty lines and comments at brace level 0
        if brace_count == 0 {
            // Function detection
            if let Some(caps) = RUST_FN_REGEX.captures(trimmed) {
                if current_item.is_some() {
                    // Close previous item
                    let mut item = current_item.take().unwrap();
                    item.end_line = line_num - 1;
                    items.push(item);
                }

                let is_async = caps.get(1).is_some();
                let is_unsafe = caps.get(2).is_some();
                let name = caps.get(3).map(|m| m.as_str().to_string()).unwrap_or_default();

                current_item = Some(TopLevelItem {
                    kind: ItemKind::Function,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line: line_num,
                    is_async,
                    is_unsafe,
                });

                // Count braces on this line
                brace_count += line.matches('{').count();
                brace_count = brace_count.saturating_sub(line.matches('}').count());
                continue;
            }

            // Struct detection
            if let Some(caps) = RUST_STRUCT_REGEX.captures(trimmed) {
                if current_item.is_some() {
                    let mut item = current_item.take().unwrap();
                    item.end_line = line_num - 1;
                    items.push(item);
                }

                let name = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();

                current_item = Some(TopLevelItem {
                    kind: ItemKind::Struct,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line: line_num,
                    is_async: false,
                    is_unsafe: false,
                });

                brace_count += line.matches('{').count();
                brace_count = brace_count.saturating_sub(line.matches('}').count());
                continue;
            }

            // Trait detection
            if let Some(caps) = RUST_TRAIT_REGEX.captures(trimmed) {
                if current_item.is_some() {
                    let mut item = current_item.take().unwrap();
                    item.end_line = line_num - 1;
                    items.push(item);
                }

                let name = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();

                current_item = Some(TopLevelItem {
                    kind: ItemKind::Trait,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line: line_num,
                    is_async: false,
                    is_unsafe: false,
                });

                brace_count += line.matches('{').count();
                brace_count = brace_count.saturating_sub(line.matches('}').count());
                continue;
            }

            // Impl detection
            if let Some(caps) = RUST_IMPL_REGEX.captures(trimmed) {
                if current_item.is_some() {
                    let mut item = current_item.take().unwrap();
                    item.end_line = line_num - 1;
                    items.push(item);
                }

                let name = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();

                current_item = Some(TopLevelItem {
                    kind: ItemKind::Impl,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line: line_num,
                    is_async: false,
                    is_unsafe: false,
                });

                brace_count += line.matches('{').count();
                brace_count = brace_count.saturating_sub(line.matches('}').count());
                continue;
            }
        }

        // Track brace depth
        brace_count += line.matches('{').count();
        brace_count = brace_count.saturating_sub(line.matches('}').count());

        // If we're back to brace level 0, close current item
        if brace_count == 0 {
            if let Some(mut item) = current_item.take() {
                item.end_line = line_num;
                items.push(item);
            }
        }
    }

    // Close any remaining item
    if let Some(mut item) = current_item {
        item.end_line = lines.len();
        items.push(item);
    }

    items
}

// Simple regex-like patterns for Rust
lazy_static::lazy_static! {
    static ref RUST_FN_REGEX: regex::Regex = regex::Regex::new(
        r"^\s*(?:pub\s+)?(async\s+)?(unsafe\s+)?fn\s+(\w+)"
    ).unwrap();

    static ref RUST_STRUCT_REGEX: regex::Regex = regex::Regex::new(
        r"^\s*(?:pub\s+)?struct\s+(\w+)"
    ).unwrap();

    static ref RUST_TRAIT_REGEX: regex::Regex = regex::Regex::new(
        r"^\s*(?:pub\s+)?trait\s+(\w+)"
    ).unwrap();

    static ref RUST_IMPL_REGEX: regex::Regex = regex::Regex::new(
        r"^\s*impl\s+(?:<[^>]+>\s+)?(\w+)"
    ).unwrap();
}

/// Extract Python items
fn extract_python_items(content: &str) -> Vec<TopLevelItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        // Skip if line is indented (not top-level)
        if !line.trim().is_empty() && line.starts_with(' ') {
            continue;
        }

        // Function: def name(...)
        if let Some(name) = trimmed.strip_prefix("def ") {
            let name = name.split('(').next().unwrap_or("").trim().to_string();
            if !name.is_empty() {
                // Find end (next non-indented line or EOF)
                let mut end_line = lines.len();
                for (next_idx, next_line) in lines.iter().enumerate().skip(idx + 1) {
                    if !next_line.trim().is_empty() && !next_line.starts_with(' ') {
                        end_line = next_idx;
                        break;
                    }
                }

                items.push(TopLevelItem {
                    kind: ItemKind::Function,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line,
                    is_async: trimmed.starts_with("async "),
                    is_unsafe: false,
                });
            }
        }

        // Class: class Name(...)
        if let Some(name) = trimmed.strip_prefix("class ") {
            let name = name.split('(').next().unwrap_or("").split(':').next().unwrap_or("").trim().to_string();
            if !name.is_empty() {
                let mut end_line = lines.len();
                for (next_idx, next_line) in lines.iter().enumerate().skip(idx + 1) {
                    if !next_line.trim().is_empty() && !next_line.starts_with(' ') {
                        end_line = next_idx;
                        break;
                    }
                }

                items.push(TopLevelItem {
                    kind: ItemKind::Struct,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line,
                    is_async: false,
                    is_unsafe: false,
                });
            }
        }
    }

    items
}

/// Extract JavaScript/TypeScript items
fn extract_js_items(content: &str) -> Vec<TopLevelItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut brace_count = 0;
    let mut current_item: Option<TopLevelItem> = None;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        if brace_count == 0 {
            // Function detection: function name(...) or const name = (...) =>
            if trimmed.starts_with("function ") || trimmed.starts_with("export function ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1].split('(').next().unwrap_or("").to_string();
                    if !name.is_empty() {
                        current_item = Some(TopLevelItem {
                            kind: ItemKind::Function,
                            name,
                            signature: trimmed.to_string(),
                            start_line: line_num,
                            end_line: line_num,
                            is_async: trimmed.contains("async "),
                            is_unsafe: false,
                        });
                    }
                }
            }

            // Class detection
            if trimmed.starts_with("class ") || trimmed.starts_with("export class ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1].split('{').next().unwrap_or("").trim().to_string();
                    if !name.is_empty() {
                        current_item = Some(TopLevelItem {
                            kind: ItemKind::Struct,
                            name,
                            signature: trimmed.to_string(),
                            start_line: line_num,
                            end_line: line_num,
                            is_async: false,
                            is_unsafe: false,
                        });
                    }
                }
            }
        }

        // Track brace depth
        brace_count += line.matches('{').count();
        brace_count = brace_count.saturating_sub(line.matches('}').count());

        if brace_count == 0 {
            if let Some(mut item) = current_item.take() {
                item.end_line = line_num;
                items.push(item);
            }
        }
    }

    if let Some(mut item) = current_item {
        item.end_line = lines.len();
        items.push(item);
    }

    items
}

/// Extract Go items
fn extract_go_items(content: &str) -> Vec<TopLevelItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        // Function: func Name(...) or func (r *Receiver) Name(...)
        if trimmed.starts_with("func ") {
            let after_func = &trimmed[5..];
            let name = if after_func.starts_with('(') {
                // Method
                after_func
                    .split(')')
                    .nth(1)
                    .and_then(|s| s.split('(').next())
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                // Function
                after_func
                    .split('(')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string()
            };

            if !name.is_empty() {
                // Find end (closing brace at column 0)
                let mut end_line = lines.len();
                for (next_idx, next_line) in lines.iter().enumerate().skip(idx + 1) {
                    if next_line.trim() == "}" {
                        end_line = next_idx + 1;
                        break;
                    }
                }

                items.push(TopLevelItem {
                    kind: ItemKind::Function,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line,
                    is_async: false,
                    is_unsafe: false,
                });
            }
        }

        // Type: type Name struct { ... }
        if trimmed.starts_with("type ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                let name = parts[1].to_string();

                let mut end_line = lines.len();
                for (next_idx, next_line) in lines.iter().enumerate().skip(idx + 1) {
                    if next_line.trim() == "}" {
                        end_line = next_idx + 1;
                        break;
                    }
                }

                let kind = if trimmed.contains("interface") {
                    ItemKind::Trait
                } else {
                    ItemKind::Struct
                };

                items.push(TopLevelItem {
                    kind,
                    name,
                    signature: trimmed.to_string(),
                    start_line: line_num,
                    end_line,
                    is_async: false,
                    is_unsafe: false,
                });
            }
        }
    }

    items
}

/// Generic item extraction for other languages
fn extract_generic_items(content: &str) -> Vec<TopLevelItem> {
    // Simple heuristic: split on blank lines for large blocks
    let mut items = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut current_start = 0;
    let mut current_tokens = 0;

    for (idx, line) in lines.iter().enumerate() {
        let line_tokens = estimate_tokens(line);

        if line.trim().is_empty() && current_tokens > 100 {
            // Create a chunk
            items.push(TopLevelItem {
                kind: ItemKind::CodeBlock,
                name: format!("block_{}", items.len()),
                signature: String::new(),
                start_line: current_start + 1,
                end_line: idx + 1,
                is_async: false,
                is_unsafe: false,
            });
            current_start = idx + 1;
            current_tokens = 0;
        } else {
            current_tokens += line_tokens;
        }
    }

    // Add final chunk
    if current_start < lines.len() {
        items.push(TopLevelItem {
            kind: ItemKind::CodeBlock,
            name: format!("block_{}", items.len()),
            signature: String::new(),
            start_line: current_start + 1,
            end_line: lines.len(),
            is_async: false,
            is_unsafe: false,
        });
    }

    items
}

/// Create a file summary chunk
fn create_file_summary(
    content: &str,
    rel_path: &Path,
    repo_root: &Path,
    language: LanguageId,
) -> CodeChunk {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    // Take first N lines as summary
    let summary_lines = lines.iter().take(30).cloned().collect::<Vec<_>>().join("\n");

    let source = SourceLocation {
        repo_root: repo_root.to_path_buf(),
        rel_path: rel_path.to_path_buf(),
        range: TextRange::new(1, total_lines.min(30)),
    };

    CodeChunk::new(
        format!("{}#summary", rel_path.display()),
        format!("// File: {} ({} lines)\n{}", rel_path.display(), total_lines, summary_lines),
        source,
        language,
        ChunkType::FileSummary,
    )
}

/// Extract imports as a separate chunk
fn extract_imports_chunk(
    content: &str,
    rel_path: &Path,
    repo_root: &Path,
    language: LanguageId,
) -> Option<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut import_lines = Vec::new();
    let mut start_line = 0;
    let mut found_import = false;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        let is_import = match language {
            LanguageId::Rust => {
                trimmed.starts_with("use ") || trimmed.starts_with("extern crate ")
            }
            LanguageId::Python => trimmed.starts_with("import ") || trimmed.starts_with("from "),
            LanguageId::JavaScript | LanguageId::TypeScript => {
                trimmed.starts_with("import ") || trimmed.starts_with("require(")
            }
            LanguageId::Go => trimmed.starts_with("import "),
            LanguageId::Java => trimmed.starts_with("import "),
            _ => false,
        };

        if is_import {
            if !found_import {
                start_line = idx + 1;
                found_import = true;
            }
            import_lines.push(*line);
        } else if found_import && !trimmed.is_empty() && !trimmed.starts_with("//") {
            // Stop at first non-import, non-comment line
            break;
        }
    }

    if import_lines.is_empty() {
        return None;
    }

    let source = SourceLocation {
        repo_root: repo_root.to_path_buf(),
        rel_path: rel_path.to_path_buf(),
        range: TextRange::new(start_line, start_line + import_lines.len() - 1),
    };

    Some(CodeChunk::new(
        format!("{}#imports", rel_path.display()),
        import_lines.join("\n"),
        source,
        language,
        ChunkType::Import,
    ))
}

/// Detect language from file path
fn detect_language(path: &Path) -> LanguageId {
    path.extension()
        .and_then(|e| e.to_str())
        .map(LanguageId::from_extension)
        .unwrap_or(LanguageId::Unknown)
}

/// Estimate token count (rough heuristic: ~4 chars per token)
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_item_extraction() {
        let code = r#"
pub fn foo() {
    println!("hello");
}

struct Bar {
    x: i32,
}

fn baz() -> i32 {
    42
}
"#;

        let items = extract_rust_items(code);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].name, "foo");
        assert_eq!(items[1].name, "Bar");
        assert_eq!(items[2].name, "baz");
    }

    #[test]
    fn test_python_item_extraction() {
        let code = r#"
def hello():
    print("world")

class MyClass:
    def method(self):
        pass

def another():
    pass
"#;

        let items = extract_python_items(code);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].name, "hello");
        assert_eq!(items[1].name, "MyClass");
        assert_eq!(items[2].name, "another");
    }

    #[test]
    fn test_fixed_size_chunking() {
        let chunker = CodeChunker::new(ChunkingStrategy::FixedSize, 100, 20);
        let code = "line\n".repeat(100);

        let chunks = chunker.chunk_file(
            &code,
            Path::new("/repo/src/main.rs"),
            Path::new("/repo"),
        );

        assert!(!chunks.is_empty());
        // Should have multiple chunks due to size limit
        assert!(chunks.len() > 1);

        // Check overlap
        if chunks.len() > 1 {
            let first_end = chunks[0].source.range.end_line;
            let second_start = chunks[1].source.range.start_line;
            assert!(second_start < first_end, "Chunks should overlap");
        }
    }

    #[test]
    fn test_chunk_embedding_text() {
        let chunk = CodeChunk::new(
            "test",
            "fn main() {}",
            SourceLocation {
                repo_root: PathBuf::from("/repo"),
                rel_path: PathBuf::from("src/main.rs"),
                range: TextRange::new(1, 1),
            },
            LanguageId::Rust,
            ChunkType::Function {
                name: "main".to_string(),
                signature: "fn main()".to_string(),
                is_async: false,
                is_unsafe: false,
            },
        );

        let text = chunk.embedding_text();
        assert!(text.contains("Function main:"));
        assert!(text.contains("fn main() {}"));
    }

    use std::path::PathBuf;
}
