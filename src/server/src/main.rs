use ahash::AHashMap;
use anyhow::Result;
use clap::{Parser, Subcommand};
use core::code_chunk::{ChunkOptions, IndexChunkOptions, RefillOptions};
use index::chunk_source;
use intelligence::TreeSitterFile;
use std::path::PathBuf;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

fn demo_tokenizer() -> Tokenizer {
    // Demo 用极简 tokenizer（真实场景应复用 embedder 的 tokenizer）
    let mut vocab = AHashMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("fn".to_string(), 1u32);
    vocab.insert("let".to_string(), 2u32);
    vocab.insert("return".to_string(), 3u32);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok
}

#[derive(Debug, Parser)]
#[command(name = "luna-server", about = "Luna server CLI (M1)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 运行内置 demo（演示解析/切分/最小 search+refill 闭环）
    Demo,
    /// 在 repo_root 下进行关键词占位检索，并输出 IndexChunk 命中与 Refill 后的 ContextChunk
    Search {
        /// 仓库根目录
        repo_root: PathBuf,
        /// 查询关键词（可多词）
        query: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Demo) {
        Command::Demo => cmd_demo(),
        Command::Search { repo_root, query } => cmd_search(repo_root, query),
    }
}

fn cmd_search(repo_root: PathBuf, query: Vec<String>) -> Result<()> {
    let query = query.join(" ");
    if query.trim().is_empty() {
        anyhow::bail!("query 不能为空（示例：luna-server search . index_chunks）");
    }
    let tok = demo_tokenizer();

    let pack = agent::build_context_pack_keyword(
        &repo_root,
        &query,
        &tok,
        agent::SearchCodeOptions::default(),
        IndexChunkOptions::default(),
        RefillOptions::default(),
    )?;

    println!("Query: {query}\n");
    println!(
        "说明：\n  - preview: 只展示 snippet 的第一行（便于扫一眼）\n  - trace: 工具调用摘要\n  - Hits: 关键词命中的索引块（IndexChunk，可能是同一函数内的多个碎片命中）\n  - ContextChunks: Refill 后的语义上下文块（ContextChunk，已按范围去重，适合后续注入 LLM）\n"
    );
    for t in &pack.trace {
        println!("[trace] {:?}: {}", t.tool, t.summary);
    }

    println!("\nHits: {}", pack.hits.len());
    for (i, h) in pack.hits.iter().take(8).enumerate() {
        let lines = h.end_line.saturating_sub(h.start_line) + 1;
        println!(
            "  [H{:02}] {}:{}..={} ({} lines) bytes {}..{} preview={}",
            i,
            h.path,
            h.start_line + 1,
            h.end_line + 1,
            lines,
            h.start_byte,
            h.end_byte,
            h.text.lines().next().unwrap_or("").trim()
        );
    }
    if pack.hits.len() > 8 {
        println!("  ... ({} hits total)", pack.hits.len());
    }

    println!("\nContextChunks (deduped): {}", pack.context.len());
    for c in pack.context.iter().take(8) {
        let lines = c.end_line.saturating_sub(c.start_line) + 1;
        let bytes = c.snippet.len();
        println!(
            "  #{:02} {}:{}..={} ({} lines, {} bytes) reason={} preview={}",
            c.alias,
            c.path,
            c.start_line + 1,
            c.end_line + 1,
            lines,
            bytes,
            c.reason,
            c.snippet.lines().next().unwrap_or("").trim()
        );
    }
    if pack.context.len() > 8 {
        println!("  ... ({} context chunks total)", pack.context.len());
    }

    Ok(())
}

fn cmd_demo() -> Result<()> {
    println!("Luna Intelligence Demo\n");

    // 1. 准备一段测试代码 (Rust)
    let code = r#"
    fn add(a: i32, b: i32) -> i32 {
        return a + b;
    }

    fn main() {
        let result = add(1, 2);
        println!("Result: {}", result);
    }
    "#;

    println!(" Analyzing Source Code:\n---\n{}\n---", code);

    // 2. 使用 Intelligence 模块进行解析
    // "Rust" 是语言 ID，对应 xc-intelligence 内部的注册
    let ts_file = TreeSitterFile::try_build(code.as_bytes(), "Rust")
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    // 3. 获取 Scope Graph (核心能力：理解作用域、定义和引用)
    let scope_graph = ts_file
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    // 4. 打印所有识别到的符号 (定义)
    println!("\n Detected Symbols (Definitions):");
    let symbols = scope_graph.symbols();

    if symbols.is_empty() {
        println!("   (No symbols found - check query files)");
    } else {
        for symbol in symbols {
            // format: [Line:Column] Kind - Name? (Name extraction might need source slicing)
            let name_range = symbol.range;
            let name = &code[name_range.start.byte..name_range.end.byte];
            println!(
                "    line {}:{} \t[{}] \t{}",
                name_range.start.line + 1,
                name_range.start.column + 1,
                symbol.kind,
                name
            );
        }
    }

    // 5. 智能切分 Demo（用于后续 Index/RAG/上下文注入）
    println!("\n Semantic Chunks (Top-level scopes):");
    let chunks = chunk_source("mem.rs", code.as_bytes(), "Rust", ChunkOptions::default())
        .map_err(|e| anyhow::anyhow!("Failed to chunk: {e}"))?;
    for c in chunks.iter().take(8) {
        println!(
            "    #{:02}  lines {}..={}  bytes={}  preview={}",
            c.alias,
            c.start_line + 1,
            c.end_line + 1,
            c.snippet.len(),
            c.snippet.lines().next().unwrap_or("").trim()
        );
    }
    if chunks.len() > 8 {
        println!("    ... ({} chunks total)", chunks.len());
    }

    // 6. IndexChunk -> Refill -> ContextChunk Demo（调用 agent 的最小闭环）
    println!("\n Search+Refill (M1 minimal loop):");
    let tok = demo_tokenizer();
    let pack = agent::build_context_pack_keyword(
        PathBuf::from(".").as_path(),
        "fn add",
        &tok,
        agent::SearchCodeOptions {
            max_files: 200,
            max_hits: 16,
            ..Default::default()
        },
        IndexChunkOptions {
            min_chunk_tokens: 1,
            max_chunk_tokens: 64,
            ..Default::default()
        },
        RefillOptions::default(),
    )?;
    println!(
        "    hits={} context={}",
        pack.hits.len(),
        pack.context.len()
    );
    for c in pack.context.iter().take(2) {
        println!(
            "    #{:02} {}:{}..={} reason={} preview={}",
            c.alias,
            c.path,
            c.start_line + 1,
            c.end_line + 1,
            c.reason,
            c.snippet.lines().next().unwrap_or("").trim()
        );
    }

    println!("\n Demo finished successfully.");
    Ok(())
}
