use ahash::AHashMap;
use anyhow::Result;
use core::code_chunk::{ChunkOptions, IndexChunkOptions, RefillOptions};
use index::{chunk_source, index_chunks, refill_chunks};
use intelligence::TreeSitterFile;
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

fn main() -> Result<()> {
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

    // 6. IndexChunk -> Refill -> ContextChunk Demo
    println!("\n IndexChunks (hybrid: scopes + token budget):");
    let tok = demo_tokenizer();
    let idx_chunks = index_chunks(
        "",
        "mem.rs",
        code.as_bytes(),
        "Rust",
        &tok,
        IndexChunkOptions::default(),
    );
    for (i, c) in idx_chunks.iter().take(4).enumerate() {
        println!(
            "    [I{:02}] lines {}..={} bytes {}..{} preview={}",
            i,
            c.start_line + 1,
            c.end_line + 1,
            c.start_byte,
            c.end_byte,
            c.text.lines().next().unwrap_or("").trim()
        );
    }
    let hit = idx_chunks
        .iter()
        .find(|c| c.text.contains("add(1, 2)"))
        .or_else(|| idx_chunks.first())
        .cloned();

    if let Some(hit) = hit {
        println!("\n Refill hit -> ContextChunk:");
        let ctx = refill_chunks(
            "mem.rs",
            code.as_bytes(),
            "Rust",
            &[hit],
            RefillOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to refill: {e}"))?;
        for c in ctx {
            println!(
                "    #{:02} lines {}..={} reason={} preview={}",
                c.alias,
                c.start_line + 1,
                c.end_line + 1,
                c.reason,
                c.snippet.lines().next().unwrap_or("").trim()
            );
        }
    }

    println!("\n Demo finished successfully.");
    Ok(())
}
