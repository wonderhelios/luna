use ahash::AHashMap;
use anyhow::Result;
use clap::{Parser, Subcommand};
use core::code_chunk::{ChunkOptions, IndexChunkOptions, RefillOptions};
use index::chunk_source;
use intelligence::TreeSitterFile;
use std::path::PathBuf;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

fn demo_tokenizer() -> Tokenizer {
    // Demo minimalist tokenizer (real scenarios should reuse embedder's tokenizer)
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
    /// Run built-in demo (demonstrate parsing/chunking/minimal search+refill loop)
    Demo,
    /// Perform keyword placeholder search under repo_root, output IndexChunk hits and Refilled ContextChunk
    Search {
        /// Repository root directory
        repo_root: PathBuf,
        /// Search keywords (multiple words allowed)
        query: Vec<String>,
        /// Additionally output context text for LLM injection
        #[arg(long)]
        prompt: bool,
        /// Maximum number of ContextChunks in prompt output (default: 8)
        #[arg(long, default_value_t = 8)]
        max_chunks: usize,
    },
    /// Assemble the context based on the search+refill+context engine and call LLM to provide the answer
    Ask {
        repo_root: PathBuf,
        question: Vec<String>,
        #[arg(long)]
        show_prompt: bool,
        #[arg(long, default_value_t = 8)]
        max_chunks: usize,
        // enable ReAct Loop (let model to decide go on or not)
        #[arg(long)]
        react: bool,
        // max times for react call
        #[arg(long, default_value_t = 3)]
        max_steps: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Demo) {
        Command::Demo => cmd_demo(),
        Command::Search {
            repo_root,
            query,
            prompt,
            max_chunks,
        } => cmd_search(repo_root, query, prompt, max_chunks),
        Command::Ask {
            repo_root,
            question,
            show_prompt,
            max_chunks,
            react,
            max_steps,
        } => cmd_ask(
            repo_root,
            question,
            show_prompt,
            max_chunks,
            react,
            max_steps,
        ),
    }
}

fn cmd_ask(
    repo_root: PathBuf,
    question: Vec<String>,
    show_prompt: bool,
    max_chunks: usize,
    react: bool,
    max_steps: usize,
) -> Result<()> {
    let question = question.join(" ");
    if question.trim().is_empty() {
        anyhow::bail!("question can not be empty");
    }
    let tok = demo_tokenizer();
    let cfg = agent::LLMConfig::from_env()?;

    if react {
        let (ans, pack, steps) = agent::react_ask(
            &repo_root,
            &question,
            &tok,
            &cfg,
            agent::ReActOptions {
                max_steps,
                context_engine: agent::ContextEngineOptions {
                    max_chunks,
                    ..Default::default()
                },
            },
        )?;

        println!("---\nTRACE\n---");
        for st in &steps {
            println!(
                "[step {}] action={:?} obs={}",
                st.step, st.action, st.observation
            );
        }

        if show_prompt {
            let prompt_context = agent::render_prompt_context(
                &repo_root,
                &pack,
                &tok,
                agent::ContextEngineOptions {
                    max_chunks,
                    ..Default::default()
                },
            )?;
            println!("---\nPROMPT CONTEXT\n---\n{prompt_context}\n");
        }
        println!("---\nANSWER\n--\n{}", ans.trim());
        return Ok(());
    }

    let search_query = {
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in question.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                cur.push(ch);
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        let out = out
            .into_iter()
            .filter(|s| {
                s.chars()
                    .next()
                    .map(|c| c.is_ascii_alphabetic() || c == '_')
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if out.is_empty() {
            question.clone()
        } else {
            out.join(" ")
        }
    };

    let (hits, mut trace) = agent::search_code_keyword(
        &repo_root,
        &search_query,
        &tok,
        IndexChunkOptions::default(),
        agent::SearchCodeOptions::default(),
    )?;

    let (context, mut trace2) = agent::refill_hits(&repo_root, &hits, RefillOptions::default())?;
    trace.append(&mut trace2);
    let pack = agent::ContextPack {
        query: question.clone(),
        hits,
        context,
        trace,
    };

    let prompt_context = agent::render_prompt_context(
        &repo_root,
        &pack,
        &tok,
        agent::ContextEngineOptions {
            max_chunks,
            ..Default::default()
        },
    )?;

    if show_prompt {
        println!("---\nPROMPT CONTEXT\n---\n{prompt_context}\n");
    }

    let ans = agent::llm_answer(&cfg, &question, &prompt_context)?;
    println!("---\nANSWER\n---\n{}", ans.trim());
    Ok(())
}

fn cmd_search(
    repo_root: PathBuf,
    query: Vec<String>,
    prompt: bool,
    max_chunks: usize,
) -> Result<()> {
    let query = query.join(" ");
    if query.trim().is_empty() {
        anyhow::bail!("query cannot be empty (example: luna-server search . index_chunks)");
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
        "Note:\n  - preview: only first line of snippet (for quick overview)\n  - trace: tool call summary\n  - Hits: keyword-matched index chunks (IndexChunk, possibly multiple fragments in same function)\n  - ContextChunks: refilled semantic context blocks (ContextChunk, deduped by range, ready for LLM injection)\n"
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

    if prompt {
        let rendered = agent::render_prompt_context(
            &repo_root,
            &pack,
            &tok,
            agent::ContextEngineOptions {
                max_chunks,
                ..Default::default()
            },
        )?;
        println!("\n---\nPROMPT CONTEXT\n---\n{}", rendered);
    }
    Ok(())
}

fn cmd_demo() -> Result<()> {
    println!("Luna Intelligence Demo\n");

    // 1. Prepare test code (Rust)
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

    // 2. Parse using Intelligence module
    // "Rust" is language ID, corresponding to internal registration in xc-intelligence
    let ts_file = TreeSitterFile::try_build(code.as_bytes(), "Rust")
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    // 3. Get Scope Graph (core capability: understand scopes, definitions, and references)
    let scope_graph = ts_file
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    // 4. Print all detected symbols (definitions)
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

    // 5. Intelligent Chunking Demo (for subsequent Index/RAG/context injection)
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

    // 6. IndexChunk -> Refill -> ContextChunk Demo (calling agent's minimal loop)
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
