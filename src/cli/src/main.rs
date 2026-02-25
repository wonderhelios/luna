use ahash::AHashMap;
use anyhow::Result;
use clap::{Parser, Subcommand};
use core::code_chunk::{ChunkOptions, IndexChunkOptions, RefillOptions};
use intelligence::TreeSitterFile;
use llm::LLMConfig;
use react::{render_prompt_context, ReactOptions};
use std::path::PathBuf;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};
use tools::{build_context_pack_keyword, read_file, EditOp, SearchCodeOptions};
use tools::{edit_file, list_dir, run_terminal};

fn demo_tokenizer() -> Tokenizer {
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
#[command(name = "luna", about = "Luna CLI - Agentic IDE Assistant")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run built-in demo
    Demo,

    /// Search code with keyword matching
    Search {
        /// Repository root directory
        #[arg(short, long, default_value = ".")]
        repo_root: PathBuf,
        /// Search keywords
        query: Vec<String>,
        /// Output prompt context for LLM
        #[arg(long)]
        prompt: bool,
        /// Maximum number of chunks
        #[arg(long, default_value_t = 8)]
        max_chunks: usize,
    },

    /// Ask a question using the ReAct agent
    Ask {
        /// Repository root directory (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        repo_root: PathBuf,
        /// Question to ask
        question: Vec<String>,
        /// Show full prompt context
        #[arg(long)]
        show_prompt: bool,
        /// Maximum number of chunks
        #[arg(long, default_value_t = 8)]
        max_chunks: usize,
        /// Enable ReAct loop
        #[arg(long)]
        react: bool,
        /// Maximum ReAct steps
        #[arg(long, default_value_t = 3)]
        max_steps: usize,
    },

    // (Dev)
    #[command(hide = true)]
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

#[derive(Debug, Subcommand)]
enum DevCommand {
    /// List directory contents
    ListDir {
        /// Directory path
        path: PathBuf,
    },

    /// Read file content
    ReadFile {
        /// File path
        path: PathBuf,
        /// Start line (1-based)
        #[arg(long)]
        start: Option<usize>,
        /// End line (1-based)
        #[arg(long)]
        end: Option<usize>,
    },

    /// Edit file content
    EditFile {
        /// File path
        path: PathBuf,
        /// Start line (1-based)
        #[arg(long)]
        start: usize,
        /// End line (1-based)
        #[arg(long)]
        end: usize,
        /// New content
        #[arg(long)]
        content: String,
        /// Create backup
        #[arg(long)]
        backup: bool,
    },

    /// Run terminal command
    RunTerminal {
        /// Command to run
        command: Vec<String>,
        /// Working directory
        #[arg(long)]
        cwd: Option<PathBuf>,
        /// Allow dangerous commands
        #[arg(long)]
        allow_dangerous: bool,
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
        Command::Dev { command } => match command {
            DevCommand::ListDir { path } => cmd_list_dir(path),
            DevCommand::ReadFile { path, start, end } => cmd_read_file(path, start, end),
            DevCommand::EditFile {
                path,
                start,
                end,
                content,
                backup,
            } => cmd_edit_file(path, start, end, content, backup),
            DevCommand::RunTerminal {
                command,
                cwd,
                allow_dangerous,
            } => cmd_run_terminal(command, cwd, allow_dangerous),
        },
    }
}

fn cmd_demo() -> Result<()> {
    println!("Luna Intelligence Demo\n");

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

    let ts_file = TreeSitterFile::try_build(code.as_bytes(), "Rust")
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    let scope_graph = ts_file
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    println!("\n Detected Symbols (Definitions):");
    let symbols = scope_graph.symbols();

    if symbols.is_empty() {
        println!("   (No symbols found)");
    } else {
        for symbol in symbols {
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

    println!("\n Semantic Chunks:");
    let chunks = index::chunk_source("mem.rs", code.as_bytes(), "Rust", ChunkOptions::default())
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

    println!("\n Search+Refill (M1 minimal loop):");
    let tok = demo_tokenizer();
    let pack = build_context_pack_keyword(
        PathBuf::from(".").as_path(),
        "fn add",
        &tok,
        SearchCodeOptions::default(),
        IndexChunkOptions::default(),
        RefillOptions::default(),
    )?;
    println!(
        "    hits={} context={}",
        pack.hits.len(),
        pack.context.len()
    );

    println!("\n Demo finished successfully.");
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
        anyhow::bail!("query cannot be empty");
    }

    let tok = demo_tokenizer();
    let pack = build_context_pack_keyword(
        &repo_root,
        &query,
        &tok,
        SearchCodeOptions::default(),
        IndexChunkOptions::default(),
        RefillOptions::default(),
    )?;

    println!("Query: {query}\n");
    println!("Note:");
    println!("  - preview: first line of snippet");
    println!("  - trace: tool call summary");
    println!("  - Hits: keyword-matched index chunks");
    println!("  - ContextChunks: refilled semantic context blocks\n");

    for t in &pack.trace {
        println!("[trace] {:?}: {}", t.tool, t.summary);
    }

    println!("\nHits: {}", pack.hits.len());
    for (i, h) in pack.hits.iter().take(8).enumerate() {
        let lines = h.end_line.saturating_sub(h.start_line) + 1;
        println!(
            "  [H{:02}] {}:{}..={} ({} lines) preview={}",
            i,
            h.path,
            h.start_line + 1,
            h.end_line + 1,
            lines,
            h.text.lines().next().unwrap_or("").trim()
        );
    }
    if pack.hits.len() > 8 {
        println!("  ... ({} hits total)", pack.hits.len());
    }

    println!("\nContextChunks: {}", pack.context.len());
    for c in pack.context.iter().take(8) {
        let lines = c.end_line.saturating_sub(c.start_line) + 1;
        println!(
            "  #{:02} {}:{}..={} ({} lines) reason={} preview={}",
            c.alias,
            c.path,
            c.start_line + 1,
            c.end_line + 1,
            lines,
            c.reason,
            c.snippet.lines().next().unwrap_or("").trim()
        );
    }
    if pack.context.len() > 8 {
        println!("  ... ({} chunks total)", pack.context.len());
    }

    if prompt {
        let rendered = render_prompt_context(
            &repo_root,
            &pack,
            &tok,
            react::ContextEngineOptions {
                max_chunks,
                ..Default::default()
            },
        )?;
        println!("\n---\nPROMPT CONTEXT\n---\n{}", rendered);
    }

    Ok(())
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
        anyhow::bail!("question cannot be empty");
    }

    let cfg = LLMConfig::from_env()?;
    let tok = demo_tokenizer();

    if react {
        let (ans, pack, steps) = react::react_ask(
            &repo_root,
            &question,
            &tok,
            &cfg,
            ReactOptions {
                max_steps,
                context_engine: react::ContextEngineOptions {
                    max_chunks,
                    ..Default::default()
                },
                ..Default::default()
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
            let prompt_context = render_prompt_context(
                &repo_root,
                &pack,
                &tok,
                react::ContextEngineOptions {
                    max_chunks,
                    ..Default::default()
                },
            )?;
            println!("---\nPROMPT CONTEXT\n---\n{prompt_context}\n");
        }

        println!("---\nANSWER\n---\n{}", ans.trim());
        return Ok(());
    }

    // Non-react path (deprecated, will be removed)
    let pack = build_context_pack_keyword(
        &repo_root,
        &question,
        &tok,
        SearchCodeOptions::default(),
        IndexChunkOptions::default(),
        RefillOptions::default(),
    )?;

    let prompt_context = render_prompt_context(
        &repo_root,
        &pack,
        &tok,
        react::ContextEngineOptions {
            max_chunks,
            ..Default::default()
        },
    )?;

    if show_prompt {
        println!("---\nPROMPT CONTEXT\n---\n{prompt_context}\n");
    }

    let ans = llm::llm_chat(&cfg, "You are a helpful assistant.", &prompt_context)?;
    println!("---\nANSWER\n---\n{}", ans.trim());

    Ok(())
}

fn cmd_list_dir(path: PathBuf) -> Result<()> {
    println!("Listing: {:?}\n", path);

    let entries = list_dir(&path)?;

    if entries.is_empty() {
        println!("(empty directory)");
        return Ok(());
    }

    println!("{:<40} {:<10} {:<10}", "Name", "Type", "Size");
    println!("{}", "-".repeat(60));

    for entry in entries {
        let type_str = if entry.is_dir { "DIR" } else { "FILE" };
        let size_str = entry
            .size
            .map(|s| format!("{} B", s))
            .unwrap_or_else(|| "-".to_string());
        println!("{:<40} {:<10} {:<10}", entry.name, type_str, size_str);
    }

    Ok(())
}

fn cmd_read_file(path: PathBuf, start: Option<usize>, end: Option<usize>) -> Result<()> {
    let range = match (start, end) {
        (Some(s), Some(e)) => Some((s.saturating_sub(1), e.saturating_sub(1))),
        (Some(s), None) => Some((s.saturating_sub(1), s.saturating_sub(1))),
        _ => None,
    };

    let content = read_file(&path, range)?;
    println!("{}", content);

    Ok(())
}

fn cmd_edit_file(
    path: PathBuf,
    start: usize,
    end: usize,
    content: String,
    backup: bool,
) -> Result<()> {
    println!("Editing: {:?}", path);
    println!("Lines: {}..={}", start, end);
    println!("Backup: {}", backup);

    // Validate: CLI uses 1-based line numbers (minimum is 1)
    if start == 0 || end == 0 {
        anyhow::bail!("line numbers must be >= 1 (1-based)");
    }
    if start > end {
        anyhow::bail!("start line must be <= end line");
    }

    let start0 = start - 1;
    let end0 = end - 1;

    let op = EditOp::ReplaceLines {
        start_line: start0,
        end_line: end0,
        new_content: content,
    };

    let result = edit_file(&path, &op, backup)?;

    println!("\nResult:");
    println!("  Success: {}", result.success);
    if let Some(error) = result.error {
        println!("  Error: {}", error);
    }
    if let Some(lines) = result.lines_changed {
        println!("  Lines changed: {}", lines);
    }
    if let Some(backup) = result.backup_path {
        println!("  Backup: {}", backup);
    }

    Ok(())
}

fn cmd_run_terminal(
    command: Vec<String>,
    cwd: Option<PathBuf>,
    allow_dangerous: bool,
) -> Result<()> {
    let cmd_str = command.join(" ");
    println!("Running: {}", cmd_str);
    if let Some(dir) = &cwd {
        println!("CWD: {:?}", dir);
    }

    let result = run_terminal(&cmd_str, cwd.as_deref(), allow_dangerous)?;

    println!("\nResult:");
    println!("  Success: {}", result.success);
    if let Some(code) = result.exit_code {
        println!("  Exit code: {}", code);
    }

    if !result.stdout.is_empty() {
        println!("\nStdout:");
        println!("{}", result.stdout);
    }

    if !result.stderr.is_empty() {
        println!("\nStderr:");
        println!("{}", result.stderr);
    }

    if let Some(error) = result.error {
        println!("\nError: {}", error);
    }

    Ok(())
}
