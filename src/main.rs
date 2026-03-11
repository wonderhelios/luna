//! Luna - Minimal TPAR Code Assistant
//!
//! Usage:
//!   LUNA_LLM_API_KEY=sk-xxx LUNA_LLM_BASE_URL=https://api.deepseek.com/v1 cargo run

use std::env;
use std::io::{self, Write};
use std::sync::Arc;

mod llm;
mod tpar;

use llm::{OpenAIClient, OpenAIConfig};
use tpar::{TparExecutor, TurnContext};

#[tokio::main]
async fn main() {
    println!("🌙 Luna - Minimal TPAR Code Assistant\n");

    // Setup tracing
    tracing_subscriber::fmt::init();

    // Get LLM client from env
    let llm: Arc<dyn llm::LLMClient> = match OpenAIClient::try_from_env() {
        Some(client) => Arc::new(client),
        None => {
            eprintln!("❌ No LLM configured. Set LUNA_LLM_API_KEY environment variable.");
            eprintln!("   Example: LUNA_LLM_API_KEY=sk-xxx LUNA_LLM_BASE_URL=https://api.deepseek.com/v1");
            std::process::exit(1);
        }
    };

    // Get current directory
    let cwd = env::current_dir().expect("Failed to get current directory");

    // Create TPAR executor
    let executor = TparExecutor::new(llm);

    println!("💡 Type your questions or commands. Type 'quit' to exit.");
    println!("   Working directory: {}\n", cwd.display());

    // REPL loop
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();

                if input.is_empty() {
                    continue;
                }

                if input == "quit" || input == "exit" {
                    println!("👋 Goodbye!");
                    break;
                }

                // Enhance input with current directory context
                let enhanced_input = format!(
                    "{}\n\n[SYSTEM: Current working directory is: {}]",
                    input,
                    cwd.display()
                );

                // Create turn context
                let ctx = TurnContext::new(Some(cwd.clone()));

                // Execute
                print!("🤔 Thinking... ");
                io::stdout().flush().unwrap();

                match executor.run_turn(&enhanced_input, &ctx) {
                    Ok(response) => {
                        println!("\r🤖 {}\n", response);
                    }
                    Err(e) => {
                        println!("\r❌ Error: {}\n", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Input error: {}", e);
            }
        }
    }
}
