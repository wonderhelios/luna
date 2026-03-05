use runtime::{LunaRuntime, RunRequest, SessionRef};

#[tokio::main]
async fn main() {
    println!("🌙 Luna - AI Code Assistant");
    println!("Type 'exit' to quit\n");

    let runtime = LunaRuntime::new();
    let cwd = std::env::current_dir().ok();
    let mut session_id: Option<String> = None;

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "/help" => {
                println!("Commands:\n  /help   Show this help\n  exit    Quit\n  quit    Quit\n");
            }
            "" => continue,
            _ => {
                let session = match &session_id {
                    Some(id) => SessionRef::Existing {
                        session_id: id.clone(),
                    },
                    None => SessionRef::New { title: None },
                };

                let mut req = RunRequest::chat_turn(session, input);
                if let Some(cwd) = &cwd {
                    req = req.with_cwd(cwd.clone());
                }

                match runtime.run(req).await {
                    Ok(resp) => {
                        session_id = Some(resp.session_id);
                        println!("{}\n", resp.output);
                    }
                    Err(err) => {
                        eprintln!("Error: {err}\n");
                    }
                }
            }
        }
    }
}
