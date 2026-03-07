use runtime::LunaRuntime;
use std::sync::Arc;

mod tui;

#[tokio::main]
async fn main() {
    let runtime = Arc::new(LunaRuntime::new());
    let cwd = std::env::current_dir().ok();

    if let Err(err) = tui::run(runtime, cwd).await {
        eprintln!("Error: {err}");
    }
}
