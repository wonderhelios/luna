use std::io;
use std::sync::Arc;

use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use runtime::LunaRuntime;

mod runtime_bridge;
mod state;
mod ui;

enum UiMsg {
    RuntimeOk {
        session_id: String,
        output: String,
        from_command: bool,
    },
    RuntimeErr {
        err: String,
    },
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> error::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

pub async fn run(runtime: Arc<LunaRuntime>, cwd: Option<std::path::PathBuf>) -> error::Result<()> {
    // Avoid ANSI sequences breaking TUI. (We can add ANSI->Span later.)
    std::env::set_var("NO_COLOR", "1");

    let _guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = state::AppState::new(runtime, cwd);

    let mut reader = EventStream::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<UiMsg>();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        tokio::select! {
            maybe_ev = reader.next() => {
                let Some(ev) = maybe_ev else { continue; };
                match ev {
                    Ok(Event::Key(key)) => {
                        if handle_key(&mut app, key, tx.clone()).await? {
                            break;
                        }
                    }
                    Ok(Event::Resize(_, _)) => {
                        // redraw on next loop
                    }
                    _ => {}
                }
            }
            maybe_msg = rx.recv() => {
                if let Some(msg) = maybe_msg {
                    app.busy = false;
                    match msg {
                        UiMsg::RuntimeOk { session_id, output, from_command } => {
                            app.session_id = Some(session_id);

                            // For slash commands, keep the result visible in status bar as well.
                            // This avoids "no output" perception when users don't echo commands.
                            if from_command {
                                let trimmed = output.trim_end();
                                app.status = if trimmed.is_empty() {
                                    "命令执行完成（无输出）".to_owned()
                                } else {
                                    trimmed.to_owned()
                                };
                            } else {
                                app.status.clear();
                            }

                            app.push_assistant(output);
                            // auto scroll to bottom
                            app.scroll_y = usize::MAX / 2;
                        }
                        UiMsg::RuntimeErr { err } => {
                            app.status.clear();
                            app.push_system(format!("❌ Error: {err}"));
                        }
                    }
                }
            }
        }
        clamp_scroll(&mut app, terminal.size()?);
    }

    Ok(())
}

async fn handle_key(
    app: &mut state::AppState,
    key: KeyEvent,
    tx: mpsc::UnboundedSender<UiMsg>,
) -> error::Result<bool> {
    // Exit
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }

    match key.code {
        KeyCode::Enter => {
            if app.busy {
                return Ok(false);
            }
            let input = app.input.trim().to_owned();
            if input.is_empty() {
                return Ok(false);
            }
            // Support legacy exit/quit.
            if input == "exit" || input == "quit" {
                return Ok(true);
            }

            // UX: 对于 `/sessions`、`/switch` 这类“控制命令”，
            // 不把用户输入回显到聊天区，只展示执行结果即可。
            let is_slash_command = input.starts_with('/');
            if is_slash_command {
                app.status = format!("运行命令：{input}");
            } else {
                app.push_user(input.clone());
            }
            app.clear_input();
            app.busy = true;

            let handle = tokio::runtime::Handle::current();
            let runtime = Arc::clone(&app.runtime);
            let session_id = app.session_id.clone();
            let cwd = app.cwd.clone();
            tokio::task::spawn_blocking(move || {
                let res =
                    runtime_bridge::run_turn_blocking(handle, runtime, session_id, cwd, input);
                match res {
                    Ok((session_id, output)) => {
                        let _ = tx.send(UiMsg::RuntimeOk {
                            session_id,
                            output,
                            from_command: is_slash_command,
                        });
                    }
                    Err(err) => {
                        let _ = tx.send(UiMsg::RuntimeErr {
                            err: err.to_string(),
                        });
                    }
                }
            });
        }
        KeyCode::Backspace => app.input_backspace(),
        KeyCode::Left => app.input_move_left(),
        KeyCode::Right => app.input_move_right(),
        KeyCode::PageUp => app.scroll_y = app.scroll_y.saturating_sub(10),
        KeyCode::PageDown => app.scroll_y = app.scroll_y.saturating_add(10),
        KeyCode::Up => app.scroll_y = app.scroll_y.saturating_sub(1),
        KeyCode::Down => app.scroll_y = app.scroll_y.saturating_add(1),
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                app.input_insert(ch);
            }
        }
        _ => {}
    }

    Ok(false)
}

fn clamp_scroll(app: &mut state::AppState, size: ratatui::prelude::Size) {
    // Very rough clamp based on raw lines; good enough for MVP.
    // Layout: chat(min) + input(3) + status(1)
    // Borders take extra lines; we keep it simple here.
    let chat_height = size.height.saturating_sub(4) as usize;
    if chat_height == 0 {
        app.scroll_y = 0;
        return;
    }
    // Count raw lines.
    let mut total = 0usize;
    for m in &app.messages {
        total += m.content.lines().count().max(1) + 1;
    }
    if total > chat_height {
        let max_scroll = total - chat_height;
        app.scroll_y = app.scroll_y.min(max_scroll);
    } else {
        app.scroll_y = 0;
    }
}
