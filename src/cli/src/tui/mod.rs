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

use runtime::{LunaRuntime, RuntimeEvent};

mod runtime_bridge;
mod state;
mod ui;

/// UI message types
///
/// Uses bounded channel to prevent unbounded memory growth.
/// Old events are dropped when UI rendering lags behind.
#[allow(clippy::enum_variant_names)]
enum UiMsg {
    Ok {
        session_id: String,
        output: String,
        from_command: bool,
    },
    Err {
        err: String,
    },
    /// Streaming event - updates status bar in real-time
    Event {
        event: RuntimeEvent,
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

/// Formats RuntimeEvent into status bar text
fn format_event_status(event: &RuntimeEvent) -> String {
    match event {
        RuntimeEvent::TparTaskClassified { task } => format!("[Task] {task}"),
        RuntimeEvent::TparPlanBuilt { plan } => format!("[Plan] {plan}"),
        RuntimeEvent::TparStepStarted { step_id, step } => {
            format!("[Step {step_id}] {step} ...")
        }
        RuntimeEvent::TparStepCompleted { step_id, ok } => {
            format!("[Step {step_id}] {}", if *ok { "ok" } else { "failed" })
        }
        RuntimeEvent::TparReviewed { ok } => {
            format!("[Review] {}", if *ok { "ok" } else { "needs revision" })
        }
        RuntimeEvent::ScopeGraphSearchStarted { repo_root } => {
            format!("[ScopeGraph] searching in {repo_root}")
        }
        RuntimeEvent::ScopeGraphSearchCompleted { matches } => {
            format!("[ScopeGraph] {matches} match(es)")
        }
        RuntimeEvent::FoundIdentifier { name } => format!("[Symbol] {name}"),
        RuntimeEvent::SessionCreated { session_id } => {
            format!("[Session] created: {session_id}")
        }
        RuntimeEvent::SessionLoaded { session_id } => {
            format!("[Session] loaded: {session_id}")
        }
        RuntimeEvent::UserMessageAppended => "[Msg] User".to_owned(),
        RuntimeEvent::AssistantMessageAppended => "[Msg] Assistant".to_owned(),
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
    // Use bounded channel to prevent unbounded memory growth
    // Buffer size 128 is sufficient for smooth UI while avoiding memory bloat
    let (tx, mut rx) = mpsc::channel::<UiMsg>(128);

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        tokio::select! {
            maybe_ev = reader.next() => {
                let Some(ev) = maybe_ev else { continue };
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
                    handle_ui_msg(&mut app, msg);
                }
            }
        }
        clamp_scroll(&mut app, terminal.size()?);
    }

    Ok(())
}

/// Handles UI messages
///
/// Extracted from main loop to simplify control flow
fn handle_ui_msg(app: &mut state::AppState, msg: UiMsg) {
    match msg {
        UiMsg::Ok {
            session_id,
            output,
            from_command,
        } => {
            app.session_id = Some(session_id);
            app.busy = false;

            // For slash commands, keep the result visible in status bar as well.
            if from_command {
                let trimmed = output.trim_end();
                app.status = if trimmed.is_empty() {
                    "Command completed (no output)".to_owned()
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
        UiMsg::Event { event } => {
            // Streaming events only update status bar, don't reset busy
            app.status = format_event_status(&event);
        }
        UiMsg::Err { err } => {
            app.busy = false;
            app.status.clear();
            app.push_system(format!("❌ Error: {err}"));
        }
    }
}

/// Cancellation token
///
/// Used to interrupt long-running tasks (reserved for future use)
#[derive(Clone)]
#[allow(dead_code)]
pub struct CancelToken {
    inner: Arc<tokio::sync::RwLock<bool>>,
}

impl CancelToken {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            inner: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    #[allow(dead_code)]
    async fn cancel(&self) {
        *self.inner.write().await = true;
    }

    #[allow(dead_code)]
    async fn is_cancelled(&self) -> bool {
        *self.inner.read().await
    }
}

async fn handle_key(
    app: &mut state::AppState,
    key: KeyEvent,
    tx: mpsc::Sender<UiMsg>,
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

            let is_slash_command = input.starts_with('/');
            if is_slash_command {
                app.status = format!("Running command: {input}");
            } else {
                app.push_user(input.clone());
            }
            app.clear_input();
            app.busy = true;

            let handle = tokio::runtime::Handle::current();
            let runtime = Arc::clone(&app.runtime);
            let session_id = app.session_id.clone();
            let cwd = app.cwd.clone();

            // Create cancellation token
            let cancel = CancelToken::new();
            app.cancel_token = Some(cancel.clone());

            // Bounded channel for event streaming, drops old events when full
            let (event_tx, mut event_rx) = mpsc::channel::<RuntimeEvent>(64);
            let ui_tx = tx.clone();

            // Event forwarding task
            tokio::spawn(async move {
                while let Some(ev) = event_rx.recv().await {
                    // Use try_send for non-blocking send, drop when full
                    if ui_tx.try_send(UiMsg::Event { event: ev }).is_err() {
                        // Channel full, event dropped - UI will catch up
                        break;
                    }
                }
            });

            tokio::task::spawn_blocking(move || {
                let res = runtime_bridge::run_turn_blocking_with_events(
                    handle, runtime, session_id, cwd, input, event_tx, cancel,
                );
                match res {
                    Ok((session_id, output)) => {
                        let _ = tx.try_send(UiMsg::Ok {
                            session_id,
                            output,
                            from_command: is_slash_command,
                        });
                    }
                    Err(err) => {
                        let _ = tx.try_send(UiMsg::Err {
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
