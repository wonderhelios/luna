use std::path::PathBuf;
use std::sync::Arc;

use runtime::{LunaRuntime, RunRequest, SessionRef};
use crate::tui::CancelToken;

pub fn build_request(session_id: Option<&str>, cwd: Option<&PathBuf>, input: &str) -> RunRequest {
    let session = match session_id {
        Some(id) => SessionRef::Existing {
            session_id: id.to_owned(),
        },
        None => SessionRef::New { title: None },
    };
    let mut req = RunRequest::chat_turn(session, input);
    if let Some(cwd) = cwd {
        req = req.with_cwd(cwd.clone());
    }
    req
}

/// Run a single turn in a blocking thread, streaming RuntimeEvent to `event_tx`.
///
/// Supports cancellation via `cancel` token (checked between events).
pub fn run_turn_blocking_with_events(
    handle: tokio::runtime::Handle,
    runtime: Arc<LunaRuntime>,
    session_id: Option<String>,
    cwd: Option<PathBuf>,
    input: String,
    event_tx: tokio::sync::mpsc::Sender<runtime::RuntimeEvent>,
    _cancel: CancelToken,
) -> error::Result<(String, String)> {
    let req = build_request(session_id.as_deref(), cwd.as_ref(), &input);
    let resp = handle.block_on(runtime.run_with_event_hook(req, |ev| {
        // Bounded channel: use try_send to avoid blocking
        let _ = event_tx.try_send(ev.clone());
    }))?;
    Ok((resp.session_id, resp.output))
}
