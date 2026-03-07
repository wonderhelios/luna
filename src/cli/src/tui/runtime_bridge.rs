use std::path::PathBuf;
use std::sync::Arc;

use runtime::{LunaRuntime, RunRequest, SessionRef};

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

/// Run a single turn in a blocking thread, returning (new_session_id, output).
pub fn run_turn_blocking(
    handle: tokio::runtime::Handle,
    runtime: Arc<LunaRuntime>,
    session_id: Option<String>,
    cwd: Option<PathBuf>,
    input: String,
) -> error::Result<(String, String)> {
    let req = build_request(session_id.as_deref(), cwd.as_ref(), &input);
    let resp = handle.block_on(runtime.run(req))?;
    Ok((resp.session_id, resp.output))
}
