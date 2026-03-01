use anyhow::Result;
use llm::LLMConfig;
use react::{LunaRuntime, ReactOptions};
use std::io::{self, BufRead};

use crate::handlers;
use crate::rpc::{
    extract_id, parse_request, rpc_code_and_message, write_error, write_response, RpcErrorCode,
};
use crate::session::SessionStore;
use crate::util::demo_tokenizer;

pub fn run() -> Result<()> {
    // Note: This is a minimal MCP-like stdio JSON-RPC service.
    // When integrating with the official MCP protocol in the future, method semantics and data structures can be preserved, only replacing framing/handshake.

    let tokenizer = demo_tokenizer();
    let llm_cfg = LLMConfig::from_env().unwrap_or_default();
    let mut sessions = SessionStore::new();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let msg: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                // Parse error without id: per JSON-RPC convention, id=null
                let _ = write_error(
                    serde_json::Value::Null,
                    RpcErrorCode::ParseError.as_i64(),
                    format!("parse error: {e}"),
                );
                continue;
            }
        };

        let has_id_field = msg.get("id").is_some();
        let fallback_id = extract_id(&msg);
        let req = match parse_request(msg) {
            Ok(r) => r,
            Err(e) => {
                let (code, msg) = rpc_code_and_message(&e);
                // Notification (no id) must not be responded to (except parse error, already handled above).
                if has_id_field {
                    let _ = write_error(fallback_id, code, msg);
                }
                continue;
            }
        };

        let is_notification = req.id.is_none();
        let id = req.id_or_null();
        let method = req.method.as_str();
        let params = req.params;

        let sid = match sessions.resolve_or_create(method, &params) {
            Ok(sid) => sid,
            Err(e) => {
                let (code, msg) = rpc_code_and_message(&e);
                if !is_notification {
                    let _ = write_error(id.clone(), code, msg);
                }
                continue;
            }
        };

        let policy = match sessions.get(&sid) {
            Some(s) => s.policy.clone(),
            None => {
                if !is_notification {
                    let _ = write_error(
                        id.clone(),
                        RpcErrorCode::UnknownSession.as_i64(),
                        format!("unknown session_id: {sid}"),
                    );
                }
                continue;
            }
        };
        let runtime = LunaRuntime::new(
            tokenizer.clone(),
            llm_cfg.clone(),
            policy,
            ReactOptions::default(),
        );

        let res = handlers::dispatch(&mut sessions, &runtime, method, &params, &sid);
        match res {
            Ok(result) => {
                if !is_notification {
                    let _ = write_response(id, result);
                }
            }
            Err(e) => {
                let (code, msg) = rpc_code_and_message(&e);
                if !is_notification {
                    let _ = write_error(id, code, msg);
                }
            }
        }
    }

    Ok(())
}
