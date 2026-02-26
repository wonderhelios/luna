use anyhow::Result;
use core::code_chunk::{IndexChunk, IndexChunkOptions, RefillOptions};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use react::LunaRuntime;

use crate::rpc::{parse_params, rpc_err, RpcErrorCode};
use crate::session::{PendingToolCall, SessionMetadata, SessionState, SessionStore};
use crate::util::{
    apply_policy_patch, parse_policy_overrides, repo_root_from_opt, session_id_from_params,
};
use crate::virtual_tools::{
    filter_schemas_by_policy, refill_tool_schema, search_tool_schema, tool_output_like,
};

#[derive(Debug, Clone, Copy)]
enum Method {
    Initialize,
    ToolsList,
    ToolsCall,
    ToolsConfirm,
    AgentAsk,
    SearchCodeKeyword,
    RefillHits,
}

impl TryFrom<&str> for Method {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "initialize" => Ok(Method::Initialize),
            "tools/list" => Ok(Method::ToolsList),
            "tools/call" => Ok(Method::ToolsCall),
            "tools/confirm" => Ok(Method::ToolsConfirm),
            "agent/ask" => Ok(Method::AgentAsk),
            "search_code_keyword" => Ok(Method::SearchCodeKeyword),
            "refill_hits" => Ok(Method::RefillHits),
            _ => Err(rpc_err(
                RpcErrorCode::MethodNotFound,
                format!("method not found: {value}"),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ToolsCallParams {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
    #[serde(default)]
    repo_root: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolsConfirmParams {
    confirmation_id: String,
}

#[derive(Debug, Deserialize)]
struct AgentAskParams {
    question: String,
    #[serde(default)]
    repo_root: Option<String>,
    #[serde(default)]
    active_context: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SearchCodeKeywordParams {
    query: String,
    #[serde(default)]
    repo_root: Option<String>,
    #[serde(default)]
    max_hits: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RefillHitsParams {
    hits: Vec<IndexChunk>,
    #[serde(default)]
    repo_root: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default)]
    max_hits: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RefillArgs {
    hits: Vec<IndexChunk>,
}

pub fn dispatch(
    sessions: &mut SessionStore,
    runtime: &LunaRuntime,
    method: &str,
    params: &serde_json::Value,
    sid: &str,
) -> Result<serde_json::Value> {
    let method = Method::try_from(method)?;
    match method {
        Method::Initialize => handle_initialize(sessions, params, sid),
        Method::ToolsList => handle_tools_list(runtime, sid),
        Method::ToolsCall => handle_tools_call(sessions, runtime, params, sid),
        Method::ToolsConfirm => handle_tools_confirm(sessions, runtime, params, sid),
        Method::AgentAsk => handle_agent_ask(runtime, params, sid),
        Method::SearchCodeKeyword => handle_search_code_keyword(runtime, params),
        Method::RefillHits => handle_refill_hits(runtime, params),
    }
}

fn handle_initialize(
    sessions: &mut SessionStore,
    params: &serde_json::Value,
    sid: &str,
) -> Result<serde_json::Value> {
    // Allow client to specify explicit session_id in initialize.
    let init_sid = session_id_from_params(params).unwrap_or_else(|| sid.to_string());
    let mut state = sessions.get(&init_sid).cloned().unwrap_or(SessionState {
        policy: toolkit::ExecutionPolicy::default(),
        pending: std::collections::HashMap::new(),
        metadata: SessionMetadata::default(),
    });
    if let Some(patch) = parse_policy_overrides(params) {
        state.policy = apply_policy_patch(state.policy.clone(), patch);
    }
    sessions.upsert(init_sid.clone(), state.clone());
    Ok(json!({
        "name": "luna-server",
        "version": "0.1.0",
        "capabilities": {
            "tools": true,
            "ask": true,
            "search": true,
            "refill": true,
            "confirm": true
        },
        "session_id": init_sid,
        "policy": state.policy,
    }))
}

fn handle_tools_list(runtime: &LunaRuntime, sid: &str) -> Result<serde_json::Value> {
    let mut schemas = filter_schemas_by_policy(runtime.tool_schemas(), runtime.policy());
    // Server-side virtual tools (share the same tools/call channel)
    schemas.push(search_tool_schema());
    schemas.push(refill_tool_schema());
    Ok(json!({
        "tools": schemas,
        "policy": runtime.policy(),
        "session_id": sid,
    }))
}

fn handle_tools_call(
    sessions: &mut SessionStore,
    runtime: &LunaRuntime,
    params: &serde_json::Value,
    sid: &str,
) -> Result<serde_json::Value> {
    let p: ToolsCallParams = parse_params(params)?;
    let repo_root = repo_root_from_opt(p.repo_root);

    // Virtual tools
    if p.name == "search_code" {
        let a: SearchArgs = parse_params(&p.arguments)?;
        let mut opt = tools::SearchCodeOptions::default();
        if let Some(mh) = a.max_hits {
            opt.max_hits = mh;
        }
        let (hits, trace) =
            runtime.search_code_keyword(&repo_root, &a.query, IndexChunkOptions::default(), opt)?;
        return Ok(tool_output_like(
            true,
            json!({"hits": hits, "trace": trace}),
            "ok",
            None,
        ));
    }
    if p.name == "refill_hits" {
        let a: RefillArgs = parse_params(&p.arguments)?;
        let (context, trace) =
            runtime.refill_hits(&repo_root, &a.hits, RefillOptions::default())?;
        return Ok(tool_output_like(
            true,
            json!({"context": context, "trace": trace}),
            "ok",
            None,
        ));
    }

    let out = runtime.execute_tool(&p.name, repo_root.clone(), p.arguments.clone());

    // Human-in-the-loop: if tool reports confirmation required, mint confirmation_id and store pending call.
    if out.trace == "confirmation_required" {
        let confirmation_id = Uuid::new_v4().to_string();
        let state = sessions.get_mut(sid).ok_or_else(|| {
            rpc_err(
                RpcErrorCode::UnknownSession,
                format!("unknown session_id: {sid}"),
            )
        })?;
        let repo_root_saved = repo_root.clone();
        let args_saved = p.arguments.clone();
        state.pending.insert(
            confirmation_id.clone(),
            PendingToolCall {
                name: p.name.clone(),
                repo_root: repo_root_saved.clone(),
                arguments: args_saved.clone(),
            },
        );
        return Ok(tool_output_like(
            false,
            json!({
                "needs_confirmation": true,
                "confirmation_id": confirmation_id,
                "tool": {"name": p.name, "repo_root": repo_root_saved, "arguments": args_saved},
            }),
            "confirmation_required",
            out.error.clone(),
        ));
    }

    Ok(serde_json::to_value(out)?)
}

fn handle_tools_confirm(
    sessions: &mut SessionStore,
    runtime: &LunaRuntime,
    params: &serde_json::Value,
    sid: &str,
) -> Result<serde_json::Value> {
    let p: ToolsConfirmParams = parse_params(params)?;
    let state = sessions.get_mut(sid).ok_or_else(|| {
        rpc_err(
            RpcErrorCode::UnknownSession,
            format!("unknown session_id: {sid}"),
        )
    })?;

    let pending = state.pending.remove(&p.confirmation_id).ok_or_else(|| {
        rpc_err(
            RpcErrorCode::UnknownConfirmation,
            format!("unknown confirmation_id: {}", p.confirmation_id),
        )
    })?;

    // Force confirm=true when executing pending call.
    let mut args = pending.arguments;
    if let Some(obj) = args.as_object_mut() {
        obj.insert("confirm".to_string(), serde_json::Value::Bool(true));
    }
    let out = runtime.execute_tool(&pending.name, pending.repo_root, args);
    Ok(serde_json::to_value(out)?)
}

fn handle_agent_ask(
    runtime: &LunaRuntime,
    params: &serde_json::Value,
    sid: &str,
) -> Result<serde_json::Value> {
    let p: AgentAskParams = parse_params(params)?;
    let repo_root = repo_root_from_opt(p.repo_root);
    let (answer, pack, steps) = runtime.ask_react(&repo_root, &p.question)?;
    Ok(json!({
        "answer": answer,
        "context_pack": pack,
        "steps": steps,
        "session_id": sid,
        "active_context": p.active_context,
    }))
}

fn handle_search_code_keyword(
    runtime: &LunaRuntime,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let p: SearchCodeKeywordParams = parse_params(params)?;
    let repo_root = repo_root_from_opt(p.repo_root);
    let mut opt = tools::SearchCodeOptions::default();
    if let Some(mh) = p.max_hits {
        opt.max_hits = mh;
    }
    let (hits, trace) =
        runtime.search_code_keyword(&repo_root, &p.query, IndexChunkOptions::default(), opt)?;
    Ok(json!({"hits": hits, "trace": trace}))
}

fn handle_refill_hits(
    runtime: &LunaRuntime,
    params: &serde_json::Value,
) -> Result<serde_json::Value> {
    let p: RefillHitsParams = parse_params(params)?;
    let repo_root = repo_root_from_opt(p.repo_root);
    let (context, trace) = runtime.refill_hits(&repo_root, &p.hits, RefillOptions::default())?;
    Ok(json!({"context": context, "trace": trace}))
}
