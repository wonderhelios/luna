use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use std::io::{self, Write};

use error::LunaError;
use session::SessionError;

/// JSON-RPC 2.0 标准错误码：
/// -32700/-32600/-32601/-32602/-32603
///
/// 服务器自定义错误码建议使用 -32099..-32000 区间。
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[repr(i64)]
pub enum RpcErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,

    // Server-defined errors (reserved range)
    UnknownSession = -32001,
    UnknownConfirmation = -32002,

    // Domain-level errors mapped from LunaError / SessionError
    ConfigError = -32010,
    ValidationError = -32011,
    SearchError = -32012,
    ToolError = -32013,
    SessionError = -32014,
    NotFound = -32015,
    PermissionDenied = -32016,
    Timeout = -32017,
}

impl RpcErrorCode {
    pub fn as_i64(self) -> i64 {
        self as i64
    }

    pub fn from_luna_error(e: &LunaError) -> Self {
        match e {
            LunaError::Config { .. } => RpcErrorCode::ConfigError,
            LunaError::Validation { .. } => RpcErrorCode::ValidationError,
            LunaError::Search { .. } => RpcErrorCode::SearchError,
            LunaError::Tool { .. } => RpcErrorCode::ToolError,
            LunaError::Session { .. } => RpcErrorCode::SessionError,
            LunaError::NotFound { .. } => RpcErrorCode::NotFound,
            LunaError::Permission { .. } => RpcErrorCode::PermissionDenied,
            LunaError::Timeout { .. } => RpcErrorCode::Timeout,
            _ => RpcErrorCode::InternalError,
        }
    }
}

#[derive(Debug)]
struct RpcErrorTagged {
    code: i64,
    message: String,
}

impl std::fmt::Display for RpcErrorTagged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RpcErrorTagged {}

pub fn rpc_err(code: RpcErrorCode, message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(RpcErrorTagged {
        code: code.as_i64(),
        message: message.into(),
    })
}

pub fn rpc_code_and_message(e: &anyhow::Error) -> (i64, String) {
    if let Some(tagged) = e.downcast_ref::<RpcErrorTagged>() {
        return (tagged.code, tagged.message.clone());
    }
    if let Some(luna) = e.downcast_ref::<LunaError>() {
        let code = RpcErrorCode::from_luna_error(luna).as_i64();
        return (code, luna.to_string());
    }
    if let Some(sess) = e.downcast_ref::<SessionError>() {
        return (RpcErrorCode::SessionError.as_i64(), sess.to_string());
    }
    (RpcErrorCode::InternalError.as_i64(), e.to_string())
}

/// Extract `id` from a raw JSON value (best-effort).
///
/// For parse/invalid-request errors we still want to respond with a stable id when possible.
pub fn extract_id(v: &serde_json::Value) -> serde_json::Value {
    v.get("id").cloned().unwrap_or(serde_json::Value::Null)
}

/// A minimal JSON-RPC request envelope.
///
/// Notes:
/// - We accept missing `jsonrpc` for compatibility with early clients.
/// - `id` may be absent (notification); we still echo `null` in responses.
#[derive(Debug, Clone, Deserialize)]
pub struct RpcRequest {
    #[serde(default)]
    pub jsonrpc: Option<String>,
    #[serde(default)]
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

impl RpcRequest {
    pub fn validate(&self) -> anyhow::Result<()> {
        // Strict JSON-RPC 2.0: require explicit version marker.
        let Some(v) = &self.jsonrpc else {
            return Err(rpc_err(
                RpcErrorCode::InvalidRequest,
                "missing jsonrpc field (expected '2.0')",
            ));
        };
        if v != "2.0" {
            return Err(rpc_err(
                RpcErrorCode::InvalidRequest,
                format!("invalid jsonrpc version: {v}"),
            ));
        }
        if self.method.trim().is_empty() {
            return Err(rpc_err(
                RpcErrorCode::InvalidRequest,
                "missing or empty method",
            ));
        }
        Ok(())
    }

    pub fn id_or_null(&self) -> serde_json::Value {
        self.id.clone().unwrap_or(serde_json::Value::Null)
    }
}

pub fn parse_request(v: serde_json::Value) -> anyhow::Result<RpcRequest> {
    let req: RpcRequest = serde_json::from_value(v).map_err(|e| {
        rpc_err(
            RpcErrorCode::InvalidRequest,
            format!("invalid request: {e}"),
        )
    })?;
    req.validate()?;
    Ok(req)
}

/// Parse params to a strongly typed struct.
///
/// - `params: null` is treated as `{}` to be forgiving for param-less methods.
pub fn parse_params<T: DeserializeOwned>(params: &serde_json::Value) -> anyhow::Result<T> {
    let v = if params.is_null() {
        json!({})
    } else {
        params.clone()
    };
    serde_json::from_value(v)
        .map_err(|e| rpc_err(RpcErrorCode::InvalidParams, format!("invalid params: {e}")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

pub fn write_response(id: serde_json::Value, result: serde_json::Value) -> io::Result<()> {
    let out = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &out)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

pub fn write_error(id: serde_json::Value, code: i64, message: impl Into<String>) -> io::Result<()> {
    let out = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": RpcError { code, message: message.into(), data: None },
    });
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &out)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn test_parse_request_requires_jsonrpc() {
        let v = json!({"id": 1, "method": "tools/list", "params": {}});
        let err = parse_request(v).unwrap_err();
        let (code, _msg) = rpc_code_and_message(&err);
        assert_eq!(code, RpcErrorCode::InvalidRequest.as_i64());
    }

    #[test]
    fn test_parse_request_accepts_jsonrpc_2_0() {
        let v = json!({"jsonrpc":"2.0","id": 1, "method": "tools/list", "params": {}});
        let req = parse_request(v).unwrap();
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id_or_null(), json!(1));
    }

    #[test]
    fn test_parse_params_invalid_params_code() {
        #[derive(Debug, Deserialize)]
        struct P {
            #[allow(dead_code)]
            name: String,
        }
        let err = parse_params::<P>(&json!({"name": 123})).unwrap_err();
        let (code, _msg) = rpc_code_and_message(&err);
        assert_eq!(code, RpcErrorCode::InvalidParams.as_i64());
    }
}
