use serde::Serialize;
use serde_json::json;
use toolkit::ExecutionPolicy;

pub fn filter_schemas_by_policy(
    schemas: Vec<toolkit::ToolSchema>,
    policy: &ExecutionPolicy,
) -> Vec<toolkit::ToolSchema> {
    schemas
        .into_iter()
        .filter(|s| {
            if s.name == "run_terminal" {
                return policy.allow_run_terminal;
            }
            if s.name == "edit_file" {
                return policy.allow_edit_file;
            }
            true
        })
        .collect()
}

#[derive(Debug, Serialize)]
struct ToolOutputLike {
    success: bool,
    data: serde_json::Value,
    error: Option<String>,
    trace: String,
    context_chunks: Vec<serde_json::Value>,
    hits: Vec<serde_json::Value>,
}

pub fn tool_output_like(
    success: bool,
    data: serde_json::Value,
    trace: &str,
    error: Option<String>,
) -> serde_json::Value {
    // 这里理论上不会失败；失败时退化到 json! 以保证服务可用。
    let v = ToolOutputLike {
        success,
        data,
        error,
        trace: trace.to_string(),
        context_chunks: Vec::new(),
        hits: Vec::new(),
    };
    serde_json::to_value(v).unwrap_or_else(|_| {
        json!({
            "success": false,
            "data": serde_json::Value::Null,
            "error": "failed to serialize ToolOutputLike" ,
            "trace": "internal_error",
            "context_chunks": [],
            "hits": [],
        })
    })
}

pub fn search_tool_schema() -> toolkit::ToolSchema {
    toolkit::ToolSchema {
        name: "search_code".to_string(),
        description: "Search code (keyword placeholder backend), returns IndexChunk hits"
            .to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_hits": {"type": "number", "description": "Maximum hits (optional)"}
            }
        }),
        output_schema: json!({
            "type": "object",
            "properties": {
                "hits": {"type": "array"},
                "trace": {"type": "array"}
            }
        }),
    }
}

pub fn refill_tool_schema() -> toolkit::ToolSchema {
    toolkit::ToolSchema {
        name: "refill_hits".to_string(),
        description: "Refill IndexChunk hits into ContextChunks".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["hits"],
            "properties": {
                "hits": {"type": "array", "description": "IndexChunk array"}
            }
        }),
        output_schema: json!({
            "type": "object",
            "properties": {
                "context": {"type": "array"},
                "trace": {"type": "array"}
            }
        }),
    }
}
