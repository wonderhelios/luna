use ahash::AHashMap;
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};
use toolkit::ExecutionPolicy;

pub fn demo_tokenizer() -> Tokenizer {
    let mut vocab = AHashMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("fn".to_string(), 1u32);
    vocab.insert("let".to_string(), 2u32);
    vocab.insert("return".to_string(), 3u32);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("demo tokenizer model build must succeed");
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct PolicyPatch {
    pub allow_edit_file: Option<bool>,
    pub require_confirm_edit_file: Option<bool>,
    pub allow_run_terminal: Option<bool>,
    pub require_confirm_run_terminal: Option<bool>,
}

pub fn parse_policy_overrides(params: &serde_json::Value) -> Option<PolicyPatch> {
    let obj = params.get("policy")?;
    serde_json::from_value::<PolicyPatch>(obj.clone()).ok()
}

pub fn apply_policy_patch(mut base: ExecutionPolicy, patch: PolicyPatch) -> ExecutionPolicy {
    if let Some(v) = patch.allow_edit_file {
        base.allow_edit_file = v;
    }
    if let Some(v) = patch.require_confirm_edit_file {
        base.require_confirm_edit_file = v;
    }
    if let Some(v) = patch.allow_run_terminal {
        base.allow_run_terminal = v;
    }
    if let Some(v) = patch.require_confirm_run_terminal {
        base.require_confirm_run_terminal = v;
    }
    base
}

pub fn session_id_from_params(params: &serde_json::Value) -> Option<String> {
    params
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

pub fn repo_root_from_opt(repo_root: Option<String>) -> PathBuf {
    repo_root
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}
