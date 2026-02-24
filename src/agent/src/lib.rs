mod context_engine;
mod intel_adapter;
mod llm;
mod react;
mod tools;
mod types;

pub use context_engine::render_prompt_context;
pub use intel_adapter::{ParsedFile, SymbolKind};
pub use llm::llm_answer;
pub use react::react_ask;
pub use tools::{
    build_context_pack_keyword, detect_lang_id, edit_file, list_dir, list_symbols_enhanced,
    list_symbols_by_kind, list_public_functions,
    read_file, refill_hits, run_terminal, search_code_keyword, SearchCodeOptions,
    SymbolFilter, SymbolListOptions, SymbolSortOrder,
    SymbolDetail, SymbolVisibility,
};
pub use types::{
    ContextEngineOptions, ContextPack, EditOp, EditResult, LLMConfig, ReActAction, ReActOptions,
    ReActStepTrace, TerminalResult, ToolName, ToolTrace,
};
