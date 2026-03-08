# Luna 🌙

> **Local-first Agentic IDE Runtime with TPAR Architecture**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%203%20Complete-brightgreen.svg)]()

Luna is a local-first AI code assistant that plans before acting. Built on the **TPAR** (Task → Plan → Act → Review) architecture, it combines deterministic code intelligence with LLM-powered reasoning for safe, explainable, and efficient coding assistance.

---

## Key Features

- **TPAR Architecture** — Plan before acting, global optimization with Review reflection
- **Symbol Navigation** — Go-to-definition, cross-file reference finding via ScopeGraph
- **Deterministic Analysis** — Tree-sitter AST + Scope Graph for precise understanding
- **Context Pipeline** — IndexChunk → Refill → ContextChunk for efficient retrieval (Phase 4)
- **Safety Guard** — Dangerous command interception, duplicate edit detection
- **Session Persistence** — JSONL-based session store, multi-session management
- **Streaming UI** — Real-time event streaming in TUI
- **Language Agnostic** — 12+ languages via Tree-sitter grammars
- **Native Performance** — Rust-powered, no GIL or event loop bottlenecks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TPAR Agent Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │  Task   │───►│  Plan   │───►│   Act   │───► Review  │     │
│   │ (Intent)│    │(Planner)│    │ (Tools) │    │(Reflect)│     │
│   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
│        │              │              │              │          │
│        ▼              ▼              ▼              ▼          │
│   ┌────────────────────────────────────────────────────────┐  │
│   │           Trajectory Recorder + EventStream            │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                 │
│   SafetyGuard │ Context Pipeline │ Session Store │ Tool Registry│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Task**: Classify user intent (Query, Edit, Explain, Terminal, Chat)
2. **Plan**: Generate executable step sequence (Rule-based or LLM-based)
3. **Act**: Execute tools with safety checks (read_file, edit_file, run_terminal)
4. **Review**: Validate results, rollback on failure, reflect for learning

---

## Quick Start

### Installation

```bash
git clone https://github.com/wonderhelios/luna.git
cd luna
cargo build --release

# Run Luna
cargo run --release
```

### Configuration (Optional)

Configure LLM for intelligent planning:

```bash
# OpenAI
export LUNA_LLM_API_KEY="sk-..."
export LUNA_LLM_MODEL="gpt-4o-mini"

# Or OpenRouter
export LUNA_LLM_BASE_URL="https://openrouter.ai/api/v1"
export LUNA_LLM_MODEL="anthropic/claude-3.5-sonnet"

# Enable LLM-based planner
export LUNA_PLANNER="llm"
```

### Usage

```bash
# Interactive TUI
cargo run --release

# Session commands
> /sessions                    # List active sessions
> /switch <session_id>         # Switch session
> /session                     # Alias for /sessions

# Code assistance
> Where is the main function?  # Symbol navigation
> Explain handle_request       # Code explanation
> 修改 src/main.rs 第 10 行为 // TODO
```

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0: TPAR Skeleton** | ✅ Done | Task→Plan→Act→Review main loop |
| **Phase 1: Intelligence** | ✅ Done | ScopeGraph, 12+ languages, <10ms symbol query |
| **Phase 2: Session + Safety** | ✅ Done | JSONL persistence, SafetyGuard, Trajectory |
| **Phase 3A: Streaming** | ✅ Done | EventStream, TUI real-time rendering |
| **Phase 3B: LLM Planner** | ✅ Done | OpenAIClient, LlmBasedPlanner with JSON Plan |
| **Phase 4: Context Pipeline** | 📅 Planned | IndexChunk → Refill → ContextChunk |
| **Phase 5: Project Memory** | 📅 Planned | Auto-learned configs, LUNA.md support |
| **Phase 6: RL Plan** | 📅 Planned | Reinforcement learning for planning |

### Current Capabilities

- ✅ Multi-language symbol navigation (Rust, Python, Go, TypeScript, etc.)
- ✅ File reading and line-level editing
- ✅ Terminal command execution with safety guards
- ✅ Session persistence and management
- ✅ Real-time event streaming in TUI
- ✅ LLM-powered plan generation (with API key)

---

## Architecture Details

### TPAR vs ReAct

| Aspect | ReAct | TPAR (Luna) |
|--------|-------|-------------|
| Flow | Thought→Action→Observation | Task→Plan→Act→Review |
| Planning | Step-by-step | Global plan first |
| Recovery | Immediate | Review + rollback |
| Learning | Hard | Plan/Review as RL signals |

### Code Intelligence

```rust
// ScopeGraph-based precise symbol resolution
let symbol = scope_graph.goto_definition("handle_request");
// → src/handler.rs:45 (not text search guess)

let callers = scope_graph.find_callers("authenticate");
// → 5 precise call sites across files
```

### Safety Mechanisms

- **Dangerous Commands**: `rm -rf /`, `mkfs`, `dd`, etc. are blocked
- **Duplicate Edits**: Warning when editing same location twice
- **Token Budget**: Input/output/step limits enforced
- **Rollback**: Original content preserved for edits

---

## Development

```bash
# Run all tests
cargo test

# Run specific package tests
cargo test --package runtime      # TPAR loop, planner, safety
cargo test --package intelligence # ScopeGraph, navigation
cargo test --package tools        # File/terminal operations
cargo test --package session      # JSONL store
cargo test --package llm          # LLM clients

# Run clippy
cargo clippy

# Build release
cargo build --release
```

### Project Structure

```
src/
├── cli/           # Interactive TUI
├── runtime/       # TPAR loop, planner, safety, recorder
├── intelligence/  # ScopeGraph, language support
├── tools/         # Tool registry (read_file, edit_file, run_terminal)
├── session/       # Session persistence (JSONL)
├── llm/           # LLM clients (OpenAI, OpenRouter, etc.)
├── core/          # Shared types (TextRange, SymbolId)
└── error/         # Error handling
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LUNA_LLM_API_KEY` | LLM API key | *(none)* |
| `LUNA_LLM_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `LUNA_LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LUNA_LLM_TIMEOUT_SECS` | Request timeout | `60` |
| `LUNA_PLANNER` | Planner type (`rule`/`llm`) | `rule` |

### Supported LLM Providers

- **OpenAI**: `https://api.openai.com/v1`
- **OpenRouter**: `https://openrouter.ai/api/v1`
- **SiliconFlow**: `https://api.siliconflow.cn/v1`
- **Any OpenAI-compatible API**

---

## License

MIT © [wonder](https://github.com/wonderhelios)

<p align="center">
  Built with 🦀 Rust
</p>
