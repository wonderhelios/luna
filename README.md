# Luna 🌙

> **An Agentic IDE Companion with Symbol-Level Code Intelligence**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-M3%20Junior%20Coder-yellow.svg)]()

Luna reads code like a senior engineer—understanding symbols, scopes, and relationships through Tree-sitter AST analysis—then reasons, searches, and edits with contextual awareness via the ReAct agent loop.

---

## Key Features

- **Symbol Navigation** — Go-to-definition, automatic symbol resolution, cross-file reference finding
- **Deterministic Analysis** — Tree-sitter AST + Scope Graph for precise symbol understanding, not LLM guessing
- **Smart Context Pipeline** — IndexChunk for retrieval, ContextChunk for LLM, refined via Refill with auto symbol injection
- **Enforced Safety** — Hard constraints on dangerous commands and duplicate edits
- **Language Agnostic** — One core algorithm, 12+ languages via Tree-sitter grammars
- **Native Performance** — Rust-powered, no GIL or event loop bottlenecks

---

## Architecture

Luna is built on three pillars:

```
┌─────────────────────────────────────┐
│         🧠 ReAct Agent Loop          │
│    Think → Act → Observe → Repeat    │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐ ┌────────┐ ┌──────────┐
│Search │ │ Files  │ │ Terminal │
└───┬───┘ └────────┘ └──────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       🔬 Intelligence Engine         │
│  Tree-sitter → AST → Scope Graph    │
│  (12+ languages supported)          │
└─────────────────────────────────────┘
```

1. **Intelligence Engine** parses code into AST and builds scope graphs for semantic understanding
2. **Agent Loop** reasons about the task, plans tool invocations, and refines based on observations
3. **Tool Layer** executes file operations, searches, and terminal commands safely

---

## Quick Start

```bash
git clone https://github.com/yourusername/luna.git
cd luna
cargo build --release

# Ask about your codebase
./target/release/luna --repo ./my-project \
  "How does the connection pool handle timeouts?"
```

### As a Library

```rust
use react::LunaRuntime;

let runtime = LunaRuntime::new(tokenizer, llm_config, policy, options);
let (answer, context, traces) = runtime.ask_react(
    repo_root,
    "Find potential race conditions"
)?;
```

---

## Project Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1: Hello Agent** | ✅ Done | ReAct loop, core tools, Context Engine |
| **M2: Smart Reader** | ✅ Done | Symbol analysis, scope resolution, go-to-definition |
| **M3: Junior Coder** | 🚧 Active | Auto-fix compile errors, test-driven repair |
| **M4: IDE Integration** | 📅 Planned | MCP Server, VSCode extension |
| **M5: Senior Mode** | 📅 Planned | Vector search, Repo Map Ranking |

---

## Development

```bash
# Run tests
cargo test -p react      # 20 integration tests
cargo test -p tools      # Tool operations
cargo test -p intelligence  # AST/Scope analysis
```

---

## License

MIT © [ **wonder** ]

<p align="center">
  Built with 🦀 Rust
</p>
