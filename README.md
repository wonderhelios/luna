# Luna 🌙

> **Local-first Agentic IDE Runtime - Minimal TPAR Architecture**

Luna is a minimal, OpenCode-inspired AI code assistant. It keeps only what's essential.

## Current State

This is a clean slate. The only remaining component is:

- **`src/intelligence/`** - ScopeGraph-based code intelligence

Everything else has been removed to rebuild from scratch.

## Architecture Philosophy

Inspired by [OpenCode](https://github.com/anomalyco/opencode):

1. **No complex intent classification** - Just give input to LLM
2. **No pre-planned steps** - LLM decides tools on the fly
3. **No complex orchestration** - Simple sequential execution
4. **Minimal abstraction** - Less code, fewer bugs

## Rebuilding

The next iteration will be a single-file TPAR implementation:

```rust
// Pseudo-code
loop {
    let input = get_user_input();
    let tools = llm.plan(input);           // What tools?
    let results = execute_tools(tools);    // Execute
    let response = llm.synthesize(results);// Respond
    println!(response);
}
```

## License

MIT
