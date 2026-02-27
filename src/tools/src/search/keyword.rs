/// Check if a symbol name is a common programming keyword
///
/// Currently supports Rust keywords. Extensible for multi-language support
/// by accepting an optional `lang_id` parameter.
///
/// TODO: Extend to support other languages:
///   - "py": Python keywords (def, None, True, False, list, dict, ...)
///   - "js"/"ts": JavaScript/TypeScript keywords (function, undefined, Array, ...)
///   - "go": Go keywords (func, interface, chan, defer, ...)
pub(crate) fn is_common_keyword(name: &str, _lang_id: Option<&str>) -> bool {
    // For now, default to Rust keywords since that's our primary use case
    // When lang_id is provided, we can dispatch to language-specific functions
    match _lang_id {
        Some("rs") | None => is_rust_keyword(name),
        // Future: Add other languages here
        // Some("py") => is_python_keyword(name),
        // Some("js") | Some("ts") => is_js_keyword(name),
        // Some("go") => is_go_keyword(name),
        _ => is_universal_keyword(name),
    }
}

/// Universal keywords applicable to most languages
fn is_universal_keyword(name: &str) -> bool {
    const UNIVERSAL: &[&str] = &[
        "if", "else", "for", "while", "return", "break", "continue",
        "true", "false", "null", "undefined",
    ];
    UNIVERSAL.contains(&name)
}

/// Rust-specific keywords and standard library types
fn is_rust_keyword(name: &str) -> bool {
    const RUST_KEYWORDS: &[&str] = &[
        // Control flow
        "if", "else", "for", "while", "loop", "match", "return", "break", "continue",
        // Variable declaration
        "let", "mut", "const", "static", "ref",
        // Functions and types
        "fn", "struct", "enum", "impl", "trait", "type", "where",
        // Visibility
        "pub", "crate", "super", "self", "Self",
        // Modules and imports
        "mod", "use", "extern",
        // Error handling
        "try", "await", "async", "move",
        // Lifetimes and generics
        "dyn", "impl Trait", "'_",
        // Special values
        "true", "false", "None", "Some", "Ok", "Err",
    ];

    const RUST_STD_TYPES: &[&str] = &[
        "String", "str", "Vec", "HashMap", "BTreeMap", "HashSet", "BTreeSet",
        "Box", "Arc", "Rc", "RefCell", "Cell", "Mutex", "RwLock",
        "Option", "Result", "PhantomData", "VecDeque", "LinkedList",
    ];

    RUST_KEYWORDS.contains(&name) || RUST_STD_TYPES.contains(&name)
}
