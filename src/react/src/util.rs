//! Utility functions for the react crate

use std::str::FromStr;
use tokenizers::Tokenizer;

/// Create a placeholder tokenizer for default configurations
///
/// Note: This is a minimal tokenizer for use in tests and default configs.
/// Production code should use a properly initialized tokenizer.
pub fn demo_tokenizer() -> Tokenizer {
    // Minimal BPE tokenizer config
    let tokenizer_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3},
            "merges": []
        }
    }"#;

    Tokenizer::from_str(tokenizer_json).expect("Failed to create placeholder tokenizer")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_tokenizer() {
        let tokenizer = demo_tokenizer();
        // Just verify it doesn't panic and can encode
        let encoding = tokenizer.encode("abc", false).expect("Failed to encode");
        assert!(!encoding.get_ids().is_empty());
    }
}
