//! Refill Trigger: Dynamic context supplementation during execution
//!
//! This module detects when the LLM needs more context and triggers
//! the RefillPipeline to fetch additional information.
//!
//! ## Detection Patterns
//!
//! The trigger looks for patterns in LLM responses that indicate
//! missing context:
//! - "I don't see X..."
//! - "Missing implementation of X"
//! - "Need to see X to understand..."
//! - "The context doesn't show X..."
//!
//! ## Usage Flow
//!
//! ```text
//! Execute Plan Step
//!   │
//!   ▼
//! LLM Response: "I see handle_request but not authenticate_user"
//!   │
//!   ▼
//! RefillTrigger::detect_missing_symbols()
//!   │
//!   ▼
//! Trigger Refill for "authenticate_user"
//!   │
//!   ▼
//! Update Context with new chunks
//!   │
//!   ▼
//! Continue or Re-plan
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use context::{ContextChunk, ContextQuery, RefillPipeline, SymbolId};

/// Result of attempting to refill context
#[derive(Debug)]
pub enum RefillResult {
    /// No missing symbols detected, continue with current context
    NoActionNeeded,
    /// Successfully refilled with new context chunks
    Success { new_chunks: Vec<ContextChunk> },
    /// Failed to refill (symbol not found, etc.)
    Failed { reason: String },
}

/// Detects missing context from LLM responses and triggers Refill
pub struct RefillTrigger {
    pipeline: Arc<RefillPipeline>,
    /// Symbols that have already been refilled (to avoid loops)
    refilled_symbols: HashSet<String>,
    /// Maximum number of refill attempts per turn
    max_refills: usize,
    /// Current refill count
    refill_count: usize,
}

impl RefillTrigger {
    /// Create a new RefillTrigger
    pub fn new(pipeline: Arc<RefillPipeline>) -> Self {
        Self {
            pipeline,
            refilled_symbols: HashSet::new(),
            max_refills: 3,
            refill_count: 0,
        }
    }

    /// Check if we can still perform refills
    pub fn can_refill(&self) -> bool {
        self.refill_count < self.max_refills
    }

    /// Analyze LLM response and refill context if needed
    pub fn analyze_and_refill(&mut self, llm_response: &str) -> RefillResult {
        if !self.can_refill() {
            return RefillResult::Failed {
                reason: "Max refill attempts reached".to_string(),
            };
        }

        // Detect missing symbols from LLM response
        let missing_symbols = self.detect_missing_symbols(llm_response);

        if missing_symbols.is_empty() {
            return RefillResult::NoActionNeeded;
        }

        // Filter out already-refilled symbols
        let new_symbols: Vec<String> = missing_symbols
            .into_iter()
            .filter(|s| !self.refilled_symbols.contains(s))
            .collect();

        if new_symbols.is_empty() {
            return RefillResult::Failed {
                reason: "All detected symbols already refilled".to_string(),
            };
        }

        // Attempt to refill each symbol
        let mut new_chunks = Vec::new();
        for symbol in &new_symbols {
            match self.refill_symbol(symbol) {
                Ok(chunks) => {
                    new_chunks.extend(chunks);
                    self.refilled_symbols.insert(symbol.clone());
                }
                Err(e) => {
                    tracing::warn!("Failed to refill symbol {}: {}", symbol, e);
                }
            }
        }

        self.refill_count += 1;

        if new_chunks.is_empty() {
            RefillResult::Failed {
                reason: "Could not find any of the requested symbols".to_string(),
            }
        } else {
            RefillResult::Success { new_chunks }
        }
    }

    /// Detect symbols that LLM mentions as missing
    fn detect_missing_symbols(&self, response: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        let lower = response.to_lowercase();

        // Pattern 1: "don't see X" / "doesn't show X"
        // Example: "I don't see the implementation of authenticate_user"
        for pattern in &[
            "don't see",
            "doesn't show",
            "missing",
            "not shown",
            "need to see",
            "where is",
            "can't find",
        ] {
            if let Some(idx) = lower.find(pattern) {
                let after = &response[idx + pattern.len()..];
                // Try to extract a symbol name after the pattern
                if let Some(symbol) = extract_symbol_after_phrase(after) {
                    symbols.push(symbol);
                }
            }
        }

        // Pattern 2: "X is not defined" / "X is undefined"
        // Example: "authenticate_user is not defined"
        for pattern in &[" is not defined", " is undefined", " not found"] {
            if let Some(idx) = lower.find(pattern) {
                let before = &response[..idx];
                if let Some(symbol) = extract_last_identifier(before) {
                    symbols.push(symbol);
                }
            }
        }

        // Remove duplicates while preserving order
        let mut seen = HashSet::new();
        symbols.retain(|s| seen.insert(s.clone()));

        symbols
    }

    /// Refill context for a specific symbol
    fn refill_symbol(&self, symbol: &str) -> error::Result<Vec<ContextChunk>> {
        let query = ContextQuery::Symbol {
            name: symbol.to_string(),
        };

        let index_chunks = self.pipeline.retrieve(&query, 5)?;
        let context_chunks = self.pipeline.refine(&index_chunks);

        Ok(context_chunks)
    }

    /// Get symbols that have been refilled (for debugging)
    pub fn refilled_symbols(&self) -> &HashSet<String> {
        &self.refilled_symbols
    }

    /// Get current refill count
    pub fn refill_count(&self) -> usize {
        self.refill_count
    }
}

/// Extract a symbol name after a phrase like "don't see the"
fn extract_symbol_after_phrase(text: &str) -> Option<String> {
    // Skip common words
    let skip_words: HashSet<&str> = [
        "the", "any", "an", "a", "its", "this", "that", "of", "for", "in", "to",
        "implementation", "definition", "reference", "function", "method", "struct",
        "class", "module", "variable", "field", "property", "type",
    ]
    .iter()
    .cloned()
    .collect();

    let mut remaining = text.trim_start();

    // Keep skipping words until we find a potential symbol
    while !remaining.is_empty() {
        let lower = remaining.to_lowercase();

        // Find the first word in the remaining text
        let word_end = remaining
            .find(|c: char| !c.is_alphabetic())
            .unwrap_or(remaining.len());
        let first_word = &remaining[..word_end];

        if skip_words.contains(first_word.to_lowercase().as_str()) {
            remaining = remaining[word_end..].trim_start();
        } else {
            break;
        }
    }

    // Try to extract identifier
    extract_first_identifier(remaining)
}

/// Extract first identifier from text (snake_case or CamelCase)
fn extract_first_identifier(text: &str) -> Option<String> {
    let text = text.trim_start();

    // Find the start of an identifier
    let start = text.find(|c: char| c.is_alphabetic() || c == '_')?;
    let text = &text[start..];

    // Find the end of the identifier
    let end = text
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(text.len());

    let ident = &text[..end];

    // Filter out common words that are unlikely to be symbols
    let common_words = [
        "i", "the", "a", "an", "it", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when", "that",
        "which", "who", "whom", "whose", "what", "whatever", "whoever",
    ];

    if common_words.contains(&ident.to_lowercase().as_str()) {
        return None;
    }

    // Require at least 2 characters or start with uppercase (likely a type)
    if ident.len() >= 2 || ident.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        Some(ident.to_string())
    } else {
        None
    }
}

/// Extract last identifier from text, filtering out type descriptors
fn extract_last_identifier(text: &str) -> Option<String> {
    // Type descriptors that commonly appear before actual symbol names
    let type_words: HashSet<&str> = [
        "function", "fn", "method", "struct", "class", "enum", "trait",
        "impl", "module", "mod", "variable", "var", "let", "const",
        "static", "type", "field", "property", "argument", "arg",
        "parameter", "param", "the",
    ]
    .iter()
    .cloned()
    .collect();

    // Collect all valid identifiers with their positions
    let mut identifiers: Vec<(String, usize)> = Vec::new();
    let mut i = 0;

    while i < text.len() {
        if let Some(ident) = extract_first_identifier(&text[i..]) {
            let start_pos = i + text[i..].find(&ident).unwrap_or(0);
            identifiers.push((ident.clone(), start_pos));
            // Move past this identifier
            if let Some(idx) = text[i..].find(&ident) {
                i += idx + ident.len();
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }

    // Find the best candidate from the end
    // Prefer: snake_case > CamelCase > longer words > shorter words
    identifiers
        .iter()
        .rev()
        .find(|(ident, _)| !type_words.contains(ident.to_lowercase().as_str()))
        .map(|(ident, _)| ident.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_missing_symbol_dont_see() {
        let response = "I don't see the implementation of authenticate_user in the context.";
        let symbols = extract_missing_symbols_static(response);
        assert!(symbols.contains(&"authenticate_user".to_string()));
    }

    #[test]
    fn test_extract_missing_symbol_not_defined() {
        let response = "The function handle_request is not defined.";
        let symbols = extract_missing_symbols_static(response);
        assert!(symbols.contains(&"handle_request".to_string()));
    }

    #[test]
    fn test_extract_missing_symbol_cant_find() {
        let response = "I can't find the definition of UserService.";
        let symbols = extract_missing_symbols_static(response);
        assert!(symbols.contains(&"UserService".to_string()));
    }

    #[test]
    fn test_no_false_positives() {
        let response = "I see the implementation clearly. It looks good to me.";
        let symbols = extract_missing_symbols_static(response);
        assert!(symbols.is_empty());
    }

    fn extract_missing_symbols_static(response: &str) -> Vec<String> {
        // Simplified version for testing - mirrors detect_missing_symbols logic
        let mut symbols = Vec::new();
        let lower = response.to_lowercase();

        // Pattern 1: "don't see X" / "can't find X" - extract from after
        for pattern in &["don't see", "can't find"] {
            if let Some(idx) = lower.find(pattern) {
                let after = &response[idx + pattern.len()..];
                if let Some(symbol) = extract_symbol_after_phrase(after) {
                    symbols.push(symbol);
                }
            }
        }

        // Pattern 2: "X is not defined" - extract from before
        for pattern in &[" is not defined", " is undefined", " not found"] {
            if let Some(idx) = lower.find(pattern) {
                let before = &response[..idx];
                if let Some(symbol) = extract_last_identifier(before) {
                    symbols.push(symbol);
                }
            }
        }

        symbols
    }
}
