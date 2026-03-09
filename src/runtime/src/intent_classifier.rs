//! Intent Classification - Model-driven user intent understanding
//!
//! This module provides a pluggable architecture for intent classification,
//! supporting both rule-based and LLM-based approaches.
//!
//! ## Architecture
//!
//! ```text
//! User Input
//!   ↓
//! IntentClassifier::classify()
//!   ↓
//! ClassificationResult { intent, confidence, entities }
//!   ↓
//! TaskAnalyzer → Task → TPAR Pipeline
//! ```
//!
//! ## Implementations
//!
//! - `RuleBasedClassifier`: Fast, deterministic, no external calls
//! - `LLMClassifier`: Accurate, handles natural language variations
//! - `HybridClassifier`: Rule-based fast path + LLM fallback
//!
//! ## Future: Local Model Support
//!
//! To add a local model (e.g., Ollama, llama.cpp):
//!
//! ```rust,ignore
//! pub struct LocalModelClassifier {
//!     model: Box<dyn LocalInferenceEngine>,
//! }
//!
//! impl IntentClassifier for LocalModelClassifier {
//!     fn classify(&self, input: &str, ctx: &Context) -> ClassificationResult {
//!         // Call local model via llama.cpp or similar
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::intent;

/// Classification confidence level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Confidence {
    /// Very confident (>90%)
    High,
    /// Moderately confident (70-90%)
    Medium,
    /// Low confidence (<70%), consider fallback
    Low,
    /// Extremely uncertain, definitely needs fallback
    Uncertain,
}

impl Confidence {
    /// Check if confidence is sufficient for direct execution
    pub fn is_sufficient(&self) -> bool {
        matches!(self, Confidence::High | Confidence::Medium)
    }
}

/// Rich intent representation with structured entities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
    /// Navigate to symbol definition or references
    SymbolNavigation {
        /// The symbol to navigate to
        symbol: String,
        /// Navigation target type
        target: NavigationTarget,
    },
    /// Explain what a symbol does
    ExplainSymbol {
        /// The symbol to explain
        symbol: String,
        /// Specific aspect to explain (optional)
        aspect: Option<String>,
    },
    /// Edit code at a specific location
    Edit {
        /// Target file path (may be partial/inferred)
        path: Option<String>,
        /// Line number if specified
        line: Option<usize>,
        /// The symbol to edit (if path not specified)
        symbol: Option<String>,
        /// Description of the desired change
        description: String,
    },
    /// Run a terminal command
    Terminal {
        /// The command to execute
        command: String,
        /// Whether this is a safe read-only command
        is_safe: bool,
    },
    /// Search for code matching criteria
    Search {
        /// Search query (can be natural language or regex-like)
        query: String,
        /// Search scope
        scope: SearchScope,
    },
    /// General chat/conversation
    Chat,
    /// Request for help/documentation
    Help,
    /// Unclear intent - needs clarification
    ClarificationNeeded {
        /// What we understood
        partial_understanding: String,
        /// What we need to know
        missing_info: String,
    },
}

/// Navigation target type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NavigationTarget {
    /// Go to definition
    Definition,
    /// Find all references
    References,
    /// Go to implementation
    Implementation,
    /// Go to type definition
    TypeDefinition,
}

/// Search scope
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SearchScope {
    /// Current file only
    CurrentFile,
    /// Current module/directory
    CurrentModule,
    /// Entire project (default)
    #[default]
    Project,
    /// Specific path pattern
    PathPattern(String),
}

/// Extracted entity from user input
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Entity kind
    pub kind: EntityKind,
    /// Raw text value
    pub value: String,
    /// Normalized/canonical value
    pub normalized: Option<String>,
    /// Confidence this is a real entity (0.0-1.0)
    pub confidence: f32,
}

/// Entity kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    /// Code identifier (function, struct, variable name)
    Identifier,
    /// File path
    Path,
    /// Line number
    Line,
    /// Column number
    Column,
    /// Code snippet/pattern
    CodePattern,
    /// Natural language description
    Description,
    /// Command string
    Command,
}

/// Result of intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The classified intent
    pub intent: Intent,
    /// Confidence level
    pub confidence: Confidence,
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Raw user input (for debugging)
    pub raw_input: String,
    /// Which classifier produced this result
    pub classifier: &'static str,
    /// Optional metadata (classifier-specific)
    pub metadata: HashMap<String, String>,
}

impl ClassificationResult {
    /// Create a new classification result
    pub fn new(
        intent: Intent,
        confidence: Confidence,
        entities: Vec<Entity>,
        raw_input: &str,
        classifier: &'static str,
    ) -> Self {
        Self {
            intent,
            confidence,
            entities,
            raw_input: raw_input.to_owned(),
            classifier,
            metadata: HashMap::new(),
        }
    }

    /// Check if this result has sufficient confidence
    pub fn is_confident(&self) -> bool {
        self.confidence.is_sufficient()
    }

    /// Get entity by kind
    pub fn get_entity(&self, kind: EntityKind) -> Option<&Entity> {
        self.entities.iter().find(|e| e.kind == kind)
    }

    /// Get all entities of a specific kind
    pub fn get_entities(&self, kind: EntityKind) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.kind == kind).collect()
    }
}

/// Context for intent classification
///
/// Provides additional context that may help classification,
/// such as current file, project structure, conversation history, etc.
#[derive(Debug, Clone, Default)]
pub struct ClassificationContext {
    /// Current working directory
    pub cwd: Option<std::path::PathBuf>,
    /// Current open file (if any)
    pub current_file: Option<std::path::PathBuf>,
    /// Recent conversation history (last N turns)
    pub conversation_history: Vec<String>,
    /// Available commands/tools in the system
    pub available_tools: Vec<String>,
    /// Project language (inferred from files)
    pub project_language: Option<String>,
}

impl ClassificationContext {
    /// Create a new context with just the working directory
    pub fn with_cwd(cwd: std::path::PathBuf) -> Self {
        Self {
            cwd: Some(cwd),
            ..Default::default()
        }
    }

    /// Add conversation history
    pub fn with_history(mut self, history: Vec<String>) -> Self {
        self.conversation_history = history;
        self
    }
}

/// Trait for intent classification implementations
///
/// This is the core abstraction that allows swapping between
/// rule-based, LLM-based, and local model-based classifiers.
pub trait IntentClassifier: Send + Sync {
    /// Classify user input into an intent
    ///
    /// # Arguments
    /// * `input` - The user's raw input
    /// * `ctx` - Context that may help classification
    ///
    /// # Returns
    /// Classification result with intent, confidence, and entities
    fn classify(&self, input: &str, ctx: &ClassificationContext) -> ClassificationResult;

    /// Get the classifier name (for debugging/metrics)
    fn name(&self) -> &'static str;

    /// Check if this classifier requires external resources (API calls, etc.)
    fn requires_external(&self) -> bool {
        false
    }
}

// ============================================================================
// Rule-Based Implementation (current logic, extracted and improved)
// ============================================================================

/// Rule-based intent classifier
///
/// Fast, deterministic, no external dependencies.
/// Good for obvious patterns, serves as fallback for LLM classifier.
pub struct RuleBasedClassifier;

impl RuleBasedClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Extract entities using rule-based heuristics
    fn extract_entities(input: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Extract identifiers
        for ident in intent::extract_identifiers_dedup(input) {
            entities.push(Entity {
                kind: EntityKind::Identifier,
                value: ident.to_owned(),
                normalized: Some(ident.to_owned()),
                confidence: 0.8,
            });
        }

        // Extract file positions
        if let Some((path, line, col)) = intent::extract_file_position(input) {
            entities.push(Entity {
                kind: EntityKind::Path,
                value: path.display().to_string(),
                normalized: Some(path.display().to_string()),
                confidence: 0.9,
            });
            entities.push(Entity {
                kind: EntityKind::Line,
                value: line.to_string(),
                normalized: Some(line.to_string()),
                confidence: 0.9,
            });
            if col > 0 {
                entities.push(Entity {
                    kind: EntityKind::Column,
                    value: col.to_string(),
                    normalized: Some(col.to_string()),
                    confidence: 0.9,
                });
            }
        }

        entities
    }

    /// Try to parse edit intent
    fn try_parse_edit(input: &str) -> Option<Intent> {
        use crate::tpar::{parse_edit_intent_min, parse_edit_line_to};

        // Try full edit spec: "edit file.rs:10 to new_content"
        if let Some((path, line, description)) = parse_edit_line_to(input) {
            return Some(Intent::Edit {
                path: Some(path),
                line: Some(line),
                symbol: None,
                description,
            });
        }

        // Try minimal edit: "edit file.rs:10"
        if let Some((path, line)) = parse_edit_intent_min(input) {
            return Some(Intent::Edit {
                path: Some(path),
                line: Some(line),
                symbol: None,
                description: String::new(),
            });
        }

        None
    }

    /// Try to parse terminal intent
    fn try_parse_terminal(input: &str) -> Option<Intent> {
        use crate::tpar::parse_terminal_intent;

        parse_terminal_intent(input).map(|cmd| Intent::Terminal {
            command: cmd.clone(),
            is_safe: is_safe_command(&cmd),
        })
    }
}

impl IntentClassifier for RuleBasedClassifier {
    fn classify(&self, input: &str, _ctx: &ClassificationContext) -> ClassificationResult {
        let entities = Self::extract_entities(input);

        // Check for edit intent first (most specific)
        if let Some(intent) = Self::try_parse_edit(input) {
            return ClassificationResult::new(
                intent,
                Confidence::High,
                entities,
                input,
                self.name(),
            );
        }

        // Check for terminal intent
        if let Some(intent) = Self::try_parse_terminal(input) {
            return ClassificationResult::new(
                intent,
                Confidence::High,
                entities,
                input,
                self.name(),
            );
        }

        // Use legacy intent classification for navigation/explain
        let legacy_intent = intent::classify_intent(input);
        let (intent, confidence) = match legacy_intent {
            intent::Intent::SymbolNavigation => {
                let symbol = intent::extract_best_identifier(input)
                    .unwrap_or("")
                    .to_owned();
                (
                    Intent::SymbolNavigation {
                        symbol,
                        target: NavigationTarget::Definition,
                    },
                    Confidence::Medium,
                )
            }
            intent::Intent::ExplainSymbol => {
                let symbol = intent::extract_best_identifier(input)
                    .unwrap_or("")
                    .to_owned();
                (
                    Intent::ExplainSymbol {
                        symbol,
                        aspect: None,
                    },
                    Confidence::Medium,
                )
            }
            intent::Intent::Other => (Intent::Chat, Confidence::Low),
        };

        ClassificationResult::new(intent, confidence, entities, input, self.name())
    }

    fn name(&self) -> &'static str {
        "rule_based"
    }
}

/// Check if a command is safe (read-only)
fn is_safe_command(cmd: &str) -> bool {
    let safe_prefixes = ["ls", "cat", "find", "grep", "head", "tail", "echo", "pwd", "git status", "git log", "git diff", "git show"];
    let lower = cmd.to_ascii_lowercase();
    safe_prefixes.iter().any(|prefix| lower.starts_with(prefix))
}

// ============================================================================
// LLM-Based Implementation
// ============================================================================

/// LLM-based intent classifier
///
/// Uses an LLM to classify intent. More flexible than rule-based,
/// handles natural language variations well.
///
/// Can be configured with different models (OpenAI, Anthropic, local, etc.)
pub struct LLMClassifier {
    client: Arc<dyn llm::LLMClient>,
    /// Temperature for generation (lower = more deterministic)
    temperature: f32,
    /// Maximum tokens for response
    max_tokens: usize,
}

impl LLMClassifier {
    pub fn new(client: Arc<dyn llm::LLMClient>) -> Self {
        Self {
            client,
            temperature: 0.1, // Low temperature for consistent classification
            max_tokens: 500,
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Build the classification prompt
    fn build_prompt(input: &str, ctx: &ClassificationContext) -> String {
        let tools_list = if ctx.available_tools.is_empty() {
            "- read_file: Read file contents\n\
             - edit_file: Edit a file\n\
             - run_terminal: Execute terminal commands\n\
             - search_code: Search for code patterns"
                .to_owned()
        } else {
            ctx.available_tools.join("\n")
        };

        let history_context = if ctx.conversation_history.is_empty() {
            String::new()
        } else {
            format!(
                "\nRecent conversation:\n{}\n",
                ctx.conversation_history.join("\n")
            )
        };

        format!(
            "You are an intent classification system for a code assistant.\n\
             Analyze the user input and classify it into one of the intent categories.\n\n\
             Available intents:\n\
             - symbol_navigation: Find where a symbol is defined/referenced\n\
             - explain_symbol: Explain what a symbol does\n\
             - edit: Edit code at a specific location\n\
             - terminal: Run a terminal command\n\
             - search: Search for code matching criteria\n\
             - chat: General conversation\n\
             - help: Request for help/documentation\n\
             - clarification_needed: Unclear what user wants\n\n\
             Available tools:\n{}\n{}\n\
             Classify this input: \"{}\"\n\n\
             Respond in JSON format:\n\
             {{\n\
               \"intent\": \"<intent_name>\",\n\
               \"confidence\": \"<high|medium|low>\",\n\
             \"entities\": [\n\
               {{\n\
               \"kind\": \"<identifier|path|line|description|command>\",\n\
               \"value\": \"<extracted_value>\"\n\
               }}\n\
             ],\n\
               \"details\": {{\n\
                 // intent-specific details\n\
                 \"symbol\": \"<for_symbol_intents>\",\n\
                 \"target\": \"<definition|references|implementation>\",\n\
                 \"path\": \"<file_path_if_any>\",\n\
                 \"line\": <line_number_if_any>,\n\
                 \"description\": \"<change_description_for_edit>\"\n\
               }}\n\
             }}",
            tools_list, history_context, input
        )
    }

    /// Parse LLM response into ClassificationResult
    fn parse_response(&self, response: &str, input: &str) -> ClassificationResult {
        // Try to extract JSON from markdown code blocks or raw JSON
        let json_str = Self::extract_json(response);

        match serde_json::from_str::<serde_json::Value>(&json_str) {
            Ok(value) => self.parse_classification_json(&value, input),
            Err(e) => {
                tracing::warn!("Failed to parse LLM classification response: {}", e);
                // Return low-confidence chat as fallback
                ClassificationResult::new(
                    Intent::Chat,
                    Confidence::Uncertain,
                    Vec::new(),
                    input,
                    self.name(),
                )
            }
        }
    }

    /// Extract JSON from response (handles markdown code blocks)
    fn extract_json(response: &str) -> String {
        let trimmed = response.trim();

        // Try to find JSON in markdown code blocks
        if let Some(start) = trimmed.find("```json") {
            if let Some(end) = trimmed[start + 7..].find("```") {
                return trimmed[start + 7..start + 7 + end].trim().to_owned();
            }
        }

        // Try plain code blocks
        if let Some(start) = trimmed.find("```") {
            if let Some(end) = trimmed[start + 3..].find("```") {
                let content = trimmed[start + 3..start + 3 + end].trim();
                // Skip language identifier
                let json_start = content.find('{').unwrap_or(0);
                return content[json_start..].to_owned();
            }
        }

        // Try to find JSON object directly
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                return trimmed[start..=end].to_owned();
            }
        }

        trimmed.to_owned()
    }

    /// Parse classification JSON into result
    fn parse_classification_json(
        &self,
        value: &serde_json::Value,
        input: &str,
    ) -> ClassificationResult {
        let intent_str = value
            .get("intent")
            .and_then(|v| v.as_str())
            .unwrap_or("chat");

        let confidence_str = value
            .get("confidence")
            .and_then(|v| v.as_str())
            .unwrap_or("low");

        let confidence = match confidence_str {
            "high" => Confidence::High,
            "medium" => Confidence::Medium,
            "low" => Confidence::Low,
            _ => Confidence::Uncertain,
        };

        // Parse entities
        let mut entities = Vec::new();
        if let Some(entities_val) = value.get("entities").and_then(|v| v.as_array()) {
            for e in entities_val {
                if let (Some(kind_str), Some(val)) =
                    (e.get("kind").and_then(|v| v.as_str()), e.get("value").and_then(|v| v.as_str()))
                {
                    let kind = match kind_str {
                        "identifier" => EntityKind::Identifier,
                        "path" | "file" => EntityKind::Path,
                        "line" => EntityKind::Line,
                        "column" => EntityKind::Column,
                        "code" | "pattern" => EntityKind::CodePattern,
                        "description" => EntityKind::Description,
                        "command" => EntityKind::Command,
                        _ => EntityKind::Description,
                    };
                    entities.push(Entity {
                        kind,
                        value: val.to_owned(),
                        normalized: None,
                        confidence: 0.9,
                    });
                }
            }
        }

        // Parse details
        let details = value.get("details");

        let intent = match intent_str {
            "symbol_navigation" => {
                let symbol = details
                    .and_then(|d| d.get("symbol"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .or_else(|| entities.iter().find(|e| e.kind == EntityKind::Identifier).map(|e| e.value.clone()))
                    .unwrap_or_default();

                let target = details
                    .and_then(|d| d.get("target"))
                    .and_then(|v| v.as_str())
                    .map(|t| match t {
                        "references" => NavigationTarget::References,
                        "implementation" => NavigationTarget::Implementation,
                        "type" | "type_definition" => NavigationTarget::TypeDefinition,
                        _ => NavigationTarget::Definition,
                    })
                    .unwrap_or(NavigationTarget::Definition);

                Intent::SymbolNavigation { symbol, target }
            }
            "explain_symbol" => {
                let symbol = details
                    .and_then(|d| d.get("symbol"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .or_else(|| entities.iter().find(|e| e.kind == EntityKind::Identifier).map(|e| e.value.clone()))
                    .unwrap_or_default();

                let aspect = details
                    .and_then(|d| d.get("aspect"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());

                Intent::ExplainSymbol { symbol, aspect }
            }
            "edit" => {
                let path = details
                    .and_then(|d| d.get("path"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .or_else(|| entities.iter().find(|e| e.kind == EntityKind::Path).map(|e| e.value.clone()));

                let line = details
                    .and_then(|d| d.get("line"))
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .or_else(|| entities.iter().find(|e| e.kind == EntityKind::Line).and_then(|e| e.value.parse().ok()));

                let symbol = entities
                    .iter()
                    .find(|e| e.kind == EntityKind::Identifier)
                    .map(|e| e.value.clone());

                let description = details
                    .and_then(|d| d.get("description"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_default();

                Intent::Edit {
                    path,
                    line,
                    symbol,
                    description,
                }
            }
            "terminal" => {
                let command = details
                    .and_then(|d| d.get("command"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .or_else(|| entities.iter().find(|e| e.kind == EntityKind::Command).map(|e| e.value.clone()))
                    .unwrap_or_default();

                Intent::Terminal {
                    is_safe: is_safe_command(&command),
                    command,
                }
            }
            "search" => {
                let query = details
                    .and_then(|d| d.get("query"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| input.to_owned());

                let scope = details
                    .and_then(|d| d.get("scope"))
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "file" => SearchScope::CurrentFile,
                        "module" => SearchScope::CurrentModule,
                        "project" => SearchScope::Project,
                        _ => SearchScope::Project,
                    })
                    .unwrap_or_default();

                Intent::Search { query, scope }
            }
            "help" => Intent::Help,
            "clarification_needed" => {
                let partial = details
                    .and_then(|d| d.get("partial_understanding"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_default();

                let missing = details
                    .and_then(|d| d.get("missing_info"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| "Could you please clarify what you'd like me to do?".to_owned());

                Intent::ClarificationNeeded {
                    partial_understanding: partial,
                    missing_info: missing,
                }
            }
            _ => Intent::Chat,
        };

        let mut result =
            ClassificationResult::new(intent, confidence, entities, input, self.name());

        // Add raw response to metadata for debugging
        if let Ok(json_str) = serde_json::to_string(value) {
            result
                .metadata
                .insert("raw_response".to_owned(), json_str);
        }

        result
    }
}

impl IntentClassifier for LLMClassifier {
    fn classify(&self, input: &str, ctx: &ClassificationContext) -> ClassificationResult {
        let prompt = Self::build_prompt(input, ctx);

        let req = llm::CompletionRequest {
            prompt,
        };

        match self.client.complete(req) {
            Ok(resp) => self.parse_response(&resp.content, input),
            Err(e) => {
                tracing::error!("LLM classification failed: {}", e);
                // Fall back to uncertain chat
                ClassificationResult::new(
                    Intent::Chat,
                    Confidence::Uncertain,
                    Vec::new(),
                    input,
                    self.name(),
                )
            }
        }
    }

    fn name(&self) -> &'static str {
        "llm"
    }

    fn requires_external(&self) -> bool {
        true
    }
}

// ============================================================================
// Hybrid Implementation - Rule-based fast path + LLM fallback
// ============================================================================

/// Hybrid classifier that combines rule-based and LLM approaches
///
/// Strategy:
/// 1. Try rule-based first (fast, no external calls)
/// 2. If rule-based returns low confidence, use LLM
/// 3. Cache LLM results to avoid repeated calls for similar inputs
pub struct HybridClassifier {
    rule: RuleBasedClassifier,
    llm: LLMClassifier,
    /// Minimum confidence threshold for rule-based acceptance
    rule_threshold: Confidence,
}

impl HybridClassifier {
    pub fn new(llm_client: Arc<dyn llm::LLMClient>) -> Self {
        Self {
            rule: RuleBasedClassifier::new(),
            llm: LLMClassifier::new(llm_client),
            rule_threshold: Confidence::Medium,
        }
    }

    /// Set the confidence threshold for rule-based acceptance
    pub fn with_threshold(mut self, threshold: Confidence) -> Self {
        self.rule_threshold = threshold;
        self
    }
}

impl IntentClassifier for HybridClassifier {
    fn classify(&self, input: &str, ctx: &ClassificationContext) -> ClassificationResult {
        // First try rule-based
        let rule_result = self.rule.classify(input, ctx);

        // If rule-based is confident enough, use it
        if rule_result.confidence >= self.rule_threshold {
            tracing::debug!("Using rule-based classification: {:?}", rule_result.intent);
            return rule_result;
        }

        // Otherwise, use LLM
        tracing::debug!(
            "Rule-based confidence too low ({}), using LLM",
            rule_result.confidence as i32
        );
        let mut llm_result = self.llm.classify(input, ctx);

        // Mark as hybrid
        llm_result.classifier = "hybrid(llm)";
        llm_result
    }

    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn requires_external(&self) -> bool {
        true // May require external, depending on input
    }
}

// ============================================================================
// Factory and Configuration
// ============================================================================

/// Configuration for intent classification
#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Which classifier to use
    pub kind: ClassifierKind,
    /// Confidence threshold for rule-based (hybrid mode)
    pub rule_threshold: Confidence,
    /// LLM temperature
    pub llm_temperature: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            kind: ClassifierKind::RuleBased,
            rule_threshold: Confidence::Medium,
            llm_temperature: 0.1,
        }
    }
}

/// Classifier implementation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierKind {
    /// Rule-based only
    RuleBased,
    /// LLM-based only
    LLM,
    /// Hybrid: rule-based fast path + LLM fallback
    Hybrid,
}

/// Factory function to create a classifier from config
pub fn create_classifier(
    config: &ClassifierConfig,
    llm_client: Option<Arc<dyn llm::LLMClient>>,
) -> Arc<dyn IntentClassifier> {
    match config.kind {
        ClassifierKind::RuleBased => Arc::new(RuleBasedClassifier::new()),
        ClassifierKind::LLM => {
            let client = llm_client.expect("LLM classifier requires an LLM client");
            Arc::new(LLMClassifier::new(client).with_temperature(config.llm_temperature))
        }
        ClassifierKind::Hybrid => {
            let client = llm_client.expect("Hybrid classifier requires an LLM client");
            Arc::new(
                HybridClassifier::new(client).with_threshold(config.rule_threshold),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_symbol_navigation() {
        let classifier = RuleBasedClassifier::new();
        let ctx = ClassificationContext::default();

        let result = classifier.classify("foo 在哪里定义", &ctx);

        assert!(
            matches!(result.intent, Intent::SymbolNavigation { .. }),
            "Expected SymbolNavigation, got {:?}",
            result.intent
        );
        assert!(result.confidence >= Confidence::Medium);
    }

    #[test]
    fn test_rule_based_explain() {
        let classifier = RuleBasedClassifier::new();
        let ctx = ClassificationContext::default();

        let result = classifier.classify("TaskAnalyzer 有什么作用", &ctx);

        assert!(
            matches!(result.intent, Intent::ExplainSymbol { .. }),
            "Expected ExplainSymbol, got {:?}",
            result.intent
        );
    }

    #[test]
    fn test_entity_extraction() {
        let entities = RuleBasedClassifier::extract_entities("Check foo in src/main.rs:10");

        assert!(
            entities.iter().any(|e| e.kind == EntityKind::Identifier && e.value == "foo")
        );
        assert!(
            entities.iter().any(|e| e.kind == EntityKind::Path && e.value.contains("main.rs"))
        );
    }

    #[test]
    fn test_safe_command_detection() {
        assert!(is_safe_command("ls -la"));
        assert!(is_safe_command("git status"));
        assert!(is_safe_command("cat file.txt"));
        assert!(!is_safe_command("rm -rf /"));
        assert!(!is_safe_command("dd if=/dev/zero"));
    }
}
