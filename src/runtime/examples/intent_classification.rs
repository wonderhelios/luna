//! Example: Intent Classification Usage
//!
//! This example demonstrates how to use the new intent classification system.
//!
//! Run with:
//!   cargo run --example intent_classification -- -p runtime

use std::sync::Arc;

use runtime::intent_classifier::{
    ClassificationContext, ClassifierConfig, ClassifierKind, IntentClassifier,
    RuleBasedClassifier,
};

fn main() {
    println!("=== Luna Intent Classification Examples ===\n");

    // Create a rule-based classifier (fast, no external calls)
    let classifier = RuleBasedClassifier::new();
    let ctx = ClassificationContext::default();

    let examples = vec![
        "foo 在哪里定义",
        "TaskAnalyzer 有什么作用",
        "修改 main.rs 第 10 行",
        "运行 cargo test",
        "帮我优化这段代码",
        "什么是 RefillPipeline",
        "查找所有 TODO 注释",
        "你好",
    ];

    for input in examples {
        let result = classifier.classify(input, &ctx);

        println!("Input:    {}", input);
        println!("Intent:   {:?}", result.intent);
        println!("Confidence: {:?}", result.confidence);
        println!("Entities: {:?}", result.entities);
        println!("Classifier: {}", result.classifier);
        println!("---");
    }

    println!("\n=== Using LLM Classifier ===\n");
    println!("To use the LLM classifier, you need an LLM client:");
    println!();
    println!("use runtime::intent_classifier::{{");
    println!("    LLMClassifier, HybridClassifier, create_classifier");
    println!("}};");
    println!();
    println!("// Create LLM client");
    println!("let llm_client = Arc::new(OpenAIClient::new(...));");
    println!();
    println!("// Option 1: Pure LLM classifier");
    println!("let llm_classifier = LLMClassifier::new(llm_client.clone());");
    println!();
    println!("// Option 2: Hybrid classifier (rule-based fast path + LLM fallback)");
    println!("let hybrid = HybridClassifier::new(llm_client.clone());");
    println!();
    println!("// Option 3: Using factory function");
    println!("let config = ClassifierConfig {{");
    println!("    kind: ClassifierKind::Hybrid,");
    println!("    rule_threshold: Confidence::Medium,");
    println!("    llm_temperature: 0.1,");
    println!("}};");
    println!("let classifier = create_classifier(&config, Some(llm_client));");

    println!("\n=== Future: Local Model Support ===\n");
    println!("To add a local model (e.g., via Ollama or llama.cpp):");
    println!();
    println!("pub struct LocalModelClassifier {{");
    println!("    model: Box<dyn LocalInferenceEngine>,");
    println!("}}");
    println!();
    println!("impl IntentClassifier for LocalModelClassifier {{");
    println!("    fn classify(&self, input: &str, ctx: &ClassificationContext) -> ClassificationResult {{");
    println!("        // Call local model");
    println!("        let output = self.model.infer(build_prompt(input, ctx));");
    println!("        parse_classification(&output)");
    println!("    }}");
    println!("}}");
}
