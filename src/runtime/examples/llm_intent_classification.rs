//! Test LLM-based intent classification with DeepSeek API
//!
//! Run with:
//!   LUNA_DEEPSEEK_API_KEY=sk-xxx cargo run --example llm_intent_classification -p runtime

use std::io::Write;
use std::sync::Arc;

use llm::{OpenAIClient, OpenAIConfig};
use runtime::intent_classifier::{
    ClassificationContext, IntentClassifier, LLMClassifier, RuleBasedClassifier,
};

#[tokio::main]
async fn main() {
    // Get API key from environment
    let api_key = std::env::var("LUNA_DEEPSEEK_API_KEY")
        .expect("Please set LUNA_DEEPSEEK_API_KEY environment variable");

    println!("=== LLM Intent Classification Test (DeepSeek) ===\n");

    // Create DeepSeek client
    let config = OpenAIConfig {
        base_url: "https://api.deepseek.com/v1".to_owned(),
        api_key,
        model: "deepseek-chat".to_owned(),
        ..OpenAIConfig::default()
    };

    let client = Arc::new(
        OpenAIClient::new(config).expect("Failed to create DeepSeek client")
    );

    // Create LLM classifier
    let llm_classifier = LLMClassifier::new(client);

    // Also create rule-based for comparison
    let rule_classifier = RuleBasedClassifier::new();

    let ctx = ClassificationContext::default();

    // Test cases - progressively more complex
    let test_cases = vec![
        // Basic navigation
        "查找 foo 函数的定义",
        "go to definition of TaskAnalyzer",

        // Explanation
        "解释一下 RefillPipeline 是做什么的",
        "what does the ContextQuery struct do",

        // Edit intents
        "帮我把 main.rs 第 10 行的 println! 改成 dbg!",
        "optimize the calculate_sum function to use iterator",

        // Terminal/Command
        "运行 cargo test 看看测试是否通过",
        "check git status",

        // Search
        "搜索所有使用了 unsafe 的代码块",
        "find all TODO comments in the codebase",

        // Complex natural language (these are hard for rule-based)
        "这段代码看起来有性能问题，能帮我看看吗",
        "我觉得这个函数命名不太对，想改个名字",
        "昨天加的 authenticate_user 函数在哪里",
        "show me how errors are handled in this project",
        "I need to refactor the context pipeline to support caching",

        // Ambiguous/Clarification needed
        "修改一下",
        "fix it",

        // Chat
        "你好",
        "今天天气怎么样",
    ];

    println!("Testing {} input cases...\n", test_cases.len());

    for (i, input) in test_cases.iter().enumerate() {
        println!("{}", "─".repeat(60));
        println!("[{}] Input: {}", i + 1, input);
        println!();

        // Rule-based result
        let rule_result = rule_classifier.classify(input, &ctx);
        println!("  Rule-based:");
        println!("    Intent: {:?}", rule_result.intent);
        println!("    Confidence: {:?}", rule_result.confidence);
        println!("    Entities: {:?}", rule_result.entities.iter().map(|e| format!("{:?}({})", e.kind, e.value)).collect::<Vec<_>>().join(", "));

        // LLM result
        print!("\n  LLM (DeepSeek): ");
        std::io::stdout().flush().unwrap();

        let start = std::time::Instant::now();
        let llm_result = llm_classifier.classify(input, &ctx);
        let elapsed = start.elapsed();

        println!("({:.2}s)", elapsed.as_secs_f32());
        println!("    Intent: {:?}", llm_result.intent);
        println!("    Confidence: {:?}", llm_result.confidence);
        println!("    Entities: {:?}", llm_result.entities.iter().map(|e| format!("{:?}({})", e.kind, e.value)).collect::<Vec<_>>().join(", "));

        // Comparison
        let rule_conf = rule_result.confidence as i32;
        let llm_conf = llm_result.confidence as i32;

        if rule_conf < llm_conf {
            println!("\n  → LLM has higher confidence (improvement!)");
        } else if rule_conf > llm_conf {
            println!("\n  → Rule-based has higher confidence");
        } else {
            println!("\n  → Same confidence level");
        }

        // Show raw LLM response if available
        if let Some(raw) = llm_result.metadata.get("raw_response") {
            if raw.len() > 200 {
                println!("\n  Raw LLM response: {}...", &raw[..200]);
            } else {
                println!("\n  Raw LLM response: {}", raw);
            }
        }

        println!();
    }

    println!("\n=== Summary ===");
    println!("Complex cases where LLM should outperform rule-based:");
    println!("- '这段代码看起来有性能问题，能帮我看看吗' → should be Search/Explain");
    println!("- '我觉得这个函数命名不太对，想改个名字' → should be Edit with partial info");
    println!("- '昨天加的 authenticate_user 函数在哪里' → should be SymbolNavigation");
    println!("- 'I need to refactor the context pipeline' → should be Edit/Refactor");
}
