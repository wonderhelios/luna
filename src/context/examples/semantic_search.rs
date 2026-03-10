//! Semantic Search Example
//!
//! This example demonstrates how to use the vector-based semantic search
//! to find relevant code using natural language queries.
//!
//! Run with:
//!   cargo run --example semantic_search -p context
//!
//! Or with Ollama local embedding:
//!   cargo run --example semantic_search -p context --features ollama

use std::sync::Arc;

use context::vector::{
    ChunkingStrategy, DistanceMetric, MockEmbedding, SemanticRetriever, SemanticRetrieverBuilder,
    SemanticSearchOptions,
};

fn main() {
    println!("=== Luna Semantic Search Example ===\n");

    // Create a mock embedding model for demonstration
    // In production, use OllamaEmbedding or OpenAIEmbedding
    let embedding = Arc::new(MockEmbedding::new(128));

    // Create semantic retriever
    let temp_dir = tempfile::tempdir().unwrap();
    let mut retriever = SemanticRetriever::with_config(
        embedding,
        temp_dir.path().to_path_buf(),
        ChunkingStrategy::Semantic,
        256, // max chunk tokens
    );

    // Index some sample code
    println!("Indexing sample code...\n");

    let sample_code = r#"
/// Calculates the factorial of a number
pub fn factorial(n: u64) -> u64 {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

/// Finds the maximum value in a slice
pub fn find_max<T: Ord>(values: &[T]) -> Option<&T> {
    values.iter().max()
}

/// A simple user struct
pub struct User {
    pub name: String,
    pub email: String,
    pub age: u32,
}

impl User {
    /// Creates a new user
    pub fn new(name: impl Into<String>, email: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            email: email.into(),
            age: 0,
        }
    }

    /// Sets the user's age
    pub fn with_age(mut self, age: u32) -> Self {
        self.age = age;
        self
    }

    /// Validates the user's email
    pub fn is_valid_email(&self) -> bool {
        self.email.contains('@') && self.email.contains('.')
    }
}

/// Error types for authentication
#[derive(Debug)]
pub enum AuthError {
    InvalidCredentials,
    UserNotFound,
    DatabaseError(String),
}

/// Authenticates a user
pub fn authenticate_user(username: &str, password: &str) -> Result<User, AuthError> {
    // Implementation would check credentials
    if username.is_empty() || password.is_empty() {
        return Err(AuthError::InvalidCredentials);
    }

    Ok(User::new(username, format!("{}@example.com", username)))
}
"#;

    retriever
        .index_file_content(
            temp_dir.path().join("src/lib.rs").as_path(),
            sample_code,
        )
        .unwrap();

    println!("Indexed {} chunks\n", retriever.len());

    // Perform searches
    let queries = vec![
        "how to calculate factorial",
        "find maximum value in array",
        "user creation and validation",
        "authentication error handling",
        "struct definition",
    ];

    let options = SemanticSearchOptions::default();

    for query in queries {
        println!("Query: \"{}\"", query);
        println!("{}", "-".repeat(50));

        match retriever.search(query, &options) {
            Ok(results) => {
                if results.is_empty() {
                    println!("  No results found\n");
                } else {
                    for (i, result) in results.iter().take(3).enumerate() {
                        println!(
                            "  {}. {} (score: {:.2})",
                            i + 1,
                            result.chunk.id,
                            result.score
                        );
                        // Show first 100 chars of content
                        let preview: String = result
                            .chunk
                            .content
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(80)
                            .collect();
                        println!("     {}...", preview);
                    }
                    println!();
                }
            }
            Err(e) => {
                println!("  Error: {}\n", e);
            }
        }
    }

    // Example: Using builder pattern
    println!("\n=== Builder Pattern Example ===\n");

    let _builder_example = SemanticRetrieverBuilder::default()
        .repo_root(temp_dir.path())
        .chunking_strategy(ChunkingStrategy::Hierarchical)
        .max_chunk_tokens(512)
        .distance_metric(DistanceMetric::Cosine);

    println!("Builder configured successfully!");
    println!("In production, you would call .embedding(your_model).build()");

    // Example: Using Ollama (commented out - requires running Ollama server)
    /*
    println!("\n=== Ollama Integration Example ===\n");

    let ollama_embedding = OllamaEmbedding::nomic_embed_text();

    if ollama_embedding.is_available() {
        println!("Ollama server is available!");

        if let Err(e) = ollama_embedding.ensure_model() {
            println!("Failed to ensure model: {}", e);
        } else {
            let ollama_retriever = SemanticRetrieverBuilder::default()
                .embedding(Arc::new(ollama_embedding))
                .repo_root(temp_dir.path())
                .build()
                .unwrap();

            println!("Ollama retriever created successfully!");
        }
    } else {
        println!("Ollama server not available.");
        println!("Install from https://ollama.ai and run: ollama serve");
        println!("Then: ollama pull nomic-embed-text");
    }
    */

    println!("\n=== Done ===");
}
