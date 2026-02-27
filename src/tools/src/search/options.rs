use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCodeOptions {
    pub max_files: usize,
    pub max_hits: usize,
    pub max_file_bytes: usize,
    pub ignore_dirs: Vec<String>,
}

impl Default for SearchCodeOptions {
    fn default() -> Self {
        Self {
            max_files: 8_000,
            max_hits: 64,
            max_file_bytes: 500 * 1_000,
            ignore_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "dist".to_string(),
                "build".to_string(),
            ],
        }
    }
}
