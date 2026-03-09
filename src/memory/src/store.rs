//! Memory persistence - load/save project memory

use std::path::{Path, PathBuf};

use crate::ProjectMemory;

/// Storage backend for project memory
#[derive(Debug, Clone)]
pub struct MemoryStore {
    /// Path to memory file
    path: PathBuf,
    /// Format for serialization
    format: StorageFormat,
}

/// Storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    /// JSON format
    Json,
    /// YAML format (requires feature)
    #[cfg(feature = "yaml")]
    Yaml,
}

impl Default for StorageFormat {
    fn default() -> Self {
        Self::Json
    }
}

impl MemoryStore {
    /// Create a new memory store at the given path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: StorageFormat::default(),
        }
    }

    /// Create with specific format
    pub fn with_format(path: impl Into<PathBuf>, format: StorageFormat) -> Self {
        Self {
            path: path.into(),
            format,
        }
    }

    /// Create a store in the project directory
    ///
    /// Uses `.luna/memory.json` by default
    pub fn in_project(project_root: impl AsRef<Path>) -> Self {
        let mut path = project_root.as_ref().to_path_buf();
        path.push(".luna");
        path.push("memory.json");
        Self::new(path)
    }

    /// Create a store using LUNA.md file
    ///
    /// This stores memory in a special section of LUNA.md
    pub fn luna_md(project_root: impl AsRef<Path>) -> Self {
        let mut path = project_root.as_ref().to_path_buf();
        path.push("LUNA.md");
        Self::new(path)
    }

    /// Get the storage path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if memory file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Load memory from storage
    pub fn load(&self) -> error::Result<ProjectMemory> {
        if !self.exists() {
            return Err(error::LunaError::invalid_input(
                "Memory file does not exist",
            ));
        }

        let content = std::fs::read_to_string(&self.path)?;

        // Check if this is LUNA.md format
        if self.path.file_name().map(|n| n == "LUNA.md").unwrap_or(false) {
            self.parse_from_luna_md(&content)
        } else {
            self.parse_from_json(&content)
        }
    }

    /// Load or create default memory
    pub fn load_or_default(&self, project_root: &Path) -> ProjectMemory {
        self.load().unwrap_or_else(|_| {
            let mut memory = ProjectMemory::detect(project_root);
            // Try to save the default memory
            let _ = self.save(&memory);
            memory
        })
    }

    /// Save memory to storage
    pub fn save(&self, memory: &ProjectMemory) -> error::Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = if self.path.file_name().map(|n| n == "LUNA.md").unwrap_or(false) {
            self.serialize_to_luna_md(memory)?
        } else {
            serde_json::to_string_pretty(memory)?
        };

        std::fs::write(&self.path, content)?;
        Ok(())
    }

    /// Update memory (load, modify, save)
    pub fn update<F>(&self, project_root: &Path, f: F) -> error::Result<()>
    where
        F: FnOnce(&mut ProjectMemory),
    {
        let mut memory = self.load_or_default(project_root);
        f(&mut memory);
        self.save(&memory)
    }

    /// Parse memory from JSON content
    fn parse_from_json(&self, content: &str) -> error::Result<ProjectMemory> {
        let memory: ProjectMemory = serde_json::from_str(content)?;
        Ok(memory)
    }

    /// Parse memory from LUNA.md content
    ///
    /// LUNA.md format:
    /// ```markdown
    /// # LUNA Configuration
    ///
    /// ## Commands
    ///
    /// - Build: `cargo build --release`
    /// - Test: `cargo test`
    ///
    /// ## Memory
    ///
    /// ```json
    /// { ...memory json... }
    /// ```
    /// ```
    fn parse_from_luna_md(
        &self,
        content: &str,
    ) -> error::Result<ProjectMemory> {
        // Find JSON block in ## Memory section
        if let Some(memory_section) = content.split("## Memory").nth(1) {
            if let Some(json_block) = memory_section.split("```json").nth(1) {
                if let Some(json_content) = json_block.split("```").next() {
                    return self.parse_from_json(json_content.trim());
                }
            }
        }

        // If no memory section found, try parsing the whole file as JSON
        self.parse_from_json(content)
    }

    /// Serialize memory to LUNA.md format
    fn serialize_to_luna_md(
        &self,
        memory: &ProjectMemory,
    ) -> error::Result<String> {
        let json = serde_json::to_string_pretty(memory)?;

        let mut output = String::new();
        output.push_str("# LUNA Configuration\n\n");
        output.push_str("This file contains project-specific configuration for Luna AI assistant.\n\n");

        // Add Commands section from memory
        output.push_str("## Commands\n\n");
        for cmd_type in [
            crate::learner::CommandType::Build,
            crate::learner::CommandType::Test,
            crate::learner::CommandType::Check,
        ] {
            if let Some(cmd) = memory.best_command(cmd_type) {
                output.push_str(&format!(
                    "- {}: `{}`\n",
                    format!("{:?}", cmd_type),
                    cmd
                ));
            }
        }

        // Add Preferences section
        output.push_str("\n## Preferences\n\n");
        output.push_str(&format!("- Style: {:?}\n", memory.preferences.style));
        output.push_str(&format!(
            "- Confirm edits: {}\n",
            memory.preferences.confirm_edits
        ));

        // Add Memory section with JSON
        output.push_str("\n## Memory\n\n");
        output.push_str("```json\n");
        output.push_str(&json);
        output.push_str("\n```\n");

        Ok(output)
    }

    /// Merge with another memory store (for multi-project workspaces)
    pub fn merge(&self,
        _other: &MemoryStore,
    ) -> error::Result<ProjectMemory> {
        // TODO: Implement merging logic for workspaces
        todo!("Memory merging not yet implemented")
    }
}

/// Find memory store for a path (walk up directory tree)
pub fn find_memory_store(start_path: impl AsRef<Path>) -> Option<MemoryStore> {
    let mut current = start_path.as_ref();

    loop {
        // Check for .luna/memory.json
        let luna_path = current.join(".luna").join("memory.json");
        if luna_path.exists() {
            return Some(MemoryStore::new(luna_path));
        }

        // Check for LUNA.md
        let luna_md = current.join("LUNA.md");
        if luna_md.exists() {
            return Some(MemoryStore::luna_md(current));
        }

        // Walk up
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    None
}

/// Auto-detect project and create appropriate store
pub fn auto_detect(path: impl AsRef<Path>) -> MemoryStore {
    let path = path.as_ref();

    // Try to find existing store
    if let Some(store) = find_memory_store(path) {
        return store;
    }

    // Create new store at project root (with git) or current directory
    let mut current = path;
    loop {
        if current.join(".git").is_dir() {
            return MemoryStore::in_project(current);
        }

        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    // Fallback to current directory
    MemoryStore::in_project(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_memory_store_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = MemoryStore::new(tmp.path().join("memory.json"));

        let mut memory = ProjectMemory::default();
        memory.record_command(crate::learner::CommandType::Build, "make", true);

        // Save
        store.save(&memory).unwrap();
        assert!(store.exists());

        // Load
        let loaded = store.load().unwrap();
        assert_eq!(loaded.best_command(crate::learner::CommandType::Build), Some("make"));
    }

    #[test]
    fn test_in_project() {
        let tmp = TempDir::new().unwrap();
        let store = MemoryStore::in_project(tmp.path());

        assert_eq!(
            store.path(),
            tmp.path().join(".luna").join("memory.json")
        );
    }

    #[test]
    fn test_luna_md_format() {
        let tmp = TempDir::new().unwrap();
        let store = MemoryStore::luna_md(tmp.path());

        let mut memory = ProjectMemory::default();
        memory.record_command(crate::learner::CommandType::Build, "cargo build", true);
        memory.preferences.style = crate::ProgrammingStyle::Performance;

        // Save to LUNA.md
        store.save(&memory).unwrap();

        // Read raw content
        let content = fs::read_to_string(store.path()).unwrap();
        println!("LUNA.md content:\n{}", content);

        // Verify markdown format
        assert!(content.contains("# LUNA Configuration"));
        assert!(content.contains("## Commands"));
        assert!(content.contains("## Memory"));
        assert!(content.contains("```json"));

        // Load and verify
        let loaded = store.load().unwrap();
        assert_eq!(
            loaded.best_command(crate::learner::CommandType::Build),
            Some("cargo build")
        );
    }

    #[test]
    fn test_find_memory_store() {
        let tmp = TempDir::new().unwrap();
        let luna_dir = tmp.path().join(".luna");
        fs::create_dir_all(&luna_dir).unwrap();
        fs::write(luna_dir.join("memory.json"), "{}").unwrap();

        let subdir = tmp.path().join("src").join("nested");
        fs::create_dir_all(&subdir).unwrap();

        let found = find_memory_store(&subdir);
        assert!(found.is_some());
        assert!(found.unwrap().exists());
    }

    #[test]
    fn test_load_or_default() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();

        let store = MemoryStore::in_project(tmp.path());
        let memory = store.load_or_default(tmp.path());

        // Should detect Rust project
        assert!(memory.fingerprint.is_type(crate::fingerprint::ProjectType::Rust));
    }
}
