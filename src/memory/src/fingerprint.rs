//! Project fingerprinting - auto-detect project type and characteristics

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Detected project type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectType {
    /// Rust project
    Rust,
    /// Rust workspace
    RustWorkspace,
    /// Python project
    Python,
    /// Node.js/JavaScript project
    NodeJs,
    /// TypeScript project
    TypeScript,
    /// Go project
    Go,
    /// Java project
    Java,
    /// Kotlin project
    Kotlin,
    /// C/C++ project
    Cpp,
    /// .NET/C# project
    DotNet,
    /// Ruby project
    Ruby,
    /// PHP project
    Php,
    /// Swift project
    Swift,
    /// Zig project
    Zig,
    /// Generic project with build files
    GenericBuild,
    /// Unknown/simple project
    Unknown,
}

impl ProjectType {
    /// Get the name of this project type
    pub fn name(&self) -> &'static str {
        match self {
            ProjectType::Rust => "Rust",
            ProjectType::RustWorkspace => "Rust Workspace",
            ProjectType::Python => "Python",
            ProjectType::NodeJs => "Node.js",
            ProjectType::TypeScript => "TypeScript",
            ProjectType::Go => "Go",
            ProjectType::Java => "Java",
            ProjectType::Kotlin => "Kotlin",
            ProjectType::Cpp => "C/C++",
            ProjectType::DotNet => ".NET",
            ProjectType::Ruby => "Ruby",
            ProjectType::Php => "PHP",
            ProjectType::Swift => "Swift",
            ProjectType::Zig => "Zig",
            ProjectType::GenericBuild => "Generic Build",
            ProjectType::Unknown => "Unknown",
        }
    }

    /// Get default build command for this project type
    pub fn default_build_command(&self) -> Option<&'static str> {
        match self {
            ProjectType::Rust | ProjectType::RustWorkspace => Some("cargo build"),
            ProjectType::Go => Some("go build"),
            ProjectType::NodeJs | ProjectType::TypeScript => Some("npm run build"),
            ProjectType::Python => None, // Python typically doesn't need build
            ProjectType::Java | ProjectType::Kotlin => Some("./gradlew build"),
            ProjectType::Cpp => Some("make"),
            ProjectType::DotNet => Some("dotnet build"),
            ProjectType::Ruby => None,
            ProjectType::Php => None,
            ProjectType::Swift => Some("swift build"),
            ProjectType::Zig => Some("zig build"),
            ProjectType::GenericBuild => Some("make"),
            ProjectType::Unknown => None,
        }
    }

    /// Get default test command for this project type
    pub fn default_test_command(&self) -> Option<&'static str> {
        match self {
            ProjectType::Rust | ProjectType::RustWorkspace => Some("cargo test"),
            ProjectType::Go => Some("go test"),
            ProjectType::NodeJs | ProjectType::TypeScript => Some("npm test"),
            ProjectType::Python => Some("pytest"),
            ProjectType::Java | ProjectType::Kotlin => Some("./gradlew test"),
            ProjectType::Cpp => Some("make test"),
            ProjectType::DotNet => Some("dotnet test"),
            ProjectType::Ruby => Some("bundle exec rake"),
            ProjectType::Php => Some("phpunit"),
            ProjectType::Swift => Some("swift test"),
            ProjectType::Zig => Some("zig build test"),
            ProjectType::GenericBuild => Some("make test"),
            ProjectType::Unknown => None,
        }
    }

    /// Get default check/lint command for this project type
    pub fn default_check_command(&self) -> Option<&'static str> {
        match self {
            ProjectType::Rust | ProjectType::RustWorkspace => Some("cargo clippy"),
            ProjectType::Go => Some("go vet"),
            ProjectType::NodeJs | ProjectType::TypeScript => Some("npm run lint"),
            ProjectType::Python => Some("ruff check ."),
            ProjectType::Java | ProjectType::Kotlin => Some("./gradlew check"),
            ProjectType::Cpp => None,
            ProjectType::DotNet => Some("dotnet analyze"),
            ProjectType::Ruby => Some("rubocop"),
            ProjectType::Php => Some("phpstan"),
            ProjectType::Swift => Some("swiftlint"),
            ProjectType::Zig => Some("zig fmt --check"),
            ProjectType::GenericBuild => None,
            ProjectType::Unknown => None,
        }
    }
}

/// Project fingerprint containing detected characteristics
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ProjectFingerprint {
    /// Detected project type(s)
    pub project_types: Vec<ProjectType>,
    /// Primary language
    pub primary_language: Option<String>,
    /// Project root files found
    pub root_files: Vec<PathBuf>,
    /// Source directories found
    pub src_dirs: Vec<PathBuf>,
    /// Has git repository
    pub has_git: bool,
    /// Has CI configuration
    pub has_ci: bool,
    /// CI provider detected
    pub ci_provider: Option<CiProvider>,
}

/// CI/CD provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CiProvider {
    GitHubActions,
    GitLabCi,
    TravisCi,
    CircleCi,
    Jenkins,
    AzurePipelines,
    Drone,
}

impl ProjectFingerprint {
    /// Detect project fingerprint from directory
    pub fn detect(project_root: &Path) -> Self {
        let mut types = HashSet::new();
        let mut root_files = Vec::new();
        let mut src_dirs = Vec::new();

        // Check for Rust
        if project_root.join("Cargo.toml").exists() {
            let cargo_toml = project_root.join("Cargo.toml");
            root_files.push(cargo_toml.clone());

            // Check if workspace
            if is_cargo_workspace(&cargo_toml) {
                types.insert(ProjectType::RustWorkspace);
            } else {
                types.insert(ProjectType::Rust);
            }
        }

        // Check for Python
        if project_root.join("pyproject.toml").exists()
            || project_root.join("setup.py").exists()
            || project_root.join("requirements.txt").exists()
            || project_root.join("Pipfile").exists()
        {
            types.insert(ProjectType::Python);
            root_files.push(project_root.join("pyproject.toml"));
        }

        // Check for Node.js
        if project_root.join("package.json").exists() {
            root_files.push(project_root.join("package.json"));

            // Check if TypeScript
            if project_root.join("tsconfig.json").exists() {
                types.insert(ProjectType::TypeScript);
                root_files.push(project_root.join("tsconfig.json"));
            } else {
                types.insert(ProjectType::NodeJs);
            }
        }

        // Check for Go
        if project_root.join("go.mod").exists() {
            types.insert(ProjectType::Go);
            root_files.push(project_root.join("go.mod"));
        }

        // Check for Java/Kotlin
        if project_root.join("pom.xml").exists() || project_root.join("build.gradle").exists() {
            if project_root.join("build.gradle.kts").exists()
                || has_kotlin_files(project_root)
            {
                types.insert(ProjectType::Kotlin);
                root_files.push(project_root.join("build.gradle.kts"));
            } else {
                types.insert(ProjectType::Java);
                root_files.push(project_root.join("build.gradle"));
            }
        }

        // Check for C/C++
        if project_root.join("CMakeLists.txt").exists()
            || project_root.join("Makefile").exists()
            || has_c_or_cpp_files(project_root)
        {
            types.insert(ProjectType::Cpp);
            if project_root.join("CMakeLists.txt").exists() {
                root_files.push(project_root.join("CMakeLists.txt"));
            }
            if project_root.join("Makefile").exists() {
                root_files.push(project_root.join("Makefile"));
            }
        }

        // Check for .NET
        if has_extension(project_root, ".csproj") || has_extension(project_root, ".sln") {
            types.insert(ProjectType::DotNet);
        }

        // Check for Ruby
        if project_root.join("Gemfile").exists() {
            types.insert(ProjectType::Ruby);
            root_files.push(project_root.join("Gemfile"));
        }

        // Check for PHP
        if project_root.join("composer.json").exists() {
            types.insert(ProjectType::Php);
            root_files.push(project_root.join("composer.json"));
        }

        // Check for Swift
        if project_root.join("Package.swift").exists() {
            types.insert(ProjectType::Swift);
            root_files.push(project_root.join("Package.swift"));
        }

        // Check for Zig
        if project_root.join("build.zig").exists() {
            types.insert(ProjectType::Zig);
            root_files.push(project_root.join("build.zig"));
        }

        // Check for git
        let has_git = project_root.join(".git").is_dir();

        // Check for CI
        let (has_ci, ci_provider) = detect_ci_provider(project_root);

        // Detect source directories
        for dir in ["src", "lib", "app", "source"] {
            if project_root.join(dir).is_dir() {
                src_dirs.push(project_root.join(dir));
            }
        }

        // Determine primary language
        let primary_language = if let Some(first_type) = types.iter().next() {
            Some(first_type.name().to_owned())
        } else {
            None
        };

        // If no specific type detected but has Makefile
        if types.is_empty() && project_root.join("Makefile").exists() {
            types.insert(ProjectType::GenericBuild);
            root_files.push(project_root.join("Makefile"));
        }

        // If still unknown, check for common source files
        if types.is_empty() {
            types.insert(ProjectType::Unknown);
        }

        Self {
            project_types: types.into_iter().collect(),
            primary_language,
            root_files: root_files.into_iter().filter(|p| p.exists()).collect(),
            src_dirs,
            has_git,
            has_ci,
            ci_provider,
        }
    }

    /// Get the primary project type
    pub fn primary_type(&self) -> Option<ProjectType> {
        self.project_types.first().copied()
    }

    /// Check if this is a specific project type
    pub fn is_type(&self, project_type: ProjectType) -> bool {
        self.project_types.contains(&project_type)
    }

    /// Get suggested build commands
    pub fn suggested_build_commands(&self) -> Vec<&'static str> {
        let mut commands = Vec::new();
        for pt in &self.project_types {
            if let Some(cmd) = pt.default_build_command() {
                if !commands.contains(&cmd) {
                    commands.push(cmd);
                }
            }
        }
        commands
    }

    /// Get suggested test commands
    pub fn suggested_test_commands(&self) -> Vec<&'static str> {
        let mut commands = Vec::new();
        for pt in &self.project_types {
            if let Some(cmd) = pt.default_test_command() {
                if !commands.contains(&cmd) {
                    commands.push(cmd);
                }
            }
        }
        commands
    }
}

/// Check if Cargo.toml defines a workspace
fn is_cargo_workspace(cargo_toml: &Path) -> bool {
    if let Ok(content) = std::fs::read_to_string(cargo_toml) {
        // Simple heuristic: check for [workspace] section
        content.contains("[workspace]")
    } else {
        false
    }
}

/// Check if project has Kotlin files
fn has_kotlin_files(project_root: &Path) -> bool {
    has_extension(project_root, ".kt")
        || has_extension(project_root, ".kts")
}

/// Check if project has C/C++ files
fn has_c_or_cpp_files(project_root: &Path) -> bool {
    has_extension(project_root, ".c")
        || has_extension(project_root, ".cpp")
        || has_extension(project_root, ".cc")
        || has_extension(project_root, ".h")
        || has_extension(project_root, ".hpp")
}

/// Check if any file in project has extension
fn has_extension(project_root: &Path, ext: &str) -> bool {
    walkdir::WalkDir::new(project_root)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
        .any(|e| {
            e.path()
                .extension()
                .map(|e| e == &ext[1..])
                .unwrap_or(false)
        })
}

/// Detect CI provider from configuration files
fn detect_ci_provider(project_root: &Path) -> (bool, Option<CiProvider>) {
    if project_root.join(".github/workflows").is_dir() {
        return (true, Some(CiProvider::GitHubActions));
    }
    if project_root.join(".gitlab-ci.yml").exists() {
        return (true, Some(CiProvider::GitLabCi));
    }
    if project_root.join(".travis.yml").exists() {
        return (true, Some(CiProvider::TravisCi));
    }
    if project_root.join(".circleci").is_dir() {
        return (true, Some(CiProvider::CircleCi));
    }
    if project_root.join("Jenkinsfile").exists() {
        return (true, Some(CiProvider::Jenkins));
    }
    if project_root.join("azure-pipelines.yml").exists() {
        return (true, Some(CiProvider::AzurePipelines));
    }
    if project_root.join(".drone.yml").exists() {
        return (true, Some(CiProvider::Drone));
    }
    (false, None)
}

// Add walkdir dependency for file walking
use walkdir;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_rust_project() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();

        let fp = ProjectFingerprint::detect(tmp.path());

        assert!(fp.is_type(ProjectType::Rust));
        assert_eq!(fp.primary_type(), Some(ProjectType::Rust));
        assert!(fp.root_files.iter().any(|p| p.ends_with("Cargo.toml")));
    }

    #[test]
    fn test_detect_rust_workspace() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"a\", \"b\"]",
        )
        .unwrap();

        let fp = ProjectFingerprint::detect(tmp.path());

        assert!(fp.is_type(ProjectType::RustWorkspace));
    }

    #[test]
    fn test_detect_nodejs() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("package.json"), "{}").unwrap();

        let fp = ProjectFingerprint::detect(tmp.path());

        assert!(fp.is_type(ProjectType::NodeJs));
    }

    #[test]
    fn test_suggested_commands() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();

        let fp = ProjectFingerprint::detect(tmp.path());

        let build = fp.suggested_build_commands();
        assert!(build.contains(&"cargo build"));

        let test = fp.suggested_test_commands();
        assert!(test.contains(&"cargo test"));
    }

    #[test]
    fn test_project_type_name() {
        assert_eq!(ProjectType::Rust.name(), "Rust");
        assert_eq!(ProjectType::NodeJs.name(), "Node.js");
    }
}
