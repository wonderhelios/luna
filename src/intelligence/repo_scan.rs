use std::collections::VecDeque;
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Options controlling how repository files are discovered.
#[derive(Debug, Clone)]
pub struct RepoScanOptions {
    /// File extensions (without dot) to include, e.g. `"rs"`.
    pub include_extensions: &'static [&'static str],

    /// Directory names (not paths) to skip, e.g. `.git`, `target`.
    pub exclude_dir_names: &'static [&'static str],

    /// Skip files larger than this size.
    pub max_file_size_bytes: usize,
}

impl Default for RepoScanOptions {
    fn default() -> Self {
        Self {
            // Keep this list aligned with the language set supported by `intelligence`.
            include_extensions: &[
                "rs", "go", "py", "js", "ts", "tsx", "java", "c", "cpp", "cc", "cxx", "h", "hpp",
                "rb", "php", "r", "proto",
            ],
            exclude_dir_names: &[".git", "target"],
            max_file_size_bytes: 500 * 10usize.pow(3),
        }
    }
}

/// A file collected from a repository scan.
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub rel_path: PathBuf,
    pub abs_path: PathBuf,
    pub content: String,
}

#[derive(Debug)]
pub enum RepoScanError {
    RepoRootNotFound {
        repo_root: PathBuf,
    },

    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    StripPrefix {
        repo_root: PathBuf,
        path: PathBuf,
        source: std::path::StripPrefixError,
    },
}

impl std::fmt::Display for RepoScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepoRootNotFound { repo_root } => {
                write!(f, "repo root does not exist: {}", repo_root.display())
            }
            Self::Io { path, source } => {
                write!(f, "I/O error at {}: {source}", path.display())
            }
            Self::StripPrefix {
                repo_root,
                path,
                source,
            } => write!(
                f,
                "failed to strip prefix {} from {}: {source}",
                repo_root.display(),
                path.display()
            ),
        }
    }
}

impl std::error::Error for RepoScanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::StripPrefix { source, .. } => Some(source),
            Self::RepoRootNotFound { .. } => None,
        }
    }
}

/// Provides repository files for navigation.
///
/// This abstraction is intentionally small so we can later swap in:
/// - cached scans
/// - incremental updates
/// - prebuilt indexes
pub trait RepoFileProvider {
    fn list_files(
        &self,
        repo_root: &Path,
        opt: &RepoScanOptions,
    ) -> Result<Vec<RepoFile>, RepoScanError>;
}

/// File-system based repo scanner.
#[derive(Debug, Default, Clone)]
pub struct FsRepoFileProvider;

impl FsRepoFileProvider {
    fn should_exclude_dir(name: Option<&str>, opt: &RepoScanOptions) -> bool {
        let Some(name) = name else {
            return false;
        };
        opt.exclude_dir_names.iter().any(|&d| d == name)
    }

    fn should_include_file(path: &Path, opt: &RepoScanOptions) -> bool {
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            return false;
        };
        opt.include_extensions.iter().any(|&e| e == ext)
    }

    fn walk_dir(
        repo_root: &Path,
        dir: &Path,
        opt: &RepoScanOptions,
        acc: &mut Vec<RepoFile>,
    ) -> Result<(), RepoScanError> {
        let mut queue: VecDeque<PathBuf> = VecDeque::new();
        queue.push_back(dir.to_path_buf());

        while let Some(cur_dir) = queue.pop_front() {
            let entries = match fs::read_dir(&cur_dir) {
                Ok(v) => v,
                Err(e) => {
                    // - if the repo root itself is unreadable, fail fast
                    // - otherwise, warn and continue
                    let err = RepoScanError::Io {
                        path: cur_dir.clone(),
                        source: e,
                    };
                    if cur_dir == dir {
                        return Err(err);
                    }
                    tracing::warn!("skip unreadable dir: {cur_dir:?}, err={err}");
                    continue;
                }
            };

            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(err) => {
                        tracing::warn!("skip unreadable dir entry: {err}");
                        continue;
                    }
                };

                let path = entry.path();

                let file_type = match entry.file_type() {
                    Ok(t) => t,
                    Err(err) => {
                        tracing::warn!("skip entry with unknown type: {path:?}, err={err}");
                        continue;
                    }
                };

                // Ignore symlinks.
                if file_type.is_symlink() {
                    continue;
                }

                if file_type.is_dir() {
                    // Ignore excluded dirs.
                    if !Self::should_exclude_dir(path.file_name().and_then(|s| s.to_str()), opt) {
                        queue.push_back(path.clone());
                    }
                    continue;
                }

                // Ignore non-regular and non-included files.
                if !file_type.is_file() || !Self::should_include_file(&path, opt) {
                    continue;
                }

                // Ignore files larger than max_file_size_bytes.
                let meta = match fs::metadata(&path) {
                    Ok(m) => m,
                    Err(err) => {
                        tracing::warn!("skip file (stat failed): {path:?}, err={err}");
                        continue;
                    }
                };

                if meta.len() as usize > opt.max_file_size_bytes {
                    continue;
                }

                let bytes = match fs::read(&path) {
                    Ok(b) => b,
                    Err(err) => {
                        tracing::warn!("skip file (read failed): {path:?}, err={err}");
                        continue;
                    }
                };
                if bytes.len() > opt.max_file_size_bytes {
                    continue;
                }

                let content = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_owned(),
                    Err(_) => {
                        tracing::warn!("skip file (non-utf8): {path:?}");
                        continue;
                    }
                };

                let rel_path =
                    path.strip_prefix(repo_root)
                        .map_err(|e| RepoScanError::StripPrefix {
                            repo_root: repo_root.to_path_buf(),
                            path: path.clone(),
                            source: e,
                        })?;

                acc.push(RepoFile {
                    rel_path: rel_path.to_path_buf(),
                    abs_path: path.clone(),
                    content,
                });
            }
        }

        Ok(())
    }
}

impl RepoFileProvider for FsRepoFileProvider {
    fn list_files(
        &self,
        repo_root: &Path,
        opt: &RepoScanOptions,
    ) -> Result<Vec<RepoFile>, RepoScanError> {
        if !repo_root.exists() {
            return Err(RepoScanError::RepoRootNotFound {
                repo_root: repo_root.to_path_buf(),
            });
        }

        let mut acc = Vec::new();
        Self::walk_dir(repo_root, repo_root, opt, &mut acc)?;
        Ok(acc)
    }
}
