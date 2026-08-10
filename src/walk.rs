//! Directory traversal and the filters that decide what is worth indexing.
//!
//! Built on the `ignore` crate — the ripgrep engine — so `.gitignore` is
//! honoured for free and large trees are cheap to skip.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;

#[derive(Debug, Clone)]
pub struct WalkOptions {
    /// Files above this size are skipped outright, before being opened.
    pub max_file_size: u64,
    pub follow_symlinks: bool,
    /// When non-empty, only files matching one of these globs are considered.
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub respect_gitignore: bool,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_file_size: 5 * 1024 * 1024,
            follow_symlinks: false,
            include: Vec::new(),
            exclude: Vec::new(),
            respect_gitignore: true,
        }
    }
}

/// A file that survived the filters, with the metadata the incremental check needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub path: PathBuf,
    pub size: u64,
    /// Seconds since the Unix epoch.
    pub mtime: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalkStats {
    pub too_large: usize,
    /// Entries the walker could not stat or read; reported, never fatal.
    pub unreadable: usize,
}

/// Collect indexable files under `root`, sorted by path so runs are reproducible.
pub fn walk(root: &Path, options: &WalkOptions) -> Result<(Vec<Candidate>, WalkStats)> {
    let mut overrides = OverrideBuilder::new(root);
    for glob in &options.include {
        overrides
            .add(glob)
            .with_context(|| format!("invalid --include glob `{glob}`"))?;
    }
    for glob in &options.exclude {
        // In override syntax a leading `!` marks a pattern as ignored, which is
        // the opposite of gitignore's convention.
        overrides
            .add(&format!("!{glob}"))
            .with_context(|| format!("invalid --exclude glob `{glob}`"))?;
    }
    let overrides = overrides.build().context("could not build the glob set")?;

    let mut builder = WalkBuilder::new(root);
    builder
        .overrides(overrides)
        .follow_links(options.follow_symlinks)
        .git_ignore(options.respect_gitignore)
        .git_global(options.respect_gitignore)
        .git_exclude(options.respect_gitignore)
        // Honour .gitignore even outside a repository. A directory of notes that
        // carries one has still declared what it considers junk, and behaviour
        // that flips depending on whether .git happens to exist is surprising.
        .require_git(false)
        .parents(options.respect_gitignore);

    let mut candidates = Vec::new();
    let mut stats = WalkStats::default();

    for entry in builder.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                stats.unreadable += 1;
                continue;
            }
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => {
                stats.unreadable += 1;
                continue;
            }
        };
        if metadata.len() > options.max_file_size {
            stats.too_large += 1;
            continue;
        }
        candidates.push(Candidate {
            path: entry.into_path(),
            size: metadata.len(),
            mtime: modified_seconds(&metadata),
        });
    }

    candidates.sort_by(|a, b| a.path.cmp(&b.path));
    Ok((candidates, stats))
}

/// Modification time in seconds, or 0.0 when the platform will not say.
fn modified_seconds(metadata: &std::fs::Metadata) -> f64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
