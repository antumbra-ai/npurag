//! Orchestration: walk, extract, chunk, embed, store — incrementally.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};

use crate::backend::Backend;
use crate::chunk::{chunk_text, ChunkOptions};
use crate::extract::{extract, Extraction, SkipReason};
use crate::store::Store;
use crate::walk::{walk, WalkOptions};

#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Rebuild everything, ignoring what the index already knows.
    pub reindex: bool,
    /// How many chunks are sent per embeddings request.
    pub batch_size: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            reindex: false,
            batch_size: 32,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexReport {
    pub scanned: usize,
    pub indexed: usize,
    pub unchanged: usize,
    pub skipped_binary: usize,
    pub skipped_unsupported: usize,
    pub skipped_too_large: usize,
    pub unreadable: usize,
    pub removed: usize,
    pub chunks_written: usize,
}

pub fn index_dir(
    store: &mut Store,
    backend: &dyn Backend,
    root: &Path,
    walk_options: &WalkOptions,
    chunk_options: &ChunkOptions,
    options: &IndexOptions,
) -> Result<IndexReport> {
    let root = root
        .canonicalize()
        .with_context(|| format!("could not resolve {}", root.display()))?;

    if options.reindex {
        store.clear()?;
    }

    let (candidates, walk_stats) = walk(&root, walk_options)?;
    let mut report = IndexReport {
        scanned: candidates.len(),
        skipped_too_large: walk_stats.too_large,
        unreadable: walk_stats.unreadable,
        ..Default::default()
    };

    let mut seen: HashSet<String> = HashSet::with_capacity(candidates.len());

    for candidate in candidates {
        let path = candidate.path.to_string_lossy().into_owned();
        seen.insert(path.clone());

        let known = store.file_record(&path)?;

        // Fast path: size and mtime agree, so do not even open the file.
        if let Some(record) = &known {
            if record.size == candidate.size as i64 && same_mtime(record.mtime, candidate.mtime) {
                report.unchanged += 1;
                continue;
            }
        }

        let bytes = match std::fs::read(&candidate.path) {
            Ok(bytes) => bytes,
            Err(_) => {
                report.unreadable += 1;
                continue;
            }
        };

        // Slow path: the metadata moved but the bytes may not have. A rewritten
        // or touched file is common enough to be worth the hash.
        let digest = blake3::hash(&bytes).to_hex().to_string();
        if let Some(record) = &known {
            if record.blake3 == digest {
                store.touch_file(&path, candidate.mtime, candidate.size)?;
                report.unchanged += 1;
                continue;
            }
        }

        let text = match extract(&candidate.path, &bytes) {
            Extraction::Text(text) => text,
            Extraction::Skipped(reason) => {
                match reason {
                    SkipReason::Binary => report.skipped_binary += 1,
                    SkipReason::UnsupportedFormat => report.skipped_unsupported += 1,
                }
                // A file that used to be indexable and no longer is must not
                // linger in the index as a stale answer.
                if known.is_some() {
                    store.delete_file(&path)?;
                }
                continue;
            }
        };

        let chunks = chunk_text(&text, chunk_options);
        let mut vectors = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(options.batch_size.max(1)) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let embedded = backend
                .embed(&texts)
                .with_context(|| format!("embedding {path} failed"))?;
            if embedded.len() != texts.len() {
                anyhow::bail!(
                    "the backend returned {} embeddings for {} chunks of {path}",
                    embedded.len(),
                    texts.len()
                );
            }
            vectors.extend(embedded);
        }

        store.replace_file(
            &path,
            candidate.mtime,
            candidate.size,
            &digest,
            &chunks,
            &vectors,
        )?;
        report.indexed += 1;
        report.chunks_written += chunks.len();
    }

    report.removed = store.delete_missing(&seen)?;
    Ok(report)
}

/// Filesystem timestamps survive a round trip through SQLite as `REAL`, but not
/// always bit-for-bit; a millisecond of slack avoids spurious re-indexing.
fn same_mtime(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-3
}
