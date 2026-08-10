//! Ranking stored chunks against a query by cosine similarity.
//!
//! Brute force over every vector: exact, simple, and fast enough at the scale a
//! personal index reaches. Vectors are streamed and only the surviving chunks
//! have their text read back, so the memory cost is the result set, not the
//! corpus. An approximate index is an M6 concern, not a correctness one.

use anyhow::{Context, Result};
use globset::{Glob, GlobMatcher};
use serde::Serialize;

use crate::backend::Backend;
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,
    /// Restrict results to paths matching this glob.
    pub path: Option<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 8,
            path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub path: String,
    pub ord: i64,
    pub score: f32,
    pub text: String,
    pub n_tokens: i64,
}

/// Glob matching for `--path`.
///
/// A pattern without a separator is matched against the file name too, so
/// `--path '*.md'` does the obvious thing instead of silently matching nothing.
pub struct PathFilter {
    matcher: GlobMatcher,
    match_file_name: bool,
}

impl PathFilter {
    pub fn new(pattern: &str) -> Result<Self> {
        let matcher = Glob::new(pattern)
            .with_context(|| format!("invalid --path glob `{pattern}`"))?
            .compile_matcher();
        Ok(Self {
            matcher,
            match_file_name: !pattern.contains('/'),
        })
    }

    pub fn matches(&self, path: &str) -> bool {
        if self.matcher.is_match(path) {
            return true;
        }
        if self.match_file_name {
            if let Some(name) = std::path::Path::new(path).file_name() {
                return self.matcher.is_match(name);
            }
        }
        false
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    #[cfg(feature = "simd")]
    {
        use simsimd::SpatialSimilarity;
        // simsimd reports a distance; a similarity is one minus that. It
        // returns None only for mismatched lengths, which is handled above.
        if let Some(distance) = f32::cosine(a, b) {
            return 1.0 - distance as f32;
        }
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        // An all-zero vector has no direction, so it is similar to nothing.
        return 0.0;
    }
    dot / denominator
}

pub fn search(
    store: &Store,
    backend: &dyn Backend,
    query: &str,
    options: &SearchOptions,
) -> Result<Vec<Hit>> {
    if options.top_k == 0 {
        return Ok(Vec::new());
    }
    let filter = options.path.as_deref().map(PathFilter::new).transpose()?;

    let query_vector = backend
        .embed(&[query.to_string()])?
        .into_iter()
        .next()
        .context("the backend returned no embedding for the query")?;

    let mut scored: Vec<(f32, i64, String)> = Vec::new();
    store.scan_vectors(|id, path, vector| {
        if filter.as_ref().is_some_and(|f| !f.matches(path)) {
            return;
        }
        scored.push((cosine(&query_vector, vector), id, path.to_string()));
    })?;

    // Ties are broken by chunk id so that equal scores rank deterministically.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.truncate(options.top_k);

    let ids: Vec<i64> = scored.iter().map(|(_, id, _)| *id).collect();
    let texts = store.chunk_texts(&ids)?;

    Ok(scored
        .into_iter()
        .filter_map(|(score, id, path)| {
            texts.get(&id).map(|chunk| Hit {
                path,
                ord: chunk.ord,
                score,
                text: chunk.text.clone(),
                n_tokens: chunk.n_tokens,
            })
        })
        .collect())
}
