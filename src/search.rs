//! Ranking stored chunks against a query.
//!
//! Two retrievers run over the same index and are fused into one list: nearest
//! neighbours by cosine similarity (meaning) and BM25 over the full-text index
//! (words). Both are brute force over the corpus — exact, simple, and fast
//! enough at the scale a personal index reaches. Vectors are streamed and only
//! the chunks that survive ranking have their text read back, so the memory cost
//! is the result set, not the corpus.

use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::ValueEnum;
use globset::{Glob, GlobMatcher};
use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::lexical;
use crate::rerank::{rerank, RerankOptions};
use crate::store::Store;

/// Which retrievers to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Both, fused. Prose is found by meaning, identifiers by their spelling.
    #[default]
    Hybrid,
    /// Embeddings only.
    Vector,
    /// BM25 only. Needs no backend call at all, which also makes it the mode
    /// that still works when the inference server is down.
    Lexical,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SearchMode::Hybrid => "hybrid",
            SearchMode::Vector => "vector",
            SearchMode::Lexical => "lexical",
        })
    }
}

/// How the two ranked lists are combined.
#[derive(Debug, Clone, PartialEq)]
pub struct Fusion {
    /// The rank offset in reciprocal rank fusion. The standard 60 keeps any
    /// single retriever's top hit from dominating: the difference between rank
    /// 1 and rank 2 stays small next to the agreement of both retrievers.
    pub rrf_k: f32,
    pub vector_weight: f32,
    pub lexical_weight: f32,
    /// How many candidates each retriever contributes. Zero means: work it out
    /// from `top_k`.
    pub candidates: usize,
}

impl Default for Fusion {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            vector_weight: 1.0,
            lexical_weight: 1.0,
            candidates: 0,
        }
    }
}

impl Fusion {
    /// The candidate pool each retriever fills before fusion.
    ///
    /// Wider than the result set on purpose: a chunk that both retrievers rank
    /// modestly should be able to beat one that only a single retriever loves,
    /// and it cannot do that if it was cut before fusion ever saw it.
    pub fn pool(&self, top_k: usize) -> usize {
        if self.candidates > 0 {
            return self.candidates.max(top_k);
        }
        (top_k * 5).max(50)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchOptions {
    pub top_k: usize,
    /// Restrict results to paths matching this glob.
    pub path: Option<String>,
    pub mode: SearchMode,
    pub fusion: Fusion,
    pub rerank: RerankOptions,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 8,
            path: None,
            mode: SearchMode::default(),
            fusion: Fusion::default(),
            rerank: RerankOptions::default(),
        }
    }
}

/// A chunk and what one retriever thought of it.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub id: i64,
    pub path: String,
    pub score: f32,
}

/// Where a hit's final score came from, stage by stage.
///
/// Reported rather than hidden: when a search surprises you, the useful question
/// is which half of retrieval put the chunk there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Scores {
    /// Cosine similarity, when the vector retriever found this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<f32>,
    /// BM25, when the lexical retriever found it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical: Option<f32>,
    /// The fused rank score, when both retrievers ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused: Option<f32>,
    /// What the reranker made of it, when one ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Hit {
    pub path: String,
    pub ord: i64,
    /// The score this hit was finally ranked by: the reranker's if one ran, the
    /// fused score in hybrid mode, otherwise the single retriever's own.
    pub score: f32,
    pub text: String,
    pub n_tokens: i64,
    pub scores: Scores,
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

/// The `limit` nearest chunks to `query` by cosine similarity.
pub fn vector_candidates(
    store: &Store,
    backend: &dyn Backend,
    query: &str,
    limit: usize,
    filter: Option<&PathFilter>,
) -> Result<Vec<Scored>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let query_vector = backend
        .embed(&[query.to_string()])?
        .into_iter()
        .next()
        .context("the backend returned no embedding for the query")?;

    let mut scored: Vec<Scored> = Vec::new();
    store.scan_vectors(|id, path, vector| {
        if filter.is_some_and(|f| !f.matches(path)) {
            return;
        }
        scored.push(Scored {
            id,
            path: path.to_string(),
            score: cosine(&query_vector, vector),
        });
    })?;

    // Ties are broken by chunk id so that equal scores rank deterministically.
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    scored.truncate(limit);
    Ok(scored)
}

/// A chunk after fusion, carrying what each retriever contributed.
#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    pub id: i64,
    pub path: String,
    pub score: f32,
    pub scores: Scores,
}

/// Combine two ranked lists by reciprocal rank fusion.
///
/// Each list votes with `weight / (k + rank)`, and the votes are added up.
/// Ranks, not scores: a cosine similarity of 0.82 and a BM25 score of 7.4 are
/// numbers from unrelated scales, and any attempt to normalise them into
/// comparability needs corpus statistics that change with every document added.
/// Position in a ranking needs nothing, and is what the two retrievers actually
/// agree about.
pub fn reciprocal_rank_fusion(
    vector: &[Scored],
    lexical: &[Scored],
    fusion: &Fusion,
) -> Vec<Fused> {
    let mut merged: Vec<Fused> = Vec::with_capacity(vector.len() + lexical.len());
    let mut slot_of: HashMap<i64, usize> = HashMap::new();

    for (list, weight, is_vector) in [
        (vector, fusion.vector_weight, true),
        (lexical, fusion.lexical_weight, false),
    ] {
        for (rank, candidate) in list.iter().enumerate() {
            let slot = *slot_of.entry(candidate.id).or_insert_with(|| {
                merged.push(Fused {
                    id: candidate.id,
                    path: candidate.path.clone(),
                    score: 0.0,
                    scores: Scores::default(),
                });
                merged.len() - 1
            });
            let entry = &mut merged[slot];
            entry.score += weight / (fusion.rrf_k + rank as f32 + 1.0);
            if is_vector {
                entry.scores.vector = Some(candidate.score);
            } else {
                entry.scores.lexical = Some(candidate.score);
            }
        }
    }

    for entry in &mut merged {
        entry.scores.fused = Some(entry.score);
    }
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    merged
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
    let filter = filter.as_ref();

    // A reranker can only promote what retrieval handed it, so when one is
    // going to run it gets a longer shortlist than the caller asked to see.
    let shortlist = if options.rerank.is_off() {
        options.top_k
    } else {
        options.rerank.top.max(options.top_k)
    };
    let pool = match options.mode {
        SearchMode::Hybrid => options.fusion.pool(shortlist),
        _ => shortlist,
    };

    let vector = match options.mode {
        SearchMode::Lexical => Vec::new(),
        _ => vector_candidates(store, backend, query, pool, filter)?,
    };
    let lexical = match options.mode {
        SearchMode::Vector => Vec::new(),
        _ => lexical::candidates(store, query, pool, filter)?,
    };

    let mut ranked = match options.mode {
        SearchMode::Hybrid => reciprocal_rank_fusion(&vector, &lexical, &options.fusion),
        // A single retriever needs no fusion, and reporting its own score keeps
        // `--mode vector` reading exactly as a cosine search should.
        SearchMode::Vector => vector.into_iter().map(Fused::from_vector).collect(),
        SearchMode::Lexical => lexical.into_iter().map(Fused::from_lexical).collect(),
    };
    ranked.truncate(shortlist);

    let ids: Vec<i64> = ranked.iter().map(|entry| entry.id).collect();
    let texts = store.chunk_texts(&ids)?;
    let mut hits: Vec<Hit> = ranked
        .into_iter()
        .filter_map(|entry| {
            texts.get(&entry.id).map(|chunk| Hit {
                path: entry.path,
                ord: chunk.ord,
                score: entry.score,
                text: chunk.text.clone(),
                n_tokens: chunk.n_tokens,
                scores: entry.scores,
            })
        })
        .collect();

    let documents: Vec<String> = hits.iter().map(|hit| hit.text.clone()).collect();
    if let Some(scores) = rerank(backend, query, &documents, &options.rerank)? {
        for (hit, score) in hits.iter_mut().zip(scores) {
            hit.scores.rerank = Some(score);
            hit.score = score;
        }
        // A stable sort leaves excerpts the reranker scored equally in the
        // order retrieval put them in.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    hits.truncate(options.top_k);
    Ok(hits)
}

impl Fused {
    fn from_vector(candidate: Scored) -> Self {
        Self {
            id: candidate.id,
            path: candidate.path,
            score: candidate.score,
            scores: Scores {
                vector: Some(candidate.score),
                ..Scores::default()
            },
        }
    }

    fn from_lexical(candidate: Scored) -> Self {
        Self {
            id: candidate.id,
            path: candidate.path,
            score: candidate.score,
            scores: Scores {
                lexical: Some(candidate.score),
                ..Scores::default()
            },
        }
    }
}
