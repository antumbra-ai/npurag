//! A deterministic, dependency-free backend used by tests and `--mock`.
//!
//! Embeddings are built by hashing the *distinct* words of a text into fixed
//! buckets and normalising the result, so texts that share vocabulary land
//! closer together under cosine similarity. That is enough structure to exercise
//! indexing, ranking and RAG end to end without an NPU or a running server.
//!
//! Distinct, not counted: weighting by term frequency let a document that
//! repeats a common word like "the" outrank one that actually contains the
//! query's terms. Real embedding models do not reward repetition either, so
//! counting would have made the mock both less useful and less faithful.

use std::collections::HashSet;

use anyhow::Result;

use super::{Backend, Message, Role};

pub struct MockBackend {
    dim: usize,
    reranker: bool,
}

impl MockBackend {
    /// Small enough to keep test fixtures readable, large enough that unrelated
    /// words rarely collide into the same bucket.
    pub const DEFAULT_DIM: usize = 64;

    pub fn new() -> Self {
        Self::with_dim(Self::DEFAULT_DIM)
    }

    /// # Panics
    /// Panics if `dim` is zero — a zero-dimensional embedding is never valid.
    pub fn with_dim(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be non-zero");
        Self {
            dim,
            reranker: false,
        }
    }

    /// Pretend this backend has a reranking model.
    ///
    /// Off by default, so that the mock behaves like the common case — a
    /// backend with an embedding model and a chat model and nothing else — and
    /// reranking is only exercised where a test asks for it.
    pub fn with_reranker(mut self) -> Self {
        self.reranker = true;
        self
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let mut seen = HashSet::new();
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let word = word.to_lowercase();
            if !seen.insert(word.clone()) {
                continue;
            }
            let hash = fnv1a64(&word);
            v[(hash % self.dim as u64) as usize] += 1.0;
        }

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            // A text with no words still needs a unit vector, or cosine
            // similarity against it would be NaN.
            let uniform = 1.0 / (self.dim as f32).sqrt();
            return vec![uniform; self.dim];
        }
        for x in &mut v {
            *x /= norm;
        }
        v
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MockBackend {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn chat(&self, messages: &[Message], model: Option<&str>) -> Result<String> {
        let question = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("(no user message)");
        Ok(format!(
            "[mock:{}] {question}",
            model.unwrap_or("mock-chat")
        ))
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Option<Vec<f32>>> {
        if !self.reranker {
            return Ok(None);
        }
        let wanted = words(query);
        Ok(Some(
            documents
                .iter()
                .map(|document| {
                    if wanted.is_empty() {
                        return 0.0;
                    }
                    // The share of the query's words the excerpt actually
                    // contains, on the nought-to-ten scale a real reranker is
                    // asked for. Crude, but it moves the ranking in the same
                    // direction a real one would, and it does so identically on
                    // every machine.
                    let found = words(document).intersection(&wanted).count();
                    10.0 * found as f32 / wanted.len() as f32
                })
                .collect(),
        ))
    }

    fn health(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        match self.reranker {
            true => format!("mock (dim {}, with reranker)", self.dim),
            false => format!("mock (dim {})", self.dim),
        }
    }
}

/// The distinct lowercase words of a text.
fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn fnv1a64(s: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
