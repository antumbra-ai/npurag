//! Reranking: a second, closer look at the shortlist.
//!
//! Retrieval has to be cheap, because it scores the whole index. Reranking is
//! allowed to be expensive, because it only ever sees the few dozen chunks that
//! survived — which is what lets it read the query and a passage *together*
//! instead of comparing two vectors that were computed without knowing about
//! each other.
//!
//! Two ways to get that second opinion, because the hardware targets differ:
//! a dedicated reranking model behind a `/rerank` endpoint (what OpenVINO Model
//! Server offers), or the chat model already loaded for `ask`. The first is
//! faster and better; the second works on any backend that can hold a
//! conversation, which includes every FastFlowLM setup.

use std::fmt::Write as _;

use anyhow::{anyhow, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::backend::{Backend, Message};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum RerankMode {
    /// Rerank when the backend has a reranking model configured, and quietly
    /// skip it when it has not. The default: reranking should improve an
    /// answer, never be the reason a command fails.
    #[default]
    Auto,
    /// Rank by retrieval score alone.
    Off,
    /// Insist on the backend's `/rerank` endpoint, and fail if there is none.
    Endpoint,
    /// Score the excerpts with the chat model. Works everywhere `ask` works,
    /// and costs a full generation on top of the search.
    Llm,
}

impl std::fmt::Display for RerankMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RerankMode::Auto => "auto",
            RerankMode::Off => "off",
            RerankMode::Endpoint => "endpoint",
            RerankMode::Llm => "llm",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankOptions {
    pub mode: RerankMode,
    /// How many retrieved chunks are handed to the reranker.
    pub top: usize,
    /// How much of each chunk is shown to it.
    pub max_chars: usize,
}

impl Default for RerankOptions {
    fn default() -> Self {
        Self {
            mode: RerankMode::default(),
            // Enough room for the fused ranking to be genuinely rearranged,
            // few enough that an LLM can hold them all in one prompt.
            top: 20,
            // A chunk is around 400 tokens; the opening of one is plenty to
            // judge relevance, and a shorter prompt is a faster one.
            max_chars: 1200,
        }
    }
}

impl RerankOptions {
    pub fn is_off(&self) -> bool {
        self.mode == RerankMode::Off
    }
}

/// Score `documents` against `query`, best first when sorted descending.
///
/// `Ok(None)` means the ranking should stay as retrieval left it — either
/// nothing was asked for, or the backend has no reranker, or its answer could
/// not be read. Only [`RerankMode::Endpoint`], which is an explicit demand,
/// turns a missing reranker into an error.
pub fn rerank(
    backend: &dyn Backend,
    query: &str,
    documents: &[String],
    options: &RerankOptions,
) -> Result<Option<Vec<f32>>> {
    // One document cannot be reordered, and zero is not a question.
    if documents.len() < 2 {
        return Ok(None);
    }
    let clipped: Vec<String> = documents
        .iter()
        .map(|doc| clip(doc, options.max_chars))
        .collect();

    let scores = match options.mode {
        RerankMode::Off => None,
        RerankMode::Auto => backend.rerank(query, &clipped)?,
        RerankMode::Endpoint => Some(backend.rerank(query, &clipped)?.ok_or_else(|| {
            anyhow!(
                "--rerank endpoint was asked for, but {} has no reranking model; \
                 set rerank_model for this backend, or use --rerank llm",
                backend.describe()
            )
        })?),
        RerankMode::Llm => llm_rerank(backend, query, &clipped)?,
    };

    match scores {
        Some(scores) if scores.len() != documents.len() => Err(anyhow!(
            "the reranker returned {} scores for {} excerpts",
            scores.len(),
            documents.len()
        )),
        other => Ok(other),
    }
}

const LLM_SYSTEM_PROMPT: &str = "\
You rate how well each numbered excerpt helps answer a question. \
Reply with one line per excerpt, in the form `N: S`, where N is the excerpt number \
and S is a relevance score from 0 (irrelevant) to 10 (answers the question directly). \
Rate every excerpt. Write nothing else.";

/// Ask the chat model to score the shortlist in a single call.
///
/// One call, not one per excerpt: a local model on an NPU answers in seconds,
/// and twenty round trips would cost more than the search they are refining.
fn llm_rerank(
    backend: &dyn Backend,
    query: &str,
    documents: &[String],
) -> Result<Option<Vec<f32>>> {
    let mut prompt = format!("Question: {query}\n\nExcerpts:\n\n");
    for (i, document) in documents.iter().enumerate() {
        let _ = writeln!(prompt, "[{}] {}\n", i + 1, document.trim());
    }
    let _ = write!(
        prompt,
        "Rate all {} excerpts, one per line.",
        documents.len()
    );

    let reply = backend.chat(
        &[Message::system(LLM_SYSTEM_PROMPT), Message::user(prompt)],
        None,
    )?;
    Ok(parse_scores(&reply, documents.len()))
}

/// Read `N: score` lines out of a model's reply.
///
/// Small models drift from any output format, so this reads the two numbers on
/// a line rather than matching an exact shape, and gives up — leaving the
/// retrieval order alone — when fewer than half the excerpts came back rated.
/// An excerpt the model skipped scores zero; a stable sort then keeps those in
/// the order retrieval put them in.
pub fn parse_scores(reply: &str, expected: usize) -> Option<Vec<f32>> {
    if expected == 0 {
        return None;
    }
    let mut scores: Vec<Option<f32>> = vec![None; expected];
    for line in reply.lines() {
        if let Some((index, score)) = parse_line(line) {
            if (1..=expected).contains(&index) {
                scores[index - 1] = Some(score);
            }
        }
    }
    let rated = scores.iter().filter(|s| s.is_some()).count();
    if rated * 2 < expected {
        return None;
    }
    Some(scores.into_iter().map(|s| s.unwrap_or(0.0)).collect())
}

/// The first two numbers on a line: the excerpt number, then its score.
///
/// Two things keep this from reading ratings out of ordinary prose, which a
/// model that ignored the format will happily produce: the line has to *begin*
/// with the excerpt number, and the score has to fall on the scale that was
/// asked for. A sentence like `[2] the invoice FV-2026-00431 was paid` gets no
/// further than its 2026.
fn parse_line(line: &str) -> Option<(usize, f32)> {
    let line = line.trim_start_matches(|c: char| !c.is_alphanumeric());
    if !line.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let mut numbers = line
        .split(|c: char| !c.is_ascii_digit() && c != '.')
        .map(|token| token.trim_matches('.'))
        .filter(|token| !token.is_empty());
    let index = numbers.next()?.parse::<usize>().ok()?;
    let score = numbers.next()?.parse::<f32>().ok()?;
    if !(0.0..=10.0).contains(&score) {
        return None;
    }
    Some((index, score))
}

/// Shorten a chunk to `max_chars`, on a character boundary.
fn clip(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    trimmed.chars().take(max_chars).collect()
}
