//! Retrieval-augmented answering: retrieve, ground, cite.

use anyhow::Result;
use serde::Serialize;

use crate::backend::{Backend, Message};
use crate::chunk::estimate_tokens;
use crate::search::{search, Hit, SearchOptions};
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct AskOptions {
    pub top_k: usize,
    pub path: Option<String>,
    /// Override the configured chat model for this one question.
    pub model: Option<String>,
    /// Ceiling on how much retrieved text is pasted into the prompt. Retrieval
    /// can return more than a small local model will accept, and silently
    /// overflowing its context is worse than answering from fewer excerpts.
    pub max_context_tokens: usize,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            top_k: 8,
            path: None,
            model: None,
            max_context_tokens: 3000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Source {
    /// The bracketed number the answer cites this excerpt by.
    pub marker: usize,
    pub path: String,
    pub ord: i64,
    pub score: f32,
}

/// Which index answered, so a reply can be traced to a collection and not just
/// to a file path. One index covers one root directory, built by one embedding
/// model on one backend — all three matter when several indexes exist.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Origin {
    /// The directory this index was built from.
    pub root: Option<String>,
    pub backend: Option<String>,
    pub embed_model: Option<String>,
    /// How much of the index the question was matched against.
    pub files: i64,
    pub chunks: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub question: String,
    pub answer: String,
    /// Where the answer came from, at the level of the whole collection.
    pub origin: Origin,
    /// Where it came from, excerpt by excerpt.
    pub sources: Vec<Source>,
}

pub fn origin_of(store: &Store) -> Result<Origin> {
    let stats = store.stats()?;
    Ok(Origin {
        root: stats.root_path,
        backend: stats.backend,
        embed_model: stats.embed_model,
        files: stats.files,
        chunks: stats.chunks,
    })
}

const SYSTEM_PROMPT: &str = "\
You answer questions about the user's own files, using only the excerpts you are given. \
If the excerpts do not contain the answer, say so plainly instead of guessing. \
Cite the excerpts you rely on by their bracketed numbers, like [1] or [2][3]. \
Answer in the same language as the question.";

/// Select the excerpts that fit the budget, in ranked order.
///
/// The best hit is always kept even if it alone exceeds the budget: answering
/// from one long excerpt beats answering from none.
pub fn select_context(hits: &[Hit], max_context_tokens: usize) -> Vec<&Hit> {
    let mut selected = Vec::new();
    let mut used = 0usize;
    for hit in hits {
        let cost = estimate_tokens(&hit.text);
        if !selected.is_empty() && used + cost > max_context_tokens {
            break;
        }
        used += cost;
        selected.push(hit);
    }
    selected
}

pub fn build_prompt(question: &str, hits: &[&Hit]) -> Vec<Message> {
    let mut context = String::from("Excerpts from the indexed files:\n\n");
    for (i, hit) in hits.iter().enumerate() {
        context.push_str(&format!(
            "[{}] {}\n{}\n\n",
            i + 1,
            hit.path,
            hit.text.trim()
        ));
    }
    context.push_str(&format!("Question: {question}"));

    vec![Message::system(SYSTEM_PROMPT), Message::user(context)]
}

pub fn ask(
    store: &Store,
    backend: &dyn Backend,
    question: &str,
    options: &AskOptions,
) -> Result<Answer> {
    let hits = search(
        store,
        backend,
        question,
        &SearchOptions {
            top_k: options.top_k,
            path: options.path.clone(),
        },
    )?;

    if hits.is_empty() {
        return Ok(Answer {
            question: question.to_string(),
            answer: "The index has nothing to answer this from.".to_string(),
            origin: origin_of(store)?,
            sources: Vec::new(),
        });
    }

    let selected = select_context(&hits, options.max_context_tokens);
    let messages = build_prompt(question, &selected);
    let answer = backend.chat(&messages, options.model.as_deref())?;

    Ok(Answer {
        question: question.to_string(),
        answer,
        origin: origin_of(store)?,
        sources: selected
            .iter()
            .enumerate()
            .map(|(i, hit)| Source {
                marker: i + 1,
                path: hit.path.clone(),
                ord: hit.ord,
                score: hit.score,
            })
            .collect(),
    })
}
