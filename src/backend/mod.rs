//! The backend abstraction.
//!
//! npurag never talks to hardware. It speaks OpenAI-compatible HTTP, so support
//! for a new accelerator is a matter of pointing [`OpenAiBackend`] at a different
//! `base_url` with different model names — see [`crate::config`].

mod mock;
mod openai;

pub use mock::MockBackend;
pub use openai::OpenAiBackend;

use serde::{Deserialize, Serialize};

/// Who authored a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single chat message, serialised exactly as the OpenAI chat API expects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Everything npurag needs from an inference server.
pub trait Backend {
    /// Embed a batch of texts. The returned vectors are in the same order as the
    /// input and all share the same dimensionality.
    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Run a chat completion, optionally overriding the configured chat model.
    fn chat(&self, messages: &[Message], model: Option<&str>) -> anyhow::Result<String>;

    /// Score how well each document answers `query`, higher being better.
    ///
    /// `Ok(None)` means this backend has no reranking model — a normal state,
    /// not a failure: reranking is an optional refinement, and a backend
    /// without one still searches and answers. Defaulted so that adding a
    /// backend never means writing a reranker first.
    fn rerank(&self, query: &str, documents: &[String]) -> anyhow::Result<Option<Vec<f32>>> {
        let _ = (query, documents);
        Ok(None)
    }

    /// Cheap reachability probe used by `npurag status`. Never returns an error:
    /// an unreachable backend is a normal, reportable state, not a failure.
    fn health(&self) -> bool;

    /// Human-readable name of this backend, for diagnostics.
    fn describe(&self) -> String;
}
