//! HTTP backend speaking the OpenAI API.
//!
//! One implementation serves every target: AMD FastFlowLM and Intel OpenVINO
//! Model Server differ only in `base_url` (including the version prefix, which is
//! `/v1` on some servers and `/v3` on others) and in model names.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::{Backend, Message};

pub struct OpenAiBackend {
    name: String,
    base_url: String,
    embed_model: String,
    chat_model: String,
    agent: ureq::Agent,
}

impl OpenAiBackend {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        embed_model: impl Into<String>,
        chat_model: impl Into<String>,
    ) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120)))
            .build();
        Self {
            name: name.into(),
            // Stored without a trailing slash so `endpoint` can join naively.
            base_url: base_url.into().trim_end_matches('/').to_string(),
            embed_model: embed_model.into(),
            chat_model: chat_model.into(),
            agent: config.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Join a path onto the configured base URL, preserving whatever version
    /// prefix the base URL already carries.
    pub fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

impl Backend for OpenAiBackend {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = self.endpoint("embeddings");
        let body = json!({ "model": self.embed_model, "input": texts });

        let mut response = self
            .agent
            .post(&url)
            .send_json(&body)
            .with_context(|| format!("POST {url} failed"))?;

        let parsed: EmbeddingResponse = response
            .body_mut()
            .read_json()
            .with_context(|| format!("could not parse the embeddings response from {url}"))?;

        if parsed.data.len() != texts.len() {
            return Err(anyhow!(
                "{url} returned {} embeddings for {} inputs",
                parsed.data.len(),
                texts.len()
            ));
        }

        // The API is allowed to answer out of order, so restore the input order
        // before the vectors are paired back up with their chunks.
        let mut items = parsed.data;
        items.sort_by_key(|item| item.index);

        let vectors: Vec<Vec<f32>> = items.into_iter().map(|item| item.embedding).collect();
        if let Some(first) = vectors.first() {
            if let Some(bad) = vectors.iter().find(|v| v.len() != first.len()) {
                return Err(anyhow!(
                    "{url} returned embeddings of mixed dimensions ({} and {})",
                    first.len(),
                    bad.len()
                ));
            }
        }
        Ok(vectors)
    }

    fn chat(&self, messages: &[Message], model: Option<&str>) -> Result<String> {
        let url = self.endpoint("chat/completions");
        let body = json!({
            "model": model.unwrap_or(&self.chat_model),
            "messages": messages,
        });

        let mut response = self
            .agent
            .post(&url)
            .send_json(&body)
            .with_context(|| format!("POST {url} failed"))?;

        let parsed: ChatResponse = response
            .body_mut()
            .read_json()
            .with_context(|| format!("could not parse the chat response from {url}"))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("{url} returned no choices"))
    }

    fn health(&self) -> bool {
        self.agent
            .get(&self.endpoint("models"))
            .call()
            .map(|response| response.status().is_success())
            .unwrap_or(false)
    }

    fn describe(&self) -> String {
        format!("{} ({})", self.name, self.base_url)
    }
}
