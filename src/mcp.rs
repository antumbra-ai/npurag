//! An MCP server, so an assistant can search the index directly.
//!
//! Model Context Protocol over stdio: JSON-RPC 2.0, one message per line, in on
//! stdin and out on stdout. No socket, no port, nothing listening — the client
//! launches npurag as a child process and talks to it down a pipe. For a tool
//! whose whole point is that your files never leave the machine, that is the
//! transport to want.
//!
//! ## Two eras on one process
//!
//! Revision `2026-07-28` made the protocol stateless: there is no `initialize`
//! handshake any more, every request carries its own protocol version and
//! client capabilities in `_meta`, results are tagged with `resultType`, and a
//! server is discovered through `server/discover`. Clients built against the
//! older revisions still open with `initialize` and send no `_meta` at all.
//!
//! Both are answered here. The rule is the one the specification gives for a
//! dual-era server: a request carrying a modern protocol version in `_meta` is
//! served statelessly under the new revision, and an `initialize` request
//! selects the old semantics. Everything below that split — the tools, the
//! index, the answers — is identical either way.

use std::io::{BufRead, Write};

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::ask::{ask, AskOptions};
use crate::backend::Backend;
use crate::rerank::RerankMode;
use crate::search::{search, search_payload, SearchMode, SearchOptions};
use crate::store::Store;

/// The stateless revision this server implements.
pub const MODERN_VERSION: &str = "2026-07-28";

/// Handshake-based revisions still answered, newest first. A client that opens
/// with `initialize` gets whichever of these it asked for.
pub const LEGACY_VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

/// JSON-RPC and MCP error codes used here.
const INVALID_PARAMS: i64 = -32602;
const METHOD_NOT_FOUND: i64 = -32601;
const PARSE_ERROR: i64 = -32700;
const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

const INSTRUCTIONS: &str = "\
Searches a directory that has been indexed on this machine, and answers questions from it. \
`search` returns the passages themselves and is the one to reach for when you want to read \
the user's own files; `ask` additionally has the local model write an answer from them. \
Everything stays on the machine.";

/// Everything a request needs to be answered.
pub struct McpServer<'a> {
    pub store: &'a Store,
    pub backend: &'a dyn Backend,
    /// Retrieval settings from the config, before a tool call overrides them.
    pub defaults: SearchOptions,
}

impl McpServer<'_> {
    /// Answer one line from the client.
    ///
    /// `None` means there is nothing to send back: JSON-RPC notifications get
    /// no reply, and a blank line is not a message.
    pub fn handle(&self, line: &str) -> Option<String> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        let request: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            // No id can be recovered from a message that would not parse.
            Err(e) => return Some(error_response(Value::Null, PARSE_ERROR, &e.to_string())),
        };

        let id = request.get("id").cloned();
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let params = request.get("params").cloned().unwrap_or(Value::Null);

        // A notification has no id, and the specification is explicit that it
        // MUST NOT be answered. `notifications/cancelled` lands here too: work
        // is done synchronously, so by the time a cancellation could be read
        // the request it refers to has already been answered.
        let id = id?;

        Some(match self.dispatch(method, &params) {
            Ok(result) => result_response(id, result),
            Err(McpError {
                code,
                message,
                data,
            }) => error_response_with_data(id, code, &message, data),
        })
    }

    fn dispatch(&self, method: &str, params: &Value) -> Result<Value, McpError> {
        // `initialize` is how a client from the handshake era opens, and it is
        // the one method that must not be checked for modern metadata.
        if method == "initialize" {
            return Ok(self.initialize(params));
        }
        check_protocol(params, method)?;

        match method {
            "server/discover" => Ok(self.discover()),
            "tools/list" => Ok(json!({ "tools": tool_definitions() })),
            "tools/call" => self.call_tool(params),
            "ping" => Ok(json!({})),
            other => Err(McpError::new(
                METHOD_NOT_FOUND,
                format!("unknown method: {other}"),
            )),
        }
    }

    fn discover(&self) -> Value {
        let mut versions = vec![Value::from(MODERN_VERSION)];
        versions.extend(LEGACY_VERSIONS.iter().map(|v| Value::from(*v)));
        json!({
            "supportedVersions": versions,
            "capabilities": { "tools": {} },
            "instructions": INSTRUCTIONS,
        })
    }

    /// The handshake the older revisions open with.
    fn initialize(&self, params: &Value) -> Value {
        // Echo the revision the client asked for when it is one we answer, so
        // it does not have to downgrade; otherwise name our newest legacy one.
        let asked = params.get("protocolVersion").and_then(Value::as_str);
        let version = asked
            .filter(|v| LEGACY_VERSIONS.contains(v) || *v == MODERN_VERSION)
            .unwrap_or(LEGACY_VERSIONS[0]);
        json!({
            "protocolVersion": version,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": server_info(),
            "instructions": INSTRUCTIONS,
        })
    }

    fn call_tool(&self, params: &Value) -> Result<Value, McpError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| McpError::new(INVALID_PARAMS, "tools/call needs a tool name"))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        // Unknown tool is a protocol error; anything that goes wrong *inside* a
        // tool comes back as a result with isError, which is the form a model
        // can read and correct.
        let outcome = match name {
            "search" => self.tool_search(&arguments),
            "ask" => self.tool_ask(&arguments),
            "status" => self.tool_status(),
            other => {
                return Err(McpError::new(
                    INVALID_PARAMS,
                    format!("unknown tool: {other}"),
                ))
            }
        };

        Ok(match outcome {
            Ok((text, structured)) => json!({
                "content": [{ "type": "text", "text": text }],
                "structuredContent": structured,
                "isError": false,
            }),
            Err(e) => json!({
                "content": [{ "type": "text", "text": format!("{e:#}") }],
                "isError": true,
            }),
        })
    }

    /// Retrieval settings for one call: the configured ones, with whatever the
    /// arguments named applied on top. Mirrors what the command line does.
    fn options_from(&self, arguments: &Value) -> Result<SearchOptions> {
        let mut options = self.defaults.clone();
        if let Some(k) = arguments.get("k").and_then(Value::as_u64) {
            options.top_k = k as usize;
        }
        options.path = arguments
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(mode) = arguments.get("mode").and_then(Value::as_str) {
            options.mode = match mode {
                "hybrid" => SearchMode::Hybrid,
                "vector" => SearchMode::Vector,
                "lexical" => SearchMode::Lexical,
                other => anyhow::bail!("unknown mode `{other}`; use hybrid, vector or lexical"),
            };
        }
        if let Some(rerank) = arguments.get("rerank").and_then(Value::as_str) {
            options.rerank.mode = match rerank {
                "auto" => RerankMode::Auto,
                "off" => RerankMode::Off,
                "endpoint" => RerankMode::Endpoint,
                "llm" => RerankMode::Llm,
                other => {
                    anyhow::bail!("unknown rerank mode `{other}`; use auto, off, endpoint or llm")
                }
            };
        }
        Ok(options)
    }

    fn tool_search(&self, arguments: &Value) -> Result<(String, Value)> {
        let query = required_str(arguments, "query")?;
        let options = self.options_from(arguments)?;
        let hits = search(self.store, self.backend, &query, &options)?;

        let mut text = String::new();
        if hits.is_empty() {
            text.push_str("No passage in the index matched.");
        } else {
            for (rank, hit) in hits.iter().enumerate() {
                text.push_str(&format!(
                    "[{}] {}#{}  (score {:.3})\n{}\n\n",
                    rank + 1,
                    hit.path,
                    hit.ord,
                    hit.score,
                    hit.text.trim()
                ));
            }
        }
        Ok((text, search_payload(self.store, &query, &hits)?))
    }

    fn tool_ask(&self, arguments: &Value) -> Result<(String, Value)> {
        let question = required_str(arguments, "question")?;
        let options = AskOptions {
            search: self.options_from(arguments)?,
            model: arguments
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..Default::default()
        };
        let answer = ask(self.store, self.backend, &question, &options)?;

        let mut text = answer.answer.trim().to_string();
        if !answer.sources.is_empty() {
            text.push_str("\n\nSources:\n");
            for source in &answer.sources {
                text.push_str(&format!(
                    "  [{}] {}#{}\n",
                    source.marker, source.path, source.ord
                ));
            }
        }
        Ok((text, serde_json::to_value(&answer)?))
    }

    fn tool_status(&self) -> Result<(String, Value)> {
        let stats = self.store.stats()?;
        let text = format!(
            "Index of {}: {} file(s), {} chunk(s), built with {} on backend {}. Backend is {}.",
            stats.root_path.as_deref().unwrap_or("an unknown root"),
            stats.files,
            stats.chunks,
            stats.embed_model.as_deref().unwrap_or("an unknown model"),
            stats.backend.as_deref().unwrap_or("unknown"),
            if self.backend.health() {
                "reachable"
            } else {
                "unreachable, so only mode=lexical will work"
            }
        );
        Ok((
            text,
            json!({
                "root": stats.root_path,
                "files": stats.files,
                "chunks": stats.chunks,
                "embed_model": stats.embed_model,
                "backend": stats.backend,
                "healthy": self.backend.health(),
            }),
        ))
    }
}

/// Read requests until the client closes the pipe.
///
/// Nothing but MCP messages may go to stdout, which is why npurag's usual
/// `emit` has no business here and why any diagnostics belong on stderr.
pub fn serve(server: &McpServer, input: impl BufRead, mut output: impl Write) -> Result<()> {
    for line in input.lines() {
        let line = line?;
        if let Some(response) = server.handle(&line) {
            writeln!(output, "{response}")?;
            output.flush()?;
        }
    }
    Ok(())
}

/// Reject a request whose protocol version we do not answer.
///
/// A request with no version in `_meta` comes from the handshake era and is
/// served as it always was. `server/discover` is answered either way: it is the
/// probe a client uses to find out what this server is, and refusing to answer
/// it on a technicality would defeat the one job it has.
fn check_protocol(params: &Value, method: &str) -> Result<(), McpError> {
    let meta = params.get("_meta");
    let Some(version) = meta
        .and_then(|m| m.get(META_PROTOCOL_VERSION))
        .and_then(Value::as_str)
    else {
        return Ok(());
    };

    if version != MODERN_VERSION && !LEGACY_VERSIONS.contains(&version) {
        let mut supported = vec![Value::from(MODERN_VERSION)];
        supported.extend(LEGACY_VERSIONS.iter().map(|v| Value::from(*v)));
        return Err(
            McpError::new(UNSUPPORTED_PROTOCOL_VERSION, "unsupported protocol version")
                .with_data(json!({ "supported": supported, "requested": version })),
        );
    }

    // Required on every modern request, and cheap to check — except on the
    // discovery probe, where being strict would only hide the server.
    if method != "server/discover"
        && version == MODERN_VERSION
        && meta.and_then(|m| m.get(META_CLIENT_CAPABILITIES)).is_none()
    {
        return Err(McpError::new(
            INVALID_PARAMS,
            format!("{META_CLIENT_CAPABILITIES} is required on every request"),
        ));
    }
    Ok(())
}

fn tool_definitions() -> Value {
    let retrieval = |extra: Value| {
        let mut properties = Map::new();
        if let Value::Object(map) = extra {
            properties.extend(map);
        }
        properties.insert(
            "k".to_string(),
            json!({
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "How many passages to use. Defaults to 8."
            }),
        );
        properties.insert(
            "path".to_string(),
            json!({
                "type": "string",
                "description": "Only consider files whose path matches this glob, e.g. '*.md'."
            }),
        );
        properties.insert(
            "mode".to_string(),
            json!({
                "type": "string",
                "enum": ["hybrid", "vector", "lexical"],
                "description": "hybrid (default) ranks by meaning and by wording together; \
                                vector is meaning only; lexical is BM25 only, which is best for \
                                an exact string and is the only mode that works with no \
                                inference server running."
            }),
        );
        properties.insert(
            "rerank".to_string(),
            json!({
                "type": "string",
                "enum": ["auto", "off", "endpoint", "llm"],
                "description": "How to rerank the shortlist. auto (default) uses a reranking \
                                model if the backend has one; llm scores the passages with the \
                                chat model, which is slower."
            }),
        );
        properties
    };

    let mut search_properties = retrieval(json!({}));
    search_properties.insert(
        "query".to_string(),
        json!({
            "type": "string",
            "description": "What to look for. A full sentence retrieves better than a keyword, \
                            but an exact identifier works too — the lexical half of the search \
                            is there for precisely that."
        }),
    );

    let mut ask_properties = retrieval(json!({
        "model": {
            "type": "string",
            "description": "Chat model to use instead of the configured one."
        }
    }));
    ask_properties.insert(
        "question".to_string(),
        json!({
            "type": "string",
            "description": "The question to answer from the indexed files."
        }),
    );

    json!([
        {
            "name": "search",
            "title": "Search the indexed files",
            "description": "Find passages in the user's own indexed files, by meaning and by \
                            wording. Returns the passages themselves with the file each came \
                            from, so you can read and cite them.",
            "inputSchema": {
                "type": "object",
                "properties": search_properties,
                "required": ["query"],
                "additionalProperties": false
            }
        },
        {
            "name": "ask",
            "title": "Answer from the indexed files",
            "description": "Retrieve the relevant passages and have the local model write an \
                            answer grounded in them, with the excerpts it used. Prefer `search` \
                            when you want to read the sources yourself.",
            "inputSchema": {
                "type": "object",
                "properties": ask_properties,
                "required": ["question"],
                "additionalProperties": false
            }
        },
        {
            "name": "status",
            "title": "Describe the index",
            "description": "What this index covers: the directory, how many files and passages, \
                            which embedding model built it, and whether the inference backend is \
                            reachable.",
            "inputSchema": { "type": "object", "additionalProperties": false }
        }
    ])
}

fn required_str(arguments: &Value, key: &str) -> Result<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("`{key}` is required and must be a non-empty string"))
}

fn server_info() -> Value {
    json!({ "name": "npurag", "version": env!("CARGO_PKG_VERSION") })
}

struct McpError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl McpError {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Wrap a result, tagging it the way the stateless revision expects.
///
/// `resultType` and the server identity are additive: a client from the
/// handshake era, which knows neither field, ignores them, and one from the
/// stateless era requires both. Sending them always is what lets a single
/// response shape serve both.
fn result_response(id: Value, mut result: Value) -> String {
    if let Value::Object(map) = &mut result {
        map.insert("resultType".to_string(), Value::from("complete"));
        map.insert(
            "_meta".to_string(),
            json!({ META_SERVER_INFO: server_info() }),
        );
    }
    json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
}

fn error_response(id: Value, code: i64, message: &str) -> String {
    error_response_with_data(id, code, message, None)
}

fn error_response_with_data(id: Value, code: i64, message: &str, data: Option<Value>) -> String {
    let mut error = json!({ "code": code, "message": message });
    if let (Value::Object(map), Some(data)) = (&mut error, data) {
        map.insert("data".to_string(), data);
    }
    json!({ "jsonrpc": "2.0", "id": id, "error": error }).to_string()
}
