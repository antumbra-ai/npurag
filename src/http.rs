//! A small HTTP endpoint, for callers that are not an assistant.
//!
//! MCP covers the case where a language model does the asking. This covers the
//! other one: a script, a cron job, a service, anything that would rather send
//! a request than spawn a process. The bodies are the same JSON `--json`
//! prints, so nothing new has to be learned or kept in step.
//!
//! `tiny_http` rather than axum, because §6's choice of a blocking HTTP client
//! applies just as well to the server: no async runtime, no tokio in the
//! dependency tree, and a binary that still compiles in half a minute.
//!
//! ## What this exposes
//!
//! Everything the index holds — which is to say, whatever directory the user
//! pointed npurag at, quite possibly their home. That shapes two decisions
//! here. It listens on loopback unless told otherwise, and it *refuses* to bind
//! anywhere else without a token, rather than printing a warning nobody reads.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::ask::{ask, AskOptions};
use crate::backend::Backend;
use crate::rerank::RerankMode;
use crate::search::{search, search_payload, SearchMode, SearchOptions};
use crate::store::Store;

pub const DEFAULT_BIND: &str = "127.0.0.1:8787";

#[derive(Debug, Clone)]
pub struct HttpOptions {
    pub bind: String,
    /// Required as `Authorization: Bearer …` on every route but `/health`.
    pub token: Option<String>,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND.to_string(),
            token: None,
        }
    }
}

/// A JSON reply, ready to go out.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub status: u16,
    pub body: String,
}

impl Reply {
    fn ok(value: Value) -> Self {
        Self {
            status: 200,
            body: format!("{value}"),
        }
    }

    fn error(status: u16, message: impl std::fmt::Display) -> Self {
        Self {
            status,
            body: json!({ "error": message.to_string() }).to_string(),
        }
    }
}

/// Everything a request needs, minus the socket.
///
/// Routing is separated from listening so the whole surface can be tested
/// without binding a port, which is also what keeps the test suite free of
/// timing and cleanup problems.
pub struct Service<'a> {
    pub store: &'a Store,
    pub backend: &'a dyn Backend,
    pub defaults: SearchOptions,
    pub token: Option<String>,
}

impl Service<'_> {
    /// Route one request. `auth` is the `Authorization` header, if any.
    pub fn handle(&self, method: &str, url: &str, body: &str, auth: Option<&str>) -> Reply {
        let (path, query) = url.split_once('?').unwrap_or((url, ""));

        // Left open so a monitor can see the endpoint is alive without being
        // handed a credential. It reveals nothing the port itself does not.
        if path == "/health" {
            return Reply::ok(json!({
                "status": "ok",
                "version": env!("CARGO_PKG_VERSION"),
            }));
        }

        if !self.authorized(auth) {
            return Reply::error(401, "missing or wrong bearer token");
        }

        let arguments = match parameters(method, query, body) {
            Ok(arguments) => arguments,
            Err(e) => return Reply::error(400, e),
        };

        match (method, path) {
            ("GET", "/status") => self.status(),
            ("GET" | "POST", "/search") => self.search(&arguments),
            ("GET" | "POST", "/ask") => self.ask(&arguments),
            ("GET" | "POST", _) => Reply::error(
                404,
                format!("no route {path}; try /search, /ask, /status or /health"),
            ),
            _ => Reply::error(405, format!("{method} is not allowed on {path}")),
        }
    }

    fn authorized(&self, auth: Option<&str>) -> bool {
        let Some(expected) = &self.token else {
            return true;
        };
        auth.and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .is_some_and(|given| constant_time_eq(given, expected))
    }

    fn search(&self, arguments: &Value) -> Reply {
        let query = match required(arguments, &["q", "query"]) {
            Ok(query) => query,
            Err(e) => return Reply::error(400, e),
        };
        let options = match self.options_from(arguments) {
            Ok(options) => options,
            Err(e) => return Reply::error(400, e),
        };
        match search(self.store, self.backend, &query, &options)
            .and_then(|hits| search_payload(self.store, &query, &hits))
        {
            Ok(payload) => Reply::ok(payload),
            Err(e) => Reply::error(500, format!("{e:#}")),
        }
    }

    fn ask(&self, arguments: &Value) -> Reply {
        let question = match required(arguments, &["q", "question"]) {
            Ok(question) => question,
            Err(e) => return Reply::error(400, e),
        };
        let options = match self.options_from(arguments) {
            Ok(search) => AskOptions {
                search,
                model: string_of(arguments, "model"),
                ..Default::default()
            },
            Err(e) => return Reply::error(400, e),
        };
        match ask(self.store, self.backend, &question, &options)
            .and_then(|answer| Ok(serde_json::to_value(answer)?))
        {
            Ok(payload) => Reply::ok(payload),
            Err(e) => Reply::error(500, format!("{e:#}")),
        }
    }

    fn status(&self) -> Reply {
        match self.store.stats() {
            Ok(stats) => Reply::ok(json!({
                "root": stats.root_path,
                "files": stats.files,
                "chunks": stats.chunks,
                "embed_model": stats.embed_model,
                "backend": stats.backend,
                "healthy": self.backend.health(),
                "version": env!("CARGO_PKG_VERSION"),
            })),
            Err(e) => Reply::error(500, format!("{e:#}")),
        }
    }

    /// Retrieval settings for one request, mirroring the command line.
    fn options_from(&self, arguments: &Value) -> Result<SearchOptions> {
        let mut options = self.defaults.clone();
        if let Some(k) = arguments.get("k") {
            options.top_k = as_usize(k).context("k must be a positive whole number")?;
        }
        options.path = string_of(arguments, "path");
        if let Some(mode) = string_of(arguments, "mode") {
            options.mode = match mode.as_str() {
                "hybrid" => SearchMode::Hybrid,
                "vector" => SearchMode::Vector,
                "lexical" => SearchMode::Lexical,
                other => anyhow::bail!("unknown mode `{other}`; use hybrid, vector or lexical"),
            };
        }
        if let Some(rerank) = string_of(arguments, "rerank") {
            options.rerank.mode = match rerank.as_str() {
                "auto" => RerankMode::Auto,
                "off" => RerankMode::Off,
                "endpoint" => RerankMode::Endpoint,
                "llm" => RerankMode::Llm,
                other => {
                    anyhow::bail!("unknown rerank `{other}`; use auto, off, endpoint or llm")
                }
            };
        }
        Ok(options)
    }
}

/// Listen, and answer until interrupted.
///
/// Requests are served one at a time. A personal index is not a web service,
/// and a second connection waiting on the first is a better trade than either a
/// thread pool holding a database handle each or an async runtime pulled in for
/// a loopback socket.
pub fn serve(service: &Service, options: &HttpOptions, ready: impl FnOnce(&str)) -> Result<()> {
    guard_exposure(options)?;

    let server = tiny_http::Server::http(&options.bind)
        .map_err(|e| anyhow::anyhow!("could not listen on {}: {e}", options.bind))?;
    let address = server
        .server_addr()
        .to_ip()
        .map(|a| a.to_string())
        .unwrap_or_else(|| options.bind.clone());
    ready(&address);

    for mut request in server.incoming_requests() {
        let mut body = String::new();
        // A body that is not UTF-8 is not JSON either; let the route report it.
        let _ = request.as_reader().read_to_string(&mut body);

        let auth = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Authorization"))
            .map(|h| h.value.as_str().to_string());

        let reply = service.handle(
            request.method().as_str(),
            request.url(),
            &body,
            auth.as_deref(),
        );

        let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
            .expect("a constant header always parses");
        let response = tiny_http::Response::from_string(reply.body)
            .with_status_code(reply.status)
            .with_header(header);
        // A client that hung up mid-answer is not this server's problem.
        let _ = request.respond(response);
    }
    Ok(())
}

/// Refuse to serve the index to the network without a token.
///
/// Not a warning: the whole index is readable through this port, and a warning
/// on a terminal nobody is watching is not a control.
fn guard_exposure(options: &HttpOptions) -> Result<()> {
    if options.token.is_some() || is_loopback(&options.bind) {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to serve {} without a token: every indexed file is readable through this \
         endpoint. Set NPURAG_TOKEN (or pass --token), or bind {DEFAULT_BIND} instead",
        options.bind
    )
}

fn is_loopback(bind: &str) -> bool {
    let host = match bind.rsplit_once(':') {
        Some((host, _)) => host,
        None => bind,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

/// Collect the arguments, from the query string or from a JSON body.
fn parameters(method: &str, query: &str, body: &str) -> Result<Value> {
    let mut arguments = query_parameters(query);
    if method == "POST" && !body.trim().is_empty() {
        let parsed: Value = serde_json::from_str(body).context("the request body is not JSON")?;
        let Value::Object(fields) = parsed else {
            anyhow::bail!("the request body must be a JSON object");
        };
        // The body wins: it is the more deliberate of the two.
        if let Value::Object(map) = &mut arguments {
            map.extend(fields);
        }
    }
    Ok(arguments)
}

fn query_parameters(query: &str) -> Value {
    let mut map = serde_json::Map::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(percent_decode(key), Value::from(percent_decode(value)));
    }
    Value::Object(map)
}

/// Undo the encoding a query string arrives in.
///
/// Written out rather than pulled in: it is twenty lines, and a URL decoder is
/// not a dependency worth carrying into a static binary.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                match u8::from_str_radix(&text[i + 1..i + 3], 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    // A stray percent sign is a literal one, not an error.
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The first of `keys` that carries a non-empty string.
fn required(arguments: &Value, keys: &[&str]) -> Result<String> {
    keys.iter()
        .find_map(|key| string_of(arguments, key))
        .ok_or_else(|| anyhow::anyhow!("`{}` is required", keys[0]))
}

fn string_of(arguments: &Value, key: &str) -> Option<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Numbers arrive as numbers in a JSON body and as strings in a query string.
fn as_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Number(n) => n.as_u64().map(|n| n as usize),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Compare without leaking, through timing, how much of the token was right.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |difference, (x, y)| difference | (x ^ y))
        == 0
}
