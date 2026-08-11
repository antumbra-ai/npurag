//! The HTTP endpoint: routing, arguments, authorisation, and one real socket.

use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;

use npurag::backend::MockBackend;
use npurag::chunk::ChunkOptions;
use npurag::http::{serve, HttpOptions, Reply, Service};
use npurag::index::{index_dir, IndexOptions};
use npurag::search::SearchOptions;
use npurag::store::Store;
use serde_json::Value;
use tempfile::TempDir;

fn tree() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();
    std::fs::write(
        root.join("backup.md"),
        "The nightly backup runs borgmatic against the vault drive.\n",
    )
    .unwrap();
    std::fs::write(
        root.join("invoice.md"),
        "The importer deadline, and invoice FV-2026-00431 agreed with the client.\n",
    )
    .unwrap();
    (tmp, root)
}

fn indexed(root: &Path) -> Store {
    let mut store = Store::open_in_memory().unwrap();
    index_dir(
        &mut store,
        &MockBackend::new(),
        root,
        &Default::default(),
        &ChunkOptions::default(),
        &IndexOptions::default(),
    )
    .expect("indexes");
    store
}

/// Route a request against a service with no token.
fn request(store: &Store, method: &str, url: &str, body: &str) -> (u16, Value) {
    with_token(store, None, method, url, body, None)
}

fn with_token(
    store: &Store,
    token: Option<&str>,
    method: &str,
    url: &str,
    body: &str,
    auth: Option<&str>,
) -> (u16, Value) {
    let backend = MockBackend::new();
    let service = Service {
        store,
        backend: &backend,
        defaults: SearchOptions::default(),
        token: token.map(str::to_string),
    };
    let Reply { status, body, .. } = service.handle(method, url, body, auth);
    (
        status,
        serde_json::from_str(&body).expect("every reply is JSON"),
    )
}

/// The raw reply, for routes whose body is not JSON.
fn raw(store: &Store, method: &str, url: &str, body: &str) -> Reply {
    let backend = MockBackend::new();
    let service = Service {
        store,
        backend: &backend,
        defaults: SearchOptions::default(),
        token: None,
    };
    service.handle(method, url, body, None)
}

// --- routing ---------------------------------------------------------------

#[test]
fn health_answers_without_a_credential() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    // A monitor should be able to see the endpoint is alive; it learns nothing
    // from this that the open port did not already tell it.
    let (status, body) = with_token(&store, Some("secret"), "GET", "/health", "", None);
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
}

#[test]
fn an_unknown_route_says_which_ones_exist() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(&store, "GET", "/everything", "");
    assert_eq!(status, 404);
    assert!(body["error"].as_str().unwrap().contains("/search"));
}

#[test]
fn the_wrong_method_is_refused_rather_than_misread() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, _) = request(&store, "DELETE", "/search", "");
    assert_eq!(status, 405);
}

#[test]
fn status_reports_what_the_index_covers() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(&store, "GET", "/status", "");
    assert_eq!(status, 200);
    assert_eq!(body["files"], 2);
    assert!(body["chunks"].as_i64().unwrap() >= 2);
}

// --- authorisation ---------------------------------------------------------

#[test]
fn without_a_token_configured_anything_may_ask() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, _) = request(&store, "GET", "/search?q=borgmatic", "");
    assert_eq!(status, 200);
}

#[test]
fn a_configured_token_is_required_and_must_match() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let url = "/search?q=borgmatic";

    let (status, _) = with_token(&store, Some("secret"), "GET", url, "", None);
    assert_eq!(status, 401, "no header at all");

    let (status, _) = with_token(&store, Some("secret"), "GET", url, "", Some("Bearer wrong"));
    assert_eq!(status, 401);

    // A prefix of the right token must not pass either.
    let (status, _) = with_token(&store, Some("secret"), "GET", url, "", Some("Bearer sec"));
    assert_eq!(status, 401);

    let (status, _) = with_token(&store, Some("secret"), "GET", url, "", Some("Basic secret"));
    assert_eq!(status, 401, "the scheme matters");

    let (status, _) = with_token(
        &store,
        Some("secret"),
        "GET",
        url,
        "",
        Some("Bearer secret"),
    );
    assert_eq!(status, 200);
}

#[test]
fn the_index_is_never_exposed_to_the_network_without_a_token() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let backend = MockBackend::new();
    let service = Service {
        store: &store,
        backend: &backend,
        defaults: SearchOptions::default(),
        token: None,
    };

    // Refused before the socket is ever opened, so this cannot bind by accident.
    let err = serve(
        &service,
        &HttpOptions {
            bind: "0.0.0.0:0".to_string(),
            token: None,
        },
        |_| panic!("must not start listening"),
    )
    .expect_err("binding the world without a token must fail");
    assert!(err.to_string().contains("token"), "{err}");

    // Loopback without a token is the ordinary case and stays allowed; this one
    // gets as far as binding, which is why the address is a free port.
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let store = Store::open_in_memory().unwrap();
        let backend = MockBackend::new();
        let service = Service {
            store: &store,
            backend: &backend,
            defaults: SearchOptions::default(),
            token: None,
        };
        let _ = serve(
            &service,
            &HttpOptions {
                bind: "127.0.0.1:0".to_string(),
                token: None,
            },
            |address| tx.send(address.to_string()).unwrap(),
        );
    });
    let address = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("loopback should be allowed to listen");
    assert!(address.starts_with("127.0.0.1:"));
}

// --- arguments -------------------------------------------------------------

#[test]
fn a_query_string_is_decoded_before_it_is_searched_for() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    // Spaces arrive as %20 or as +, and neither should reach the retriever.
    let (status, body) = request(&store, "GET", "/search?q=borgmatic%20vault+drive&k=1", "");
    assert_eq!(status, 200);
    assert_eq!(body["query"], "borgmatic vault drive");
    assert_eq!(body["hits"].as_array().unwrap().len(), 1);
}

#[test]
fn a_json_body_carries_the_same_arguments() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(
        &store,
        "POST",
        "/search",
        r#"{"query": "FV-2026-00431", "mode": "lexical", "k": 3}"#,
    );

    assert_eq!(status, 200);
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("invoice.md"));
    assert!(
        hits[0]["scores"]["vector"].is_null(),
        "mode=lexical should not have run the vector half"
    );
}

#[test]
fn the_body_wins_over_the_query_string() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (_, body) = request(&store, "POST", "/search?q=kayaks", r#"{"q": "borgmatic"}"#);
    assert_eq!(body["query"], "borgmatic");
}

#[test]
fn k_is_accepted_as_a_number_and_as_a_string() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let (_, from_query) = request(&store, "GET", "/search?q=the&k=1", "");
    assert_eq!(from_query["hits"].as_array().unwrap().len(), 1);

    let (_, from_body) = request(&store, "POST", "/search", r#"{"q": "the", "k": 1}"#);
    assert_eq!(from_body["hits"].as_array().unwrap().len(), 1);
}

#[test]
fn a_missing_query_is_a_bad_request_not_an_empty_result() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(&store, "GET", "/search", "");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains('q'));
}

#[test]
fn nonsense_arguments_are_named_in_the_error() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let (status, body) = request(&store, "GET", "/search?q=x&mode=telepathy", "");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("telepathy"));

    let (status, body) = request(&store, "GET", "/search?q=x&k=lots", "");
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains('k'));

    let (status, _) = request(&store, "POST", "/search", "not json at all");
    assert_eq!(status, 400);
}

#[test]
fn ask_answers_with_the_sources_it_used() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(
        &store,
        "POST",
        "/ask",
        r#"{"question": "how is the backup configured?", "k": 2}"#,
    );

    assert_eq!(status, 200);
    assert!(!body["answer"].as_str().unwrap().is_empty());
    assert!(!body["sources"].as_array().unwrap().is_empty());
    assert!(body["origin"]["chunks"].as_i64().unwrap() > 0);
}

// --- the OpenAI-compatible surface -----------------------------------------

#[test]
fn the_model_list_advertises_one_model() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(&store, "GET", "/v1/models", "");

    assert_eq!(status, 200);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "npurag");
    assert_eq!(body["data"][0]["object"], "model");
}

#[test]
fn a_chat_completion_comes_back_in_the_shape_clients_expect() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let (status, body) = request(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"model": "npurag", "messages": [{"role": "user", "content": "how is the backup configured?"}]}"#,
    );

    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["usage"]["total_tokens"].is_number());
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));

    // The citations have to survive a shape that has no field for them.
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("Sources:"), "{content}");
    assert!(!body["npurag"]["sources"].as_array().unwrap().is_empty());
}

#[test]
fn the_last_user_message_is_what_gets_retrieved_on() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    // The mock's chat echoes the prompt it was given, so the excerpts that
    // reached the model are visible in the answer.
    let (_, body) = request(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"messages": [
             {"role": "system", "content": "be brief"},
             {"role": "user", "content": "tell me about kayaks"},
             {"role": "assistant", "content": "nothing here about those"},
             {"role": "user", "content": "borgmatic vault drive"}
           ]}"#,
    );

    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("borgmatic"), "{content}");
    let sources = body["npurag"]["sources"].as_array().unwrap();
    assert!(sources[0]["path"].as_str().unwrap().ends_with("backup.md"));
}

#[test]
fn naming_another_model_passes_it_to_the_backend() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    // The mock reports which chat model it was called with, so the override is
    // observable rather than merely plausible.
    let (_, overridden) = request(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"model": "gemma3:4b", "messages": [{"role": "user", "content": "backups?"}]}"#,
    );
    let content = overridden["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("[mock:gemma3:4b]"), "{content}");
    assert_eq!(overridden["model"], "gemma3:4b");

    // Our own advertised name is not an override; it means "whatever you have".
    let (_, default) = request(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"model": "npurag", "messages": [{"role": "user", "content": "backups?"}]}"#,
    );
    assert!(default["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .contains("[mock:mock-chat]"));
}

#[test]
fn streaming_answers_in_frames_a_client_can_parse() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let reply = raw(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"stream": true, "messages": [{"role": "user", "content": "backups?"}]}"#,
    );

    assert_eq!(reply.status, 200);
    assert_eq!(reply.content_type, "text/event-stream");
    assert!(reply.body.ends_with("data: [DONE]\n\n"), "{}", reply.body);

    let frames: Vec<&str> = reply
        .body
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .filter(|f| *f != "[DONE]")
        .collect();
    assert_eq!(frames.len(), 3, "role, content, then the finish frame");

    let first: Value = serde_json::from_str(frames[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    let middle: Value = serde_json::from_str(frames[1]).unwrap();
    assert!(middle["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap()
        .contains("Sources:"));

    let last: Value = serde_json::from_str(frames[2]).unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[test]
fn a_conversation_with_nothing_to_answer_is_refused_in_their_error_shape() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let (status, body) = request(&store, "POST", "/v1/chat/completions", r#"{"model": "x"}"#);
    assert_eq!(status, 400);
    // OpenAI clients read error.message, not a bare string.
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("messages"));

    let (status, body) = request(
        &store,
        "POST",
        "/v1/chat/completions",
        r#"{"messages": [{"role": "system", "content": "be brief"}]}"#,
    );
    assert_eq!(status, 400);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("user message"));
}

#[test]
fn the_token_doubles_as_an_api_key() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    // An OpenAI client sends its key exactly the way this endpoint wants it,
    // so configuring one is the same act as configuring the other.
    let body = r#"{"messages": [{"role": "user", "content": "backups?"}]}"#;

    let (status, _) = with_token(
        &store,
        Some("sk-local"),
        "POST",
        "/v1/chat/completions",
        body,
        None,
    );
    assert_eq!(status, 401);

    let (status, _) = with_token(
        &store,
        Some("sk-local"),
        "POST",
        "/v1/chat/completions",
        body,
        Some("Bearer sk-local"),
    );
    assert_eq!(status, 200);
}

// --- over a real socket ----------------------------------------------------

#[test]
fn it_answers_an_actual_http_request() {
    let (tmp, root) = tree();
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        // Keep the fixture alive for as long as the server runs.
        let _tmp = tmp;
        let store = indexed(&root);
        let backend = MockBackend::new();
        let service = Service {
            store: &store,
            backend: &backend,
            defaults: SearchOptions::default(),
            token: Some("s3cret".to_string()),
        };
        let _ = serve(
            &service,
            &HttpOptions {
                bind: "127.0.0.1:0".to_string(),
                token: Some("s3cret".to_string()),
            },
            |address| tx.send(address.to_string()).unwrap(),
        );
    });

    let address = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the server should report the port it took");

    let refused = ureq::get(&format!("http://{address}/search?q=borgmatic")).call();
    assert_eq!(
        refused.err().and_then(|e| match e {
            ureq::Error::StatusCode(code) => Some(code),
            _ => None,
        }),
        Some(401),
        "the token is enforced on the wire, not only in the router"
    );

    let mut response = ureq::get(&format!("http://{address}/search?q=borgmatic&k=1"))
        .header("Authorization", "Bearer s3cret")
        .call()
        .expect("an authorised request succeeds");
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/json"
    );
    let body: Value = response.body_mut().read_json().expect("a JSON body");
    assert!(body["hits"][0]["path"]
        .as_str()
        .unwrap()
        .ends_with("backup.md"));
}
