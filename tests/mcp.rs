//! The MCP server: framing, both protocol eras, and the tools themselves.

use std::path::{Path, PathBuf};

use npurag::backend::MockBackend;
use npurag::chunk::ChunkOptions;
use npurag::index::{index_dir, IndexOptions};
use npurag::mcp::{serve, McpServer, LEGACY_VERSIONS, MODERN_VERSION};
use npurag::search::SearchOptions;
use npurag::store::Store;
use serde_json::{json, Value};
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

/// One request in, the parsed response out.
fn ask_server(store: &Store, request: Value) -> Value {
    let backend = MockBackend::new();
    let server = McpServer {
        store,
        backend: &backend,
        defaults: SearchOptions::default(),
    };
    let line = server
        .handle(&request.to_string())
        .expect("a request gets a response");
    serde_json::from_str(&line).expect("the response is JSON")
}

/// A request in the stateless era: version and capabilities travel with it.
fn modern(id: u32, method: &str, mut params: Value) -> Value {
    params["_meta"] = json!({
        "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
        "io.modelcontextprotocol/clientInfo": { "name": "test", "version": "1" },
        "io.modelcontextprotocol/clientCapabilities": {},
    });
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

/// A request in the handshake era: no metadata at all.
fn legacy(id: u32, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

fn tool_call(store: &Store, era: fn(u32, &str, Value) -> Value, name: &str, args: Value) -> Value {
    ask_server(
        store,
        era(9, "tools/call", json!({ "name": name, "arguments": args })),
    )
}

// --- framing --------------------------------------------------------------

#[test]
fn a_notification_is_never_answered() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let backend = MockBackend::new();
    let server = McpServer {
        store: &store,
        backend: &backend,
        defaults: SearchOptions::default(),
    };

    // No id, so no reply — the specification is explicit about this, and a
    // stray response would desynchronise the client.
    assert!(server
        .handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string())
        .is_none());
    assert!(server
        .handle(
            &json!({"jsonrpc": "2.0", "method": "notifications/cancelled",
                        "params": {"requestId": 1}})
            .to_string()
        )
        .is_none());
    assert!(server.handle("   ").is_none());
}

#[test]
fn a_malformed_line_is_reported_rather_than_ignored() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let backend = MockBackend::new();
    let server = McpServer {
        store: &store,
        backend: &backend,
        defaults: SearchOptions::default(),
    };

    let response: Value =
        serde_json::from_str(&server.handle("{not json").expect("answers")).unwrap();
    assert_eq!(response["error"]["code"], -32700);
}

#[test]
fn every_response_is_one_line_of_json() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let backend = MockBackend::new();
    let server = McpServer {
        store: &store,
        backend: &backend,
        defaults: SearchOptions::default(),
    };

    let input = format!(
        "{}\n{}\n",
        modern(1, "tools/list", json!({})),
        modern(
            2,
            "tools/call",
            json!({"name": "search", "arguments": {"query": "borgmatic"}})
        )
    );
    let mut output = Vec::new();
    serve(&server, input.as_bytes(), &mut output).expect("serves");

    let text = String::from_utf8(output).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "one response per request: {text}");
    for line in lines {
        // Embedded newlines would break the framing even though the JSON
        // itself would still be valid.
        assert!(!line.contains('\n'));
        let _: Value = serde_json::from_str(line).expect("each line parses on its own");
    }
}

// --- the stateless era ----------------------------------------------------

#[test]
fn discovery_reports_the_versions_and_the_tools_capability() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = ask_server(&store, modern(1, "server/discover", json!({})));

    let result = &response["result"];
    assert_eq!(result["resultType"], "complete");
    assert!(result["supportedVersions"]
        .as_array()
        .unwrap()
        .contains(&json!(MODERN_VERSION)));
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(
        result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "npurag"
    );
}

#[test]
fn discovery_answers_even_without_client_metadata() {
    // It is the probe a client uses to find out what this server is; refusing
    // it on a technicality would make npurag look like a legacy server.
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = ask_server(&store, legacy(1, "server/discover", json!({})));
    assert_eq!(response["result"]["resultType"], "complete");
}

#[test]
fn an_unknown_protocol_version_is_refused_with_the_ones_we_speak() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let mut request = modern(1, "tools/list", json!({}));
    request["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = json!("1900-01-01");

    let response = ask_server(&store, request);
    assert_eq!(response["error"]["code"], -32022);
    assert_eq!(response["error"]["data"]["requested"], "1900-01-01");
    assert!(response["error"]["data"]["supported"]
        .as_array()
        .unwrap()
        .contains(&json!(MODERN_VERSION)));
}

#[test]
fn a_modern_request_without_client_capabilities_is_malformed() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let mut request = modern(1, "tools/list", json!({}));
    request["params"]["_meta"]
        .as_object_mut()
        .unwrap()
        .remove("io.modelcontextprotocol/clientCapabilities");

    let response = ask_server(&store, request);
    assert_eq!(response["error"]["code"], -32602);
}

#[test]
fn results_carry_the_result_type_the_stateless_era_requires() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(&store, modern, "search", json!({"query": "borgmatic"}));
    assert_eq!(response["result"]["resultType"], "complete");
}

// --- the handshake era ----------------------------------------------------

#[test]
fn an_older_client_still_gets_its_handshake() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = ask_server(
        &store,
        legacy(
            1,
            "initialize",
            json!({
                "protocolVersion": LEGACY_VERSIONS[1],
                "capabilities": {},
                "clientInfo": { "name": "old-client", "version": "1" }
            }),
        ),
    );

    let result = &response["result"];
    assert_eq!(
        result["protocolVersion"], LEGACY_VERSIONS[1],
        "the revision the client asked for should be echoed back"
    );
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(result["serverInfo"]["name"], "npurag");
}

#[test]
fn an_unknown_handshake_version_falls_back_to_one_we_speak() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = ask_server(
        &store,
        legacy(1, "initialize", json!({ "protocolVersion": "1999-01-01" })),
    );
    assert_eq!(response["result"]["protocolVersion"], LEGACY_VERSIONS[0]);
}

#[test]
fn an_older_client_can_list_and_call_tools_without_any_metadata() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let listed = ask_server(&store, legacy(2, "tools/list", json!({})));
    let names: Vec<&str> = listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["search", "ask", "status"]);

    let called = tool_call(&store, legacy, "search", json!({"query": "borgmatic"}));
    assert_eq!(called["result"]["isError"], false);
}

// --- the tools ------------------------------------------------------------

#[test]
fn the_tools_declare_schemas_a_client_can_validate_against() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let listed = ask_server(&store, modern(1, "tools/list", json!({})));

    for tool in listed["result"]["tools"].as_array().unwrap() {
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object", "{tool}");
        assert!(tool["description"].as_str().is_some_and(|d| !d.is_empty()));
    }
    let search = &listed["result"]["tools"][0];
    assert_eq!(search["inputSchema"]["required"], json!(["query"]));
}

#[test]
fn search_returns_the_passages_as_text_and_as_data() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(
        &store,
        modern,
        "search",
        json!({ "query": "borgmatic vault drive", "k": 1 }),
    );

    let result = &response["result"];
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("backup.md"), "{text}");

    let hits = result["structuredContent"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("backup.md"));
    assert!(
        result["structuredContent"]["origin"]["chunks"]
            .as_i64()
            .unwrap()
            > 0
    );
}

#[test]
fn search_arguments_reach_the_retrieval_they_name() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(
        &store,
        modern,
        "search",
        json!({ "query": "FV-2026-00431", "mode": "lexical", "path": "*.md" }),
    );

    let hits = response["result"]["structuredContent"]["hits"]
        .as_array()
        .unwrap();
    assert_eq!(hits.len(), 1, "only the invoice note says that literally");
    assert!(hits[0]["path"].as_str().unwrap().ends_with("invoice.md"));
    assert!(
        hits[0]["scores"]["lexical"].is_number() && hits[0]["scores"]["vector"].is_null(),
        "mode=lexical should not have run the vector half: {hits:?}"
    );
}

#[test]
fn ask_answers_with_the_sources_it_used() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(
        &store,
        modern,
        "ask",
        json!({ "question": "how is the backup configured?", "k": 2 }),
    );

    let result = &response["result"];
    assert_eq!(result["isError"], false);
    assert!(result["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("Sources:"));
    assert!(!result["structuredContent"]["sources"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn status_describes_which_index_is_being_served() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(&store, modern, "status", json!({}));

    let result = &response["result"];
    assert_eq!(result["structuredContent"]["files"], 2);
    assert!(result["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("2 file(s)"));
}

// --- failure ---------------------------------------------------------------

#[test]
fn an_unknown_tool_is_a_protocol_error() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = tool_call(&store, modern, "delete_everything", json!({}));
    assert_eq!(response["error"]["code"], -32602);
    assert!(response["error"]["message"]
        .as_str()
        .unwrap()
        .contains("delete_everything"));
}

#[test]
fn a_bad_argument_comes_back_as_something_the_model_can_fix() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    // Not a JSON-RPC error: a model that gets `isError` with an explanation can
    // correct itself and call again, which a protocol error does not invite.
    let missing = tool_call(&store, modern, "search", json!({}));
    assert_eq!(missing["result"]["isError"], true);
    assert!(missing["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("query"));

    let nonsense = tool_call(
        &store,
        modern,
        "search",
        json!({ "query": "x", "mode": "telepathy" }),
    );
    assert_eq!(nonsense["result"]["isError"], true);
    assert!(nonsense["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("telepathy"));
}

#[test]
fn an_unknown_method_is_reported_as_such() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let response = ask_server(&store, modern(1, "resources/list", json!({})));
    assert_eq!(response["error"]["code"], -32601);
}

#[test]
fn the_request_id_always_comes_back_unchanged() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    // Clients are free to use strings, and correlation breaks silently if the
    // id is coerced on the way through.
    let response = ask_server(
        &store,
        json!({ "jsonrpc": "2.0", "id": "abc-1", "method": "tools/list", "params": {} }),
    );
    assert_eq!(response["id"], "abc-1");
    assert_eq!(response["jsonrpc"], "2.0");
}
