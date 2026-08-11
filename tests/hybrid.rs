//! Hybrid retrieval: the lexical index, fusion, and reranking.

use std::fs;
use std::path::{Path, PathBuf};

use npurag::backend::{Backend, Message, MockBackend};
use npurag::chunk::{Chunk, ChunkOptions};
use npurag::index::{index_dir, IndexOptions};
use npurag::lexical::{candidates, match_expression};
use npurag::rerank::{parse_scores, rerank, RerankMode, RerankOptions};
use npurag::search::{
    reciprocal_rank_fusion, search, Fusion, PathFilter, Scored, SearchMode, SearchOptions,
};
use npurag::store::{vec_to_blob, Store};
use tempfile::TempDir;

fn write(root: &Path, rel: &str, contents: &str) {
    fs::write(root.join(rel), contents).expect("writes");
}

/// Two notes whose wording overlaps as little as their meaning.
fn tree() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();
    write(
        &root,
        "backup.md",
        "The nightly backup runs borgmatic against the vault drive.\n",
    );
    write(
        &root,
        "invoice.md",
        "The importer deadline, and invoice FV-2026-00431 agreed with the client.\n",
    );
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

fn chunk(text: &str) -> Chunk {
    Chunk {
        ord: 0,
        text: text.to_string(),
        n_tokens: text.len().div_ceil(4),
    }
}

fn scored(id: i64, score: f32) -> Scored {
    Scored {
        id,
        path: format!("/tmp/{id}.md"),
        score,
    }
}

/// A backend with nothing behind it, standing in for a server that is down.
struct Offline;

impl Backend for Offline {
    fn embed(&self, _texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::bail!("no inference server")
    }

    fn chat(&self, _messages: &[Message], _model: Option<&str>) -> anyhow::Result<String> {
        anyhow::bail!("no inference server")
    }

    fn health(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        "offline".to_string()
    }
}

// --- turning a question into an FTS5 query -------------------------------

#[test]
fn a_query_full_of_operators_is_read_as_words() {
    let expression = match_expression("what about --path? NEAR AND (x*").expect("has terms");
    assert_eq!(
        expression,
        "\"what\" OR \"about\" OR \"path\" OR \"near\" OR \"and\" OR \"x\""
    );
}

#[test]
fn a_query_with_no_words_matches_nothing_rather_than_everything() {
    assert_eq!(match_expression("--- ?! ***"), None);
    assert_eq!(match_expression(""), None);
}

#[test]
fn a_repeated_word_is_asked_for_once() {
    assert_eq!(
        match_expression("backup Backup BACKUP"),
        Some("\"backup\"".to_string())
    );
}

#[test]
fn a_search_survives_punctuation_that_would_break_the_query_language() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    // A syntax error inside FTS5 would surface here as a failed command.
    let hits = search(
        &store,
        &MockBackend::new(),
        "\"unterminated AND (broken* NEAR/",
        &SearchOptions::default(),
    )
    .expect("searches");
    assert!(hits.len() <= 2);
}

// --- the lexical retriever -----------------------------------------------

#[test]
fn bm25_finds_a_literal_string_and_scores_it_above_zero() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let found = candidates(&store, "FV-2026-00431", 10, None).expect("searches");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].path.ends_with("invoice.md"));
    assert!(found[0].score > 0.0, "bm25 should be flipped: {found:?}");
}

#[test]
fn a_word_in_no_document_finds_nothing() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let found = candidates(&store, "kayaks", 10, None).expect("searches");
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn the_path_filter_applies_to_the_lexical_half_too() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    let filter = PathFilter::new("*.md").expect("valid glob");

    let found = candidates(&store, "the", 10, Some(&filter)).expect("searches");
    assert!(!found.is_empty());
    assert!(found.iter().all(|c| c.path.ends_with(".md")));

    let filter = PathFilter::new("*.rs").expect("valid glob");
    let found = candidates(&store, "the", 10, Some(&filter)).expect("searches");
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn lexical_search_needs_no_backend_at_all() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let hits = search(
        &store,
        &Offline,
        "borgmatic",
        &SearchOptions {
            mode: SearchMode::Lexical,
            ..Default::default()
        },
    )
    .expect("searches without an inference server");

    assert_eq!(hits.len(), 1);
    assert!(hits[0].path.ends_with("backup.md"));
    assert!(hits[0].scores.lexical.is_some());
    assert!(hits[0].scores.vector.is_none());
}

// --- fusion ---------------------------------------------------------------

#[test]
fn a_chunk_both_retrievers_found_beats_one_only_a_single_retriever_liked() {
    // Chunk 2 is nobody's favourite, but it is the only one on both lists.
    let vector = vec![scored(1, 0.9), scored(2, 0.7)];
    let lexical = vec![scored(3, 12.0), scored(2, 4.0)];

    let fused = reciprocal_rank_fusion(&vector, &lexical, &Fusion::default());

    assert_eq!(fused[0].id, 2, "{fused:?}");
    assert_eq!(fused[0].scores.vector, Some(0.7));
    assert_eq!(fused[0].scores.lexical, Some(4.0));
    assert_eq!(fused.len(), 3, "every candidate survives fusion");
}

#[test]
fn a_zero_weight_hands_the_ranking_to_the_other_retriever() {
    let vector = vec![scored(1, 0.9), scored(2, 0.8)];
    let lexical = vec![scored(2, 30.0), scored(1, 1.0)];

    let fused = reciprocal_rank_fusion(
        &vector,
        &lexical,
        &Fusion {
            lexical_weight: 0.0,
            ..Fusion::default()
        },
    );

    let order: Vec<i64> = fused.iter().map(|f| f.id).collect();
    assert_eq!(order, vec![1, 2]);
}

#[test]
fn fusion_scores_are_reported_alongside_the_parts_they_came_from() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let hits = search(
        &store,
        &MockBackend::new(),
        "borgmatic vault drive",
        &SearchOptions::default(),
    )
    .expect("searches");

    let top = hits.first().expect("a hit");
    assert!(top.path.ends_with("backup.md"));
    assert_eq!(top.scores.fused, Some(top.score));
    assert!(top.scores.vector.is_some() && top.scores.lexical.is_some());
}

#[test]
fn the_candidate_pool_is_never_narrower_than_the_result_set() {
    let fusion = Fusion::default();
    assert!(fusion.pool(8) >= 8);
    assert!(fusion.pool(200) >= 200);
    assert_eq!(
        Fusion {
            candidates: 3,
            ..Fusion::default()
        }
        .pool(10),
        10,
        "an explicit candidate count still has to cover top_k"
    );
}

// --- keeping the lexical index in step with the content -------------------

#[test]
fn rewriting_a_file_retires_its_old_wording() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    index_dir(
        &mut store,
        &MockBackend::new(),
        &root,
        &Default::default(),
        &ChunkOptions::default(),
        &IndexOptions::default(),
    )
    .unwrap();

    write(&root, "backup.md", "This note is about kayaks now.\n");
    index_dir(
        &mut store,
        &MockBackend::new(),
        &root,
        &Default::default(),
        &ChunkOptions::default(),
        &IndexOptions::default(),
    )
    .unwrap();

    let stale = candidates(&store, "borgmatic", 10, None).unwrap();
    assert!(
        stale.is_empty(),
        "the old text is still findable: {stale:?}"
    );
    let fresh = candidates(&store, "kayaks", 10, None).unwrap();
    assert_eq!(fresh.len(), 1, "{fresh:?}");
}

#[test]
fn pruning_a_deleted_file_takes_its_words_with_it() {
    let (_tmp, root) = tree();
    let mut store = indexed(&root);

    fs::remove_file(root.join("backup.md")).unwrap();
    store.prune_missing().expect("prunes");

    let stale = candidates(&store, "borgmatic", 10, None).unwrap();
    assert!(stale.is_empty(), "{stale:?}");
}

#[test]
fn a_reindex_from_scratch_leaves_no_lexical_residue() {
    let (_tmp, root) = tree();
    let store = indexed(&root);
    store.clear().expect("clears");

    let stale = candidates(&store, "borgmatic", 10, None).unwrap();
    assert!(stale.is_empty(), "{stale:?}");
}

// --- migrating an index written before the lexical half existed -----------

/// Build a database in the shape schema version 1 wrote, by hand.
fn legacy_index(path: &Path, text: &str) {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
         CREATE TABLE files (
           id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, mtime REAL NOT NULL,
           size INTEGER NOT NULL, blake3 TEXT NOT NULL, n_chunks INTEGER NOT NULL,
           indexed_at REAL NOT NULL);
         CREATE TABLE chunks (
           id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
           ord INTEGER NOT NULL, text TEXT NOT NULL, n_tokens INTEGER NOT NULL, vec BLOB NOT NULL);
         INSERT INTO meta (key, value) VALUES ('schema_version', '1');
         INSERT INTO files (id, path, mtime, size, blake3, n_chunks, indexed_at)
           VALUES (1, '/tmp/old.md', 1.0, 10, 'hash', 1, 1.0);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO chunks (file_id, ord, text, n_tokens, vec) VALUES (1, 0, ?1, 3, ?2)",
        rusqlite::params![text, vec_to_blob(&[1.0, 0.0])],
    )
    .unwrap();
}

#[test]
fn an_older_index_gains_its_lexical_half_without_being_rebuilt() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("index.db");
    legacy_index(&path, "the borgmatic backup of the vault drive");

    let store = Store::open(&path).expect("opens and migrates");

    let found = candidates(&store, "borgmatic", 10, None).expect("searches");
    assert_eq!(found.len(), 1, "the migration should have indexed the text");
    assert_eq!(store.stats().unwrap().chunks, 1, "nothing was re-embedded");
}

// --- reranking ------------------------------------------------------------

#[test]
fn a_backend_without_a_reranker_leaves_the_order_alone() {
    let documents = vec!["about backups".to_string(), "about invoices".to_string()];
    let scores = rerank(
        &MockBackend::new(),
        "backups",
        &documents,
        &RerankOptions::default(),
    )
    .expect("does not fail");
    assert_eq!(scores, None);
}

#[test]
fn asking_for_the_endpoint_that_is_not_there_is_an_error_not_a_shrug() {
    let documents = vec!["a".to_string(), "b".to_string()];
    let err = rerank(
        &MockBackend::new(),
        "q",
        &documents,
        &RerankOptions {
            mode: RerankMode::Endpoint,
            ..RerankOptions::default()
        },
    )
    .expect_err("must not be silently ignored");
    assert!(err.to_string().contains("rerank"), "got {err}");
}

#[test]
fn a_reranker_scores_the_excerpt_that_answers_the_question_higher() {
    let documents = vec![
        "an unrelated note about kayaks".to_string(),
        "the borgmatic backup of the vault drive".to_string(),
    ];
    let scores = rerank(
        &MockBackend::new().with_reranker(),
        "borgmatic vault",
        &documents,
        &RerankOptions::default(),
    )
    .expect("reranks")
    .expect("the mock has a reranker");

    assert_eq!(scores.len(), 2);
    assert!(scores[1] > scores[0], "{scores:?}");
}

#[test]
fn reranking_decides_the_final_order_and_says_so() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    let hits = search(
        &store,
        &MockBackend::new().with_reranker(),
        "borgmatic vault drive",
        &SearchOptions::default(),
    )
    .expect("searches");

    assert!(hits.len() >= 2);
    for hit in &hits {
        assert!(hit.scores.rerank.is_some(), "every hit was rescored");
        assert_eq!(hit.scores.rerank, Some(hit.score));
    }
    assert!(
        hits[0].score >= hits[1].score,
        "sorted by the reranker: {hits:?}"
    );
    assert!(hits[0].path.ends_with("backup.md"));
}

#[test]
fn one_excerpt_is_not_worth_a_rerank_call() {
    let scores = rerank(
        &MockBackend::new().with_reranker(),
        "anything",
        &["only one".to_string()],
        &RerankOptions::default(),
    )
    .expect("does not fail");
    assert_eq!(scores, None, "nothing to reorder");
}

#[test]
fn the_llm_reranker_reads_scores_out_of_a_loosely_formatted_reply() {
    assert_eq!(
        parse_scores("1: 8\n2: 3\n3: 10", 3),
        Some(vec![8.0, 3.0, 10.0])
    );
    assert_eq!(
        parse_scores("[1] 9.5/10\n2. 4 out of ten\n3 — 0", 3),
        Some(vec![9.5, 4.0, 0.0])
    );
}

#[test]
fn an_unrated_excerpt_sinks_but_does_not_sink_the_ranking() {
    assert_eq!(parse_scores("1: 7\n2: 6", 3), Some(vec![7.0, 6.0, 0.0]));
}

#[test]
fn a_reply_that_rated_almost_nothing_is_discarded() {
    assert_eq!(
        parse_scores("Sure! Here are my thoughts on excerpt 1.", 4),
        None
    );
    assert_eq!(parse_scores("", 3), None);
}

#[test]
fn a_chat_model_that_ignores_the_format_leaves_retrieval_alone() {
    let (_tmp, root) = tree();
    let store = indexed(&root);

    // The mock's chat just echoes the prompt, which holds no scores.
    let hits = search(
        &store,
        &MockBackend::new(),
        "borgmatic vault drive",
        &SearchOptions {
            rerank: RerankOptions {
                mode: RerankMode::Llm,
                ..RerankOptions::default()
            },
            ..Default::default()
        },
    )
    .expect("searches");

    assert!(hits[0].path.ends_with("backup.md"));
    assert!(hits[0].scores.rerank.is_none(), "no scores were believed");
}

// --- the pieces the store owns -------------------------------------------

#[test]
fn the_lexical_index_can_be_rebuilt_from_the_stored_text() {
    let mut store = Store::open_in_memory().unwrap();
    store
        .replace_file(
            "/tmp/a.md",
            1.0,
            10,
            "hash",
            &[chunk("the borgmatic backup runs nightly")],
            &[vec![1.0, 0.0]],
        )
        .unwrap();

    store.rebuild_lexical().expect("rebuilds");

    let found = candidates(&store, "borgmatic", 10, None).unwrap();
    assert_eq!(found.len(), 1, "a rebuild must not duplicate or lose rows");
}

#[test]
fn deleting_a_file_takes_its_lexical_rows_with_it() {
    let mut store = Store::open_in_memory().unwrap();
    store
        .replace_file(
            "/tmp/a.md",
            1.0,
            10,
            "hash",
            &[chunk("the borgmatic backup runs nightly")],
            &[vec![1.0, 0.0]],
        )
        .unwrap();
    assert!(store.delete_file("/tmp/a.md").unwrap());

    let found = candidates(&store, "borgmatic", 10, None).unwrap();
    assert!(found.is_empty(), "{found:?}");
}
