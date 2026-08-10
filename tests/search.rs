use std::path::{Path, PathBuf};

use npurag::backend::MockBackend;
use npurag::chunk::ChunkOptions;
use npurag::index::{index_dir, IndexOptions};
use npurag::search::{cosine, search, PathFilter, SearchOptions};
use npurag::store::Store;
use npurag::walk::WalkOptions;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn indexed_fixtures() -> Store {
    let mut store = Store::open_in_memory().expect("opens");
    index_dir(
        &mut store,
        &MockBackend::new(),
        &fixtures(),
        &WalkOptions::default(),
        &ChunkOptions::default(),
        &IndexOptions::default(),
    )
    .expect("indexes");
    store
}

fn top_path(query: &str, options: &SearchOptions) -> String {
    let store = indexed_fixtures();
    let hits = search(&store, &MockBackend::new(), query, options).expect("searches");
    hits.first().expect("at least one hit").path.clone()
}

#[test]
fn a_question_about_backups_finds_the_backup_note() {
    let path = top_path(
        "how did I configure the nightly backup with borgmatic",
        &SearchOptions::default(),
    );
    assert!(path.ends_with("notes/backup.md"), "top hit was {path}");
}

#[test]
fn a_question_about_the_project_finds_the_project_note() {
    let path = top_path(
        "what did we decide about the importer and the deadline",
        &SearchOptions::default(),
    );
    assert!(path.ends_with("notes/project-x.md"), "top hit was {path}");
}

#[test]
fn a_question_about_code_finds_the_source_file() {
    let path = top_path(
        "where does the TcpListener bind connections",
        &SearchOptions::default(),
    );
    assert!(path.ends_with("code/server.rs"), "top hit was {path}");
}

#[test]
fn results_are_ordered_by_descending_score() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "backup archive retention",
        &SearchOptions::default(),
    )
    .expect("searches");

    assert!(hits.len() > 1);
    for pair in hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "{} came before {}",
            pair[0].score,
            pair[1].score
        );
    }
}

#[test]
fn top_k_bounds_the_result_count() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "anything at all",
        &SearchOptions {
            top_k: 2,
            path: None,
        },
    )
    .expect("searches");
    assert_eq!(hits.len(), 2);
}

#[test]
fn asking_for_no_results_returns_none_and_embeds_nothing() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "anything",
        &SearchOptions {
            top_k: 0,
            path: None,
        },
    )
    .expect("searches");
    assert!(hits.is_empty());
}

#[test]
fn the_path_filter_narrows_results_to_matching_files() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "the listener binds a socket",
        &SearchOptions {
            top_k: 8,
            path: Some("**/notes/**".to_string()),
        },
    )
    .expect("searches");

    assert!(!hits.is_empty());
    assert!(
        hits.iter().all(|h| h.path.contains("/notes/")),
        "filter leaked non-matching paths"
    );
}

#[test]
fn a_bare_extension_glob_matches_on_the_file_name() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "anything",
        &SearchOptions {
            top_k: 8,
            path: Some("*.rs".to_string()),
        },
    )
    .expect("searches");

    assert!(!hits.is_empty(), "'*.rs' should match server.rs");
    assert!(hits.iter().all(|h| h.path.ends_with(".rs")));
}

#[test]
fn a_filter_matching_nothing_yields_no_hits() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "anything",
        &SearchOptions {
            top_k: 8,
            path: Some("*.tex".to_string()),
        },
    )
    .expect("searches");
    assert!(hits.is_empty());
}

#[test]
fn an_invalid_glob_is_reported_rather_than_ignored() {
    let store = indexed_fixtures();
    let err = search(
        &store,
        &MockBackend::new(),
        "anything",
        &SearchOptions {
            top_k: 8,
            path: Some("[".to_string()),
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("--path"), "got {err}");
}

#[test]
fn searching_an_empty_index_is_not_an_error() {
    let store = Store::open_in_memory().unwrap();
    let hits = search(
        &store,
        &MockBackend::new(),
        "anything",
        &SearchOptions::default(),
    )
    .expect("searches");
    assert!(hits.is_empty());
}

#[test]
fn hits_carry_the_text_that_was_matched() {
    let store = indexed_fixtures();
    let hits = search(
        &store,
        &MockBackend::new(),
        "borgmatic retention archives",
        &SearchOptions::default(),
    )
    .expect("searches");

    let top = &hits[0];
    assert!(top.text.contains("borgmatic"));
    assert!(top.n_tokens > 0);
    assert_eq!(top.ord, 0);
}

#[test]
fn path_filters_match_full_paths_and_directories() {
    let filter = PathFilter::new("**/notes/**").unwrap();
    assert!(filter.matches("/home/u/notes/a.md"));
    assert!(!filter.matches("/home/u/code/a.rs"));

    let by_name = PathFilter::new("*.md").unwrap();
    assert!(by_name.matches("/home/u/notes/a.md"));
    assert!(!by_name.matches("/home/u/notes/a.rs"));
}

#[test]
fn cosine_behaves_at_the_edges() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    // Magnitude must not matter; only direction.
    assert!((cosine(&[2.0, 0.0], &[9.0, 0.0]) - 1.0).abs() < 1e-6);
    // Degenerate inputs score zero instead of producing NaN.
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
}
