use std::path::{Path, PathBuf};

use npurag::ask::{ask, build_prompt, origin_of, select_context, AskOptions};
use npurag::backend::{Backend, Message, MockBackend, Role};
use npurag::chunk::ChunkOptions;
use npurag::index::{index_dir, IndexOptions};
use npurag::search::Hit;
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

fn hit(path: &str, text: &str, score: f32) -> Hit {
    Hit {
        path: path.to_string(),
        ord: 0,
        score,
        n_tokens: text.len().div_ceil(4) as i64,
        text: text.to_string(),
    }
}

#[test]
fn an_answer_comes_back_with_the_sources_it_used() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "how is the backup configured?",
        &AskOptions::default(),
    )
    .expect("answers");

    assert!(!answer.answer.is_empty());
    assert!(!answer.sources.is_empty());
    assert_eq!(answer.question, "how is the backup configured?");
    assert!(
        answer.sources.iter().any(|s| s.path.ends_with("backup.md")),
        "the backup note should be among the sources"
    );
}

#[test]
fn source_markers_are_numbered_from_one_and_are_contiguous() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "what was decided about the importer?",
        &AskOptions::default(),
    )
    .expect("answers");

    let markers: Vec<usize> = answer.sources.iter().map(|s| s.marker).collect();
    assert_eq!(markers, (1..=markers.len()).collect::<Vec<_>>());
}

#[test]
fn the_path_filter_restricts_where_the_answer_may_draw_from() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "what does this code do?",
        &AskOptions {
            path: Some("*.md".to_string()),
            ..Default::default()
        },
    )
    .expect("answers");

    assert!(!answer.sources.is_empty());
    assert!(answer.sources.iter().all(|s| s.path.ends_with(".md")));
}

#[test]
fn an_empty_index_yields_an_honest_non_answer() {
    let store = Store::open_in_memory().unwrap();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "anything?",
        &AskOptions::default(),
    )
    .expect("answers");

    assert!(answer.sources.is_empty());
    assert!(
        !answer.answer.is_empty(),
        "say something rather than nothing"
    );
}

#[test]
fn the_prompt_carries_the_question_and_every_excerpt() {
    let hits = [
        hit("/n/a.md", "the backup runs nightly", 0.9),
        hit("/n/b.md", "the importer ships first", 0.5),
    ];
    let selected: Vec<&Hit> = hits.iter().collect();
    let messages = build_prompt("when does the backup run?", &selected);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[1].role, Role::User);

    let user = &messages[1].content;
    assert!(user.contains("when does the backup run?"));
    assert!(user.contains("the backup runs nightly"));
    assert!(user.contains("the importer ships first"));
    assert!(user.contains("[1] /n/a.md"));
    assert!(user.contains("[2] /n/b.md"));
}

#[test]
fn the_system_prompt_forbids_answering_beyond_the_excerpts() {
    let messages = build_prompt("q", &[]);
    let system = &messages[0].content.to_lowercase();
    assert!(system.contains("only the excerpts"));
    assert!(system.contains("cite"));
    assert!(
        system.contains("same language"),
        "a Polish question deserves a Polish answer"
    );
}

#[test]
fn the_context_budget_drops_the_weakest_excerpts() {
    let long = "word ".repeat(200); // ~250 tokens
    let hits: Vec<Hit> = (0..10)
        .map(|i| hit(&format!("/n/{i}.md"), &long, 1.0 - i as f32 / 10.0))
        .collect();

    let selected = select_context(&hits, 600);
    assert!(selected.len() < hits.len(), "the budget must bite");
    assert!(!selected.is_empty());
    // Whatever survives is the top of the ranking, in order.
    for (i, chosen) in selected.iter().enumerate() {
        assert_eq!(chosen.path, format!("/n/{i}.md"));
    }
}

#[test]
fn a_single_oversized_excerpt_is_still_used() {
    let huge = "word ".repeat(5000);
    let hits = [hit("/n/big.md", &huge, 0.9)];

    let selected = select_context(&hits, 10);
    assert_eq!(selected.len(), 1, "answering from one excerpt beats none");
}

#[test]
fn the_model_override_reaches_the_backend() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "anything",
        &AskOptions {
            model: Some("some-other-llm".to_string()),
            ..Default::default()
        },
    )
    .expect("answers");

    // The mock reports the model it was asked for, which is how we can see the
    // override travelled all the way down.
    assert!(
        answer.answer.contains("some-other-llm"),
        "got {}",
        answer.answer
    );
}

#[test]
fn sources_are_reported_in_descending_score_order() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "backup archives and retention",
        &AskOptions::default(),
    )
    .expect("answers");

    for pair in answer.sources.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

#[test]
fn the_answer_is_whatever_the_backend_said() {
    let backend = MockBackend::new();
    let store = indexed_fixtures();
    let answer = ask(&store, &backend, "the backup", &AskOptions::default()).expect("answers");

    let direct = backend
        .chat(&[Message::user("x")], None)
        .expect("mock chats");
    let prefix = direct.split(']').next().expect("has a prefix");
    assert!(
        answer.answer.starts_with(prefix),
        "the answer should be the model's, unedited"
    );
}

#[test]
fn an_answer_names_the_index_it_came_from() {
    let store = indexed_fixtures();
    let answer = ask(
        &store,
        &MockBackend::new(),
        "how is the backup configured?",
        &AskOptions::default(),
    )
    .expect("answers");

    let root = answer.origin.root.expect("the index records its root");
    assert!(root.ends_with("tests/fixtures"), "got {root}");
    assert_eq!(answer.origin.files, 3);
    assert!(answer.origin.chunks >= 3);
}

#[test]
fn the_origin_records_which_model_built_the_index() {
    let store = indexed_fixtures();
    store
        .bind_to_model("amd-flm", "embeddinggemma-300m", Path::new("/tmp/notes"))
        .expect("binds");

    let origin = origin_of(&store).expect("reads origin");
    assert_eq!(origin.backend.as_deref(), Some("amd-flm"));
    assert_eq!(origin.embed_model.as_deref(), Some("embeddinggemma-300m"));
}

#[test]
fn even_an_unanswerable_question_reports_its_origin() {
    let store = Store::open_in_memory().unwrap();
    store
        .bind_to_model("mock", "mock", Path::new("/tmp/empty"))
        .unwrap();

    let answer = ask(
        &store,
        &MockBackend::new(),
        "anything?",
        &AskOptions::default(),
    )
    .expect("answers");

    assert!(answer.sources.is_empty());
    assert_eq!(answer.origin.root.as_deref(), Some("/tmp/empty"));
    assert_eq!(answer.origin.files, 0);
}
