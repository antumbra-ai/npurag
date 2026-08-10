//! Behaviour at the size a real index reaches.
//!
//! The plan's target for M6 is fifty thousand chunks. This builds exactly that,
//! synthetically, and checks that a search still finds a planted needle — brute
//! force is only an acceptable answer if it actually holds up at that size.

use std::time::Instant;

use npurag::backend::{Backend, MockBackend};
use npurag::chunk::Chunk;
use npurag::search::{search, SearchOptions};
use npurag::store::Store;

const FILES: usize = 500;
const CHUNKS_PER_FILE: usize = 100;
const TOTAL: usize = FILES * CHUNKS_PER_FILE;

/// Fill an index with `TOTAL` chunks of filler, plus one that answers a very
/// specific question.
fn large_index(needle: &str) -> Store {
    let backend = MockBackend::new();
    let mut store = Store::open_in_memory().expect("opens");

    for file in 0..FILES {
        let mut chunks = Vec::with_capacity(CHUNKS_PER_FILE);
        for ord in 0..CHUNKS_PER_FILE {
            // Vary the wording so the vectors are not all identical, which would
            // make the ranking meaningless.
            let text = format!(
                "filler paragraph {ord} in document {file} covering routine matters \
                 such as logistics scheduling inventory and correspondence"
            );
            chunks.push(Chunk {
                ord,
                n_tokens: text.len().div_ceil(4),
                text,
            });
        }
        if file == FILES / 2 {
            chunks[CHUNKS_PER_FILE / 2] = Chunk {
                ord: CHUNKS_PER_FILE / 2,
                n_tokens: needle.len().div_ceil(4),
                text: needle.to_string(),
            };
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = backend.embed(&texts).expect("embeds");
        store
            .replace_file(
                &format!("/corpus/doc-{file:04}.md"),
                file as f64,
                texts.iter().map(|t| t.len() as u64).sum(),
                &format!("hash-{file}"),
                &chunks,
                &vectors,
            )
            .expect("writes");
    }
    store
}

#[test]
fn a_fifty_thousand_chunk_index_still_answers() {
    let needle = "the borgmatic retention policy keeps seven daily archives on the vault drive";
    let store = large_index(needle);

    let stats = store.stats().expect("stats");
    assert_eq!(stats.chunks as usize, TOTAL);
    assert_eq!(stats.files as usize, FILES);

    let started = Instant::now();
    let hits = search(
        &store,
        &MockBackend::new(),
        "what is the borgmatic retention policy for the vault drive?",
        &SearchOptions {
            top_k: 5,
            path: None,
        },
    )
    .expect("searches");
    let elapsed = started.elapsed();

    assert_eq!(hits.len(), 5);
    assert_eq!(
        hits[0].text, needle,
        "the planted chunk should outrank {TOTAL} pieces of filler"
    );
    // Not a benchmark — a guard against a regression that turns a linear scan
    // into something quadratic. Generous enough for a loaded CI runner.
    assert!(
        elapsed.as_secs() < 10,
        "scanning {TOTAL} vectors took {elapsed:?}"
    );
}

#[test]
fn a_path_filter_still_narrows_a_large_index() {
    let needle = "the quarterly invoice reconciliation ran against the ledger export";
    let store = large_index(needle);

    let hits = search(
        &store,
        &MockBackend::new(),
        "quarterly invoice reconciliation ledger",
        &SearchOptions {
            top_k: 3,
            path: Some("**/doc-0001.md".to_string()),
        },
    )
    .expect("searches");

    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.path.ends_with("doc-0001.md")));
}
