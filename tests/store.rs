use std::collections::HashSet;
use std::path::Path;

use npurag::chunk::Chunk;
use npurag::store::{blob_to_vec, vec_to_blob, Store};

fn chunk(ord: usize, text: &str) -> Chunk {
    Chunk {
        ord,
        text: text.to_string(),
        n_tokens: text.len().div_ceil(4),
    }
}

fn store_with_one_file() -> Store {
    let mut store = Store::open_in_memory().expect("opens");
    store
        .replace_file(
            "/tmp/a.md",
            100.5,
            42,
            "hash-a",
            &[chunk(0, "first chunk"), chunk(1, "second chunk")],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .expect("writes");
    store
}

#[test]
fn vectors_survive_the_blob_round_trip() {
    let vector = vec![0.5, -0.25, 1.0, 0.0];
    assert_eq!(blob_to_vec(&vec_to_blob(&vector)).unwrap(), vector);
}

#[test]
fn a_truncated_blob_is_rejected_rather_than_guessed_at() {
    assert!(blob_to_vec(&[1, 2, 3]).is_err());
}

#[test]
fn a_written_file_can_be_read_back() {
    let store = store_with_one_file();
    let record = store.file_record("/tmp/a.md").unwrap().expect("present");

    assert_eq!(record.blake3, "hash-a");
    assert_eq!(record.size, 42);
    assert_eq!(record.n_chunks, 2);

    let stats = store.stats().unwrap();
    assert_eq!(stats.files, 1);
    assert_eq!(stats.chunks, 2);
}

#[test]
fn an_unknown_path_reads_back_as_nothing() {
    let store = Store::open_in_memory().unwrap();
    assert!(store.file_record("/tmp/missing").unwrap().is_none());
}

#[test]
fn rewriting_a_file_replaces_its_chunks_instead_of_appending() {
    let mut store = store_with_one_file();
    store
        .replace_file(
            "/tmp/a.md",
            101.0,
            10,
            "hash-b",
            &[chunk(0, "only chunk now")],
            &[vec![1.0, 1.0]],
        )
        .expect("rewrites");

    let stats = store.stats().unwrap();
    assert_eq!(stats.files, 1);
    assert_eq!(stats.chunks, 1, "the old chunks must not linger");
}

#[test]
fn deleting_a_file_cascades_to_its_chunks() {
    let store = store_with_one_file();
    assert!(store.delete_file("/tmp/a.md").unwrap());

    let stats = store.stats().unwrap();
    assert_eq!(stats.files, 0);
    assert_eq!(stats.chunks, 0, "chunks must not outlive their file");
}

#[test]
fn touching_a_file_updates_metadata_without_touching_chunks() {
    let store = store_with_one_file();
    store.touch_file("/tmp/a.md", 999.0, 43).unwrap();

    let record = store.file_record("/tmp/a.md").unwrap().unwrap();
    assert_eq!(record.mtime, 999.0);
    assert_eq!(record.size, 43);
    assert_eq!(record.blake3, "hash-a");
    assert_eq!(store.stats().unwrap().chunks, 2);
}

#[test]
fn files_the_walk_no_longer_sees_are_dropped() {
    let mut store = store_with_one_file();
    store
        .replace_file(
            "/tmp/b.md",
            1.0,
            1,
            "hash-b",
            &[chunk(0, "b")],
            &[vec![1.0, 0.0]],
        )
        .unwrap();

    let seen: HashSet<String> = ["/tmp/b.md".to_string()].into_iter().collect();
    assert_eq!(store.delete_missing(&seen).unwrap(), 1);
    assert_eq!(store.all_paths().unwrap(), vec!["/tmp/b.md"]);
}

#[test]
fn a_mismatched_vector_count_is_refused() {
    let mut store = Store::open_in_memory().unwrap();
    let err = store
        .replace_file("/tmp/a.md", 1.0, 1, "h", &[chunk(0, "one")], &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("0 vectors"), "got {err}");
}

#[test]
fn the_embedding_width_is_recorded_and_then_enforced() {
    let mut store = store_with_one_file();
    assert_eq!(store.stats().unwrap().embed_dim, Some(2));

    let err = store
        .replace_file(
            "/tmp/c.md",
            1.0,
            1,
            "h",
            &[chunk(0, "c")],
            &[vec![1.0, 0.0, 0.0]],
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("--reindex"), "got {err}");
}

#[test]
fn an_index_refuses_vectors_from_a_different_embedding_model() {
    let store = Store::open_in_memory().unwrap();
    let root = Path::new("/tmp");
    store
        .bind_to_model("amd-flm", "embeddinggemma-300m", root)
        .unwrap();
    // Rebinding to the same model is how every later run starts.
    store
        .bind_to_model("amd-flm", "embeddinggemma-300m", root)
        .unwrap();

    let err = store
        .bind_to_model("intel-ovms", "some-other-model", root)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not comparable"), "got {err}");
}

#[test]
fn clearing_keeps_the_index_identity() {
    let store = store_with_one_file();
    store
        .bind_to_model("mock", "mock", Path::new("/tmp"))
        .unwrap();
    store.clear().unwrap();

    let stats = store.stats().unwrap();
    assert_eq!(stats.files, 0);
    assert_eq!(stats.chunks, 0);
    assert_eq!(stats.embed_model.as_deref(), Some("mock"));
}
