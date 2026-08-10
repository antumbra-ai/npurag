use std::fs;
use std::path::{Path, PathBuf};

use npurag::backend::MockBackend;
use npurag::chunk::ChunkOptions;
use npurag::index::{index_dir, IndexOptions, IndexReport};
use npurag::store::Store;
use npurag::walk::WalkOptions;
use tempfile::TempDir;

fn write(root: &Path, rel: &str, contents: &[u8]) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("has a parent")).expect("creates dirs");
    fs::write(path, contents).expect("writes");
}

fn run(store: &mut Store, root: &Path, options: IndexOptions) -> IndexReport {
    run_with(store, root, options, WalkOptions::default())
}

fn run_with(
    store: &mut Store,
    root: &Path,
    options: IndexOptions,
    walk: WalkOptions,
) -> IndexReport {
    index_dir(
        store,
        &MockBackend::new(),
        root,
        &walk,
        &ChunkOptions::default(),
        &options,
    )
    .expect("indexing succeeds on the mock backend")
}

fn tree() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();
    write(
        &root,
        "notes/backup.md",
        b"the backup runs nightly with borgmatic\n",
    );
    write(
        &root,
        "notes/project.md",
        b"project X ships the importer first\n",
    );
    write(
        &root,
        "code/server.rs",
        b"fn main() { println!(\"hi\"); }\n",
    );
    (tmp, root)
}

#[test]
fn a_first_run_indexes_everything() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();

    let report = run(&mut store, &root, IndexOptions::default());
    assert_eq!(report.scanned, 3);
    assert_eq!(report.indexed, 3);
    assert_eq!(report.unchanged, 0);
    assert!(report.chunks_written >= 3);

    let stats = store.stats().unwrap();
    assert_eq!(stats.files, 3);
    assert_eq!(stats.chunks as usize, report.chunks_written);
    assert_eq!(stats.embed_dim, Some(MockBackend::DEFAULT_DIM));
}

#[test]
fn a_second_run_skips_every_unchanged_file() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    let report = run(&mut store, &root, IndexOptions::default());
    assert_eq!(report.indexed, 0, "nothing changed, nothing to re-embed");
    assert_eq!(report.unchanged, 3);
    assert_eq!(report.chunks_written, 0);
    assert_eq!(store.stats().unwrap().files, 3);
}

#[test]
fn an_edited_file_is_re_embedded_and_the_rest_are_not() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    write(
        &root,
        "notes/backup.md",
        b"the backup now runs hourly to a NAS\n",
    );
    let report = run(&mut store, &root, IndexOptions::default());

    assert_eq!(report.indexed, 1);
    assert_eq!(report.unchanged, 2);
}

#[test]
fn rewriting_identical_bytes_costs_no_embeddings() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    // Same content, fresh mtime: the size and timestamp check fails, so the
    // content hash is what has to save the work here.
    let same = fs::read(root.join("notes/backup.md")).unwrap();
    write(&root, "notes/backup.md", &same);

    let report = run(&mut store, &root, IndexOptions::default());
    assert_eq!(report.indexed, 0, "identical bytes must not be re-embedded");
    assert_eq!(report.unchanged, 3);
}

#[test]
fn a_deleted_file_leaves_the_index() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    fs::remove_file(root.join("code/server.rs")).unwrap();
    let report = run(&mut store, &root, IndexOptions::default());

    assert_eq!(report.removed, 1);
    assert_eq!(store.stats().unwrap().files, 2);
    assert!(store
        .all_paths()
        .unwrap()
        .iter()
        .all(|p| !p.ends_with("server.rs")));
}

#[test]
fn a_new_file_is_picked_up_on_the_next_run() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    write(&root, "notes/new.md", b"a note added after the first run\n");
    let report = run(&mut store, &root, IndexOptions::default());

    assert_eq!(report.indexed, 1);
    assert_eq!(report.unchanged, 3);
    assert_eq!(store.stats().unwrap().files, 4);
}

#[test]
fn reindex_rebuilds_from_scratch() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    let report = run(
        &mut store,
        &root,
        IndexOptions {
            reindex: true,
            ..Default::default()
        },
    );
    assert_eq!(report.indexed, 3);
    assert_eq!(report.unchanged, 0);
    assert_eq!(store.stats().unwrap().files, 3);
}

#[test]
fn binary_and_unsupported_files_are_skipped_but_counted() {
    let (_tmp, root) = tree();
    write(&root, "data/image.bin", &[0u8, 1, 2, 3, 4]);
    write(&root, "docs/paper.pdf", b"%PDF-1.7 pretend");

    let mut store = Store::open_in_memory().unwrap();
    let report = run(&mut store, &root, IndexOptions::default());

    assert_eq!(report.indexed, 3);
    assert_eq!(report.skipped_binary, 1);
    assert_eq!(report.skipped_unsupported, 1);
    assert_eq!(store.stats().unwrap().files, 3);
}

#[test]
fn a_file_that_turns_binary_is_evicted_from_the_index() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    run(&mut store, &root, IndexOptions::default());

    write(&root, "notes/backup.md", &[b'a', 0, b'b']);
    let report = run(&mut store, &root, IndexOptions::default());

    assert_eq!(report.skipped_binary, 1);
    assert_eq!(
        store.stats().unwrap().files,
        2,
        "a file that stopped being text must not answer queries"
    );
}

#[test]
fn oversized_files_never_reach_the_backend() {
    let (_tmp, root) = tree();
    write(&root, "huge.txt", &vec![b'a'; 8192]);

    let mut store = Store::open_in_memory().unwrap();
    let report = run_with(
        &mut store,
        &root,
        IndexOptions::default(),
        WalkOptions {
            max_file_size: 1024,
            ..Default::default()
        },
    );

    assert_eq!(report.skipped_too_large, 1);
    assert_eq!(report.indexed, 3);
}

#[test]
fn an_empty_file_is_recorded_once_and_then_left_alone() {
    let (_tmp, root) = tree();
    write(&root, "notes/empty.md", b"");

    let mut store = Store::open_in_memory().unwrap();
    let first = run(&mut store, &root, IndexOptions::default());
    assert_eq!(first.indexed, 4, "an empty file still gets a record");

    let second = run(&mut store, &root, IndexOptions::default());
    assert_eq!(second.indexed, 0);
    assert_eq!(second.unchanged, 4);
}

#[test]
fn the_committed_fixtures_index_cleanly() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut store = Store::open_in_memory().unwrap();

    let report = run(&mut store, &root, IndexOptions::default());
    assert_eq!(report.indexed, 3);
    assert!(report.chunks_written >= 3);
}
