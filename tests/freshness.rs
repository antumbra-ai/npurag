//! `prune` and the debounce that drives `watch`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};

use npurag::backend::MockBackend;
use npurag::chunk::ChunkOptions;
use npurag::index::{index_dir, IndexOptions};
use npurag::store::Store;
use npurag::walk::WalkOptions;
use npurag::watch::wait_for_quiet;
use tempfile::TempDir;

fn write(root: &Path, rel: &str, contents: &[u8]) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("has a parent")).expect("creates dirs");
    fs::write(path, contents).expect("writes");
}

fn tree() -> (TempDir, PathBuf) {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_path_buf();
    write(&root, "a.md", b"the first note about backups\n");
    write(&root, "b.md", b"the second note about invoices\n");
    (tmp, root)
}

fn index(store: &mut Store, root: &Path) {
    index_dir(
        store,
        &MockBackend::new(),
        root,
        &WalkOptions::default(),
        &ChunkOptions::default(),
        &IndexOptions::default(),
    )
    .expect("indexes");
}

#[test]
fn prune_drops_files_that_left_the_disk() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    index(&mut store, &root);
    assert_eq!(store.stats().unwrap().files, 2);

    fs::remove_file(root.join("a.md")).unwrap();
    let removed = store.prune_missing().expect("prunes");

    assert_eq!(removed.len(), 1);
    assert!(removed[0].ends_with("a.md"));
    assert_eq!(store.stats().unwrap().files, 1);
}

#[test]
fn prune_takes_the_chunks_with_it() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    index(&mut store, &root);
    let before = store.stats().unwrap().chunks;

    fs::remove_file(root.join("a.md")).unwrap();
    store.prune_missing().expect("prunes");

    assert!(store.stats().unwrap().chunks < before);
}

#[test]
fn prune_on_an_intact_index_changes_nothing() {
    let (_tmp, root) = tree();
    let mut store = Store::open_in_memory().unwrap();
    index(&mut store, &root);

    assert!(store.prune_missing().expect("prunes").is_empty());
    assert_eq!(store.stats().unwrap().files, 2);
}

#[test]
fn prune_on_an_empty_index_is_not_an_error() {
    let mut store = Store::open_in_memory().unwrap();
    assert!(store.prune_missing().expect("prunes").is_empty());
}

#[test]
fn the_debounce_waits_for_the_burst_to_finish() {
    let (tx, rx) = channel();
    let debounce = Duration::from_millis(120);

    thread::spawn(move || {
        // An editor saving a file: several events in quick succession, well
        // inside the debounce window.
        for _ in 0..5 {
            tx.send(()).expect("receiver alive");
            thread::sleep(Duration::from_millis(20));
        }
        // Hold the sender open afterwards. A watcher's sender lives as long as
        // the process, so what must end the wait here is the quiet window —
        // dropping it would let the test pass for the wrong reason.
        thread::sleep(Duration::from_secs(2));
    });

    let started = Instant::now();
    assert!(wait_for_quiet(&rx, debounce));
    // It must have waited past the last event, not fired on the first one.
    assert!(
        started.elapsed() >= debounce,
        "returned after {:?}, before the window closed",
        started.elapsed()
    );
}

#[test]
fn the_debounce_returns_once_the_sender_is_gone() {
    let (tx, rx) = channel::<()>();
    tx.send(()).unwrap();
    drop(tx);
    // One event did arrive, so this burst is real and should be handled.
    assert!(wait_for_quiet(&rx, Duration::from_millis(50)));
}

#[test]
fn the_debounce_reports_a_dead_channel_rather_than_spinning() {
    let (tx, rx) = channel::<()>();
    drop(tx);
    // No event ever arrived and none can, so the watch loop must end.
    assert!(!wait_for_quiet(&rx, Duration::from_millis(50)));
}

#[test]
fn a_quiet_directory_costs_nothing_until_something_happens() {
    let (tx, rx) = channel();
    let debounce = Duration::from_millis(50);

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(150));
        let _ = tx.send(());
    });

    let started = Instant::now();
    assert!(wait_for_quiet(&rx, debounce));
    assert!(
        started.elapsed() >= Duration::from_millis(150),
        "the first event must be waited for indefinitely, not polled"
    );
}
