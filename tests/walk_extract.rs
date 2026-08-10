use std::fs;
use std::path::Path;

use npurag::walk::{walk, WalkOptions};
use tempfile::TempDir;

fn write(root: &Path, rel: &str, contents: &[u8]) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("has a parent")).expect("creates dirs");
    fs::write(path, contents).expect("writes");
}

fn names(root: &Path, options: &WalkOptions) -> Vec<String> {
    let (candidates, _) = walk(root, options).expect("walks");
    candidates
        .into_iter()
        .map(|c| {
            c.path
                .strip_prefix(root)
                .expect("under root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}

#[test]
fn finds_files_recursively_in_a_stable_order() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "b.txt", b"second");
    write(tmp.path(), "a.txt", b"first");
    write(tmp.path(), "nested/deep/c.txt", b"third");

    assert_eq!(
        names(tmp.path(), &WalkOptions::default()),
        vec!["a.txt", "b.txt", "nested/deep/c.txt"]
    );
}

#[test]
fn hidden_directories_are_left_alone() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "kept.txt", b"x");
    write(tmp.path(), ".git/objects/abc", b"x");
    write(tmp.path(), ".cache/thing", b"x");

    assert_eq!(names(tmp.path(), &WalkOptions::default()), vec!["kept.txt"]);
}

#[test]
fn exclude_globs_drop_matching_paths() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "keep.md", b"x");
    write(tmp.path(), "node_modules/lib/index.js", b"x");
    write(tmp.path(), "bundle.min.js", b"x");

    let options = WalkOptions {
        exclude: vec!["node_modules/**".to_string(), "**/*.min.js".to_string()],
        ..Default::default()
    };
    assert_eq!(names(tmp.path(), &options), vec!["keep.md"]);
}

#[test]
fn include_globs_narrow_the_walk_to_a_whitelist() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "notes.md", b"x");
    write(tmp.path(), "code.rs", b"x");
    write(tmp.path(), "data.csv", b"x");

    let options = WalkOptions {
        include: vec!["*.md".to_string()],
        ..Default::default()
    };
    assert_eq!(names(tmp.path(), &options), vec!["notes.md"]);
}

#[test]
fn oversized_files_are_skipped_and_counted() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "small.txt", b"tiny");
    write(tmp.path(), "big.txt", &vec![b'a'; 4096]);

    let options = WalkOptions {
        max_file_size: 1024,
        ..Default::default()
    };
    let (candidates, stats) = walk(tmp.path(), &options).expect("walks");
    assert_eq!(candidates.len(), 1);
    assert_eq!(stats.too_large, 1);
}

#[test]
fn gitignored_files_are_respected() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), ".gitignore", b"ignored.txt\n");
    write(tmp.path(), "ignored.txt", b"x");
    write(tmp.path(), "tracked.txt", b"x");

    // The .gitignore itself is hidden, so only the tracked file remains.
    assert_eq!(
        names(tmp.path(), &WalkOptions::default()),
        vec!["tracked.txt"]
    );
}

#[test]
fn candidates_carry_the_metadata_the_incremental_check_needs() {
    let tmp = TempDir::new().unwrap();
    write(tmp.path(), "file.txt", b"12345");

    let (candidates, _) = walk(tmp.path(), &WalkOptions::default()).expect("walks");
    assert_eq!(candidates[0].size, 5);
    assert!(candidates[0].mtime > 0.0);
}
