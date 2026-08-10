//! Keeping an index fresh by watching the directory it was built from.
//!
//! Editors write files in bursts — a save can produce a rename, a create and
//! several modifies — so events are debounced into a single quiet period before
//! anything is re-indexed. The debouncing is deliberately separated from the
//! filesystem watcher so it can be tested with a plain channel instead of by
//! poking at a real directory and hoping the timing holds.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};

use crate::backend::Backend;
use crate::chunk::ChunkOptions;
use crate::index::{index_dir, IndexOptions, IndexReport};
use crate::store::Store;
use crate::walk::WalkOptions;

#[derive(Debug, Clone)]
pub struct WatchOptions {
    /// How long the directory must be quiet before a re-index starts.
    pub debounce: Duration,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            debounce: Duration::from_millis(750),
        }
    }
}

/// Block until something happens, then until it stops happening.
///
/// Returns `true` once a burst of activity has settled, or `false` when the
/// sender is gone and no more events can arrive.
pub fn wait_for_quiet<T>(events: &Receiver<T>, debounce: Duration) -> bool {
    // Wait indefinitely for the first event: an idle directory should cost
    // nothing, not a spin.
    if events.recv().is_err() {
        return false;
    }
    // Then swallow everything that arrives within the debounce window, so one
    // save does not trigger several re-index runs.
    loop {
        match events.recv_timeout(debounce) {
            Ok(_) => continue,
            Err(RecvTimeoutError::Timeout) => return true,
            Err(RecvTimeoutError::Disconnected) => return true,
        }
    }
}

/// Everything a re-index run needs, bundled so the watch loop stays readable.
pub struct Pipeline<'a> {
    pub walk: &'a WalkOptions,
    pub chunk: &'a ChunkOptions,
    pub index: &'a IndexOptions,
    pub watch: &'a WatchOptions,
}

/// Watch `root` and re-index after each burst of changes, forever.
///
/// `on_report` is called after every run, which is how the CLI prints progress
/// and how a test could observe the loop without capturing stdout.
pub fn watch<F>(
    store: &mut Store,
    backend: &dyn Backend,
    root: &Path,
    pipeline: &Pipeline<'_>,
    mut on_report: F,
) -> Result<()>
where
    F: FnMut(&IndexReport),
{
    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        // A closed receiver means the loop below has ended; nothing to report.
        let _ = tx.send(event);
    })
    .context("could not start the filesystem watcher")?;

    watcher
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("could not watch {}", root.display()))?;

    // Index once up front so the watcher starts from a current index rather
    // than whatever state the last run left behind.
    let report = index_dir(
        store,
        backend,
        root,
        pipeline.walk,
        pipeline.chunk,
        pipeline.index,
    )?;
    on_report(&report);

    while wait_for_quiet(&rx, pipeline.watch.debounce) {
        let report = index_dir(
            store,
            backend,
            root,
            pipeline.walk,
            pipeline.chunk,
            pipeline.index,
        )?;
        on_report(&report);
    }
    Ok(())
}
