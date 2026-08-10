//! Command-line entry point.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use npurag::ask::{ask, origin_of, AskOptions};
use npurag::backend::{Backend, MockBackend, OpenAiBackend};
use npurag::config::{self, Config, Overrides};
use npurag::index::{index_dir, IndexOptions};
use npurag::search::{search, SearchOptions};
use npurag::store::Store;
use npurag::watch::{watch, Pipeline, WatchOptions};

#[derive(Parser)]
#[command(
    name = "npurag",
    version,
    about = "On-device semantic search and RAG over a local directory",
    long_about = None
)]
struct Cli {
    /// Config file to use instead of the default location.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Backend preset to activate, e.g. amd-flm or intel-ovms.
    #[arg(long, global = true, value_name = "NAME")]
    backend: Option<String>,

    /// Override the backend base URL, including its version prefix.
    #[arg(long, global = true, value_name = "URL")]
    base_url: Option<String>,

    /// Override the index database path.
    #[arg(long, global = true, value_name = "PATH")]
    db: Option<PathBuf>,

    /// Use the built-in deterministic backend; needs no server and no hardware.
    #[arg(long, global = true)]
    mock: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index a directory, incrementally.
    Index {
        path: PathBuf,
        /// Rebuild from scratch instead of skipping unchanged files.
        #[arg(long)]
        reindex: bool,
        /// Only index files matching this glob; repeatable.
        #[arg(long, value_name = "GLOB")]
        include: Vec<String>,
        /// Skip files matching this glob, on top of the configured excludes.
        #[arg(long, value_name = "GLOB")]
        exclude: Vec<String>,
        /// Skip files larger than this, in megabytes.
        #[arg(long, value_name = "MB")]
        max_size: Option<u64>,
        #[arg(long)]
        follow_symlinks: bool,
    },
    /// Search the index by meaning.
    Search {
        query: String,
        #[arg(short = 'k', long, default_value_t = 8)]
        top_k: usize,
        /// Only return hits whose path matches this glob.
        #[arg(long, value_name = "GLOB")]
        path: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Answer a question grounded in the indexed files.
    Ask {
        question: String,
        #[arg(short = 'k', long, default_value_t = 8)]
        top_k: usize,
        /// Only draw context from paths matching this glob.
        #[arg(long, value_name = "GLOB")]
        path: Option<String>,
        /// Chat model to use instead of the configured one.
        #[arg(long, value_name = "NAME")]
        model: Option<String>,
        /// Omit the list of excerpts the answer was built from.
        #[arg(long)]
        no_sources: bool,
        #[arg(long)]
        json: bool,
    },
    /// Show the active configuration and probe the backend.
    Status {
        /// Index whose statistics to report; defaults to the current directory.
        path: Option<PathBuf>,
    },
    /// Drop index entries whose files are gone.
    Prune,
    /// Re-index continuously as files change.
    Watch {
        path: PathBuf,
        /// Quiet period, in milliseconds, before a burst of changes is indexed.
        #[arg(long, default_value_t = 750)]
        debounce: u64,
    },
}

/// A backend together with the names it was resolved from, which the index
/// records so it can refuse to mix vectors from different models.
struct Active {
    backend: Box<dyn Backend>,
    name: String,
    embed_model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let overrides = Overrides {
        backend: cli.backend.clone(),
        base_url: cli.base_url.clone(),
        embed_model: None,
        chat_model: None,
        db: cli.db.clone(),
    };
    let config = Config::load(cli.config.as_deref(), &overrides)?;

    match &cli.command {
        Command::Status { path } => {
            status(&config, cli.mock, path.as_deref(), cli.config.as_deref())
        }
        Command::Index {
            path,
            reindex,
            include,
            exclude,
            max_size,
            follow_symlinks,
        } => index(
            &config,
            cli.mock,
            path,
            IndexOptions {
                reindex: *reindex,
                extract: config.extract_options(),
                ..Default::default()
            },
            include,
            exclude,
            *max_size,
            *follow_symlinks,
        ),
        Command::Search {
            query,
            top_k,
            path,
            json,
        } => search_cmd(
            &config,
            cli.mock,
            query,
            SearchOptions {
                top_k: *top_k,
                path: path.clone(),
            },
            *json,
        ),
        Command::Ask {
            question,
            top_k,
            path,
            model,
            no_sources,
            json,
        } => ask_cmd(
            &config,
            cli.mock,
            question,
            AskOptions {
                top_k: *top_k,
                path: path.clone(),
                model: model.clone(),
                ..Default::default()
            },
            !*no_sources,
            *json,
        ),
        Command::Prune => prune_cmd(&config),
        Command::Watch { path, debounce } => watch_cmd(&config, cli.mock, path, *debounce),
    }
}

fn activate(config: &Config, mock: bool) -> Result<Active> {
    if mock {
        return Ok(Active {
            backend: Box::new(MockBackend::new()),
            name: "mock".to_string(),
            embed_model: "mock".to_string(),
        });
    }
    let resolved = config.resolve_backend()?;
    Ok(Active {
        backend: Box::new(OpenAiBackend::new(
            resolved.name.clone(),
            resolved.base_url,
            resolved.embed_model.clone(),
            resolved.chat_model,
        )),
        name: resolved.name,
        embed_model: resolved.embed_model,
    })
}

/// Write to stdout, treating a closed pipe as a normal end of output.
///
/// Without this, `npurag search … | head` would abort with a broken-pipe panic,
/// because Rust ignores SIGPIPE and turns the write error into one.
fn emit(text: &str) -> Result<()> {
    use std::io::Write;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match out.write_all(text.as_bytes()).and_then(|()| out.flush()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

/// Open the index that covers the current directory.
///
/// Searching takes no path argument, so the working directory names the index
/// the same way `status` does; `--db` overrides it.
fn open_index(config: &Config) -> Result<(Store, PathBuf)> {
    let root = std::env::current_dir()?;
    let root = root.canonicalize().unwrap_or(root);
    let db_path = config.db_path_for(&root)?;
    if !db_path.is_file() {
        // Naming the directory would be misleading when --db chose the path.
        if config.db.is_some() {
            bail!("no index at {}", db_path.display());
        }
        bail!(
            "no index for {} yet — run `npurag index {}`, or point --db at an existing index",
            root.display(),
            root.display()
        );
    }
    let store = Store::open(&db_path)?;
    Ok((store, db_path))
}

/// Shorten a stored absolute path against the root the index was built from.
fn display_path<'a>(path: &'a str, root: Option<&str>) -> &'a str {
    root.and_then(|root| path.strip_prefix(root))
        .map(|rest| rest.trim_start_matches('/'))
        .unwrap_or(path)
}

/// A compact preview of a chunk: the first lines, whitespace collapsed.
fn snippet(text: &str, width: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= width {
        return flat;
    }
    let cut: String = flat.chars().take(width).collect();
    format!("{}…", cut.trim_end())
}

fn search_cmd(
    config: &Config,
    mock: bool,
    query: &str,
    options: SearchOptions,
    json: bool,
) -> Result<()> {
    use std::fmt::Write as _;

    let active = activate(config, mock)?;
    let (store, _) = open_index(config)?;
    store.ensure_model(&active.embed_model)?;

    let hits = search(&store, active.backend.as_ref(), query, &options)?;

    if json {
        let payload = serde_json::json!({
            "query": query,
            "origin": origin_of(&store)?,
            "hits": hits,
        });
        return emit(&format!("{}\n", serde_json::to_string_pretty(&payload)?));
    }

    if hits.is_empty() {
        return emit("no matches\n");
    }

    let root = store.stats()?.root_path;
    let mut out = String::new();
    for (rank, hit) in hits.iter().enumerate() {
        writeln!(
            out,
            "{:>2}. {:.3}  {}#{}",
            rank + 1,
            hit.score,
            display_path(&hit.path, root.as_deref()),
            hit.ord
        )?;
        writeln!(out, "    {}", snippet(&hit.text, 140))?;
    }
    emit(&out)
}

fn prune_cmd(config: &Config) -> Result<()> {
    use std::fmt::Write as _;

    let (mut store, db_path) = open_index(config)?;
    let removed = store.prune_missing()?;

    let mut out = String::new();
    if removed.is_empty() {
        writeln!(out, "nothing to prune in {}", db_path.display())?;
    } else {
        let root = store.stats()?.root_path;
        writeln!(out, "pruned {} file(s) no longer on disk:", removed.len())?;
        for path in &removed {
            writeln!(out, "  {}", display_path(path, root.as_deref()))?;
        }
    }
    emit(&out)
}

fn watch_cmd(config: &Config, mock: bool, path: &Path, debounce_ms: u64) -> Result<()> {
    let root = path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("could not resolve {}: {e}", path.display()))?;
    let active = activate(config, mock)?;
    let db_path = config.db_path_for(&root)?;

    let mut store = Store::open(&db_path)?;
    store.bind_to_model(&active.name, &active.embed_model, &root)?;

    emit(&format!(
        "watching {}\nindex    {}\nstop with Ctrl-C\n\n",
        root.display(),
        db_path.display()
    ))?;

    let pipeline = Pipeline {
        walk: &config.walk_options(&[], &[], None, false),
        chunk: &config.chunk_options(),
        index: &IndexOptions {
            extract: config.extract_options(),
            ..Default::default()
        },
        watch: &WatchOptions {
            debounce: std::time::Duration::from_millis(debounce_ms),
        },
    };

    watch(
        &mut store,
        active.backend.as_ref(),
        &root,
        &pipeline,
        |report| {
            // Only speak up when something actually changed, so an idle watcher
            // does not fill the terminal with noise.
            if report.indexed > 0 || report.removed > 0 {
                let _ = emit(&format!(
                    "indexed {} file(s), {} chunk(s); removed {}\n",
                    report.indexed, report.chunks_written, report.removed
                ));
            }
        },
    )
}

fn ask_cmd(
    config: &Config,
    mock: bool,
    question: &str,
    options: AskOptions,
    show_sources: bool,
    json: bool,
) -> Result<()> {
    use std::fmt::Write as _;

    let active = activate(config, mock)?;
    let (store, _) = open_index(config)?;
    store.ensure_model(&active.embed_model)?;

    let answer = ask(&store, active.backend.as_ref(), question, &options)?;

    if json {
        return emit(&format!("{}\n", serde_json::to_string_pretty(&answer)?));
    }

    let mut out = String::new();
    writeln!(out, "{}", answer.answer.trim())?;
    if show_sources && !answer.sources.is_empty() {
        let root = answer.origin.root.clone();
        match &root {
            Some(root) => writeln!(out, "\nSources (index of {root}):")?,
            None => writeln!(out, "\nSources:")?,
        }
        for source in &answer.sources {
            writeln!(
                out,
                "  [{}] {}#{}  ({:.3})",
                source.marker,
                display_path(&source.path, root.as_deref()),
                source.ord,
                source.score
            )?;
        }
    }
    emit(&out)
}

#[allow(clippy::too_many_arguments)]
fn index(
    config: &Config,
    mock: bool,
    path: &Path,
    options: IndexOptions,
    include: &[String],
    exclude: &[String],
    max_size: Option<u64>,
    follow_symlinks: bool,
) -> Result<()> {
    use std::fmt::Write as _;

    let root = path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("could not resolve {}: {e}", path.display()))?;
    let active = activate(config, mock)?;
    let db_path = config.db_path_for(&root)?;

    let mut store = Store::open(&db_path)?;
    store.bind_to_model(&active.name, &active.embed_model, &root)?;

    let walk_options = config.walk_options(include, exclude, max_size, follow_symlinks);
    let report = index_dir(
        &mut store,
        active.backend.as_ref(),
        &root,
        &walk_options,
        &config.chunk_options(),
        &options,
    )?;

    let mut out = String::new();
    writeln!(out, "root         {}", root.display())?;
    writeln!(out, "index        {}", db_path.display())?;
    writeln!(out, "scanned      {}", report.scanned)?;
    writeln!(
        out,
        "indexed      {} file(s), {} chunk(s)",
        report.indexed, report.chunks_written
    )?;
    writeln!(out, "unchanged    {}", report.unchanged)?;
    if report.removed > 0 {
        writeln!(
            out,
            "removed      {} file(s) no longer present",
            report.removed
        )?;
    }
    let skipped = report.skipped_binary
        + report.skipped_unsupported
        + report.skipped_too_large
        + report.extraction_failed
        + report.unreadable;
    if skipped > 0 {
        writeln!(
            out,
            "skipped      {skipped} ({} binary, {} needing an extractor, {} too large, \
             {} unreadable, {} failed to extract)",
            report.skipped_binary,
            report.skipped_unsupported,
            report.skipped_too_large,
            report.unreadable,
            report.extraction_failed
        )?;
    }
    emit(&out)
}

fn status(
    config: &Config,
    mock: bool,
    path: Option<&Path>,
    config_flag: Option<&Path>,
) -> Result<()> {
    use std::fmt::Write as _;

    let config_source = match config_flag {
        Some(p) => format!("{} (--config)", p.display()),
        None => match config::default_config_path() {
            Some(p) if p.is_file() => p.display().to_string(),
            Some(p) => format!("{} (not present, using defaults)", p.display()),
            None => "built-in defaults".to_string(),
        },
    };

    let root = match path {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir()?,
    };
    let root = root.canonicalize().unwrap_or(root);
    let db_path = config.db_path_for(&root)?;

    let mut out = String::new();
    writeln!(out, "config       {config_source}")?;

    if mock {
        let backend = MockBackend::new();
        writeln!(out, "backend      {} [--mock]", backend.describe())?;
        writeln!(out, "health       ok (deterministic, no server involved)")?;
    } else {
        let resolved = config.resolve_backend()?;
        writeln!(out, "backend      {}", resolved.name)?;
        writeln!(out, "base url     {}", resolved.base_url)?;
        writeln!(out, "embed model  {}", resolved.embed_model)?;
        writeln!(out, "chat model   {}", resolved.chat_model)?;

        let active = activate(config, false)?;
        writeln!(
            out,
            "health       {}",
            if active.backend.health() {
                "reachable".to_string()
            } else {
                format!("unreachable at {}/models", resolved.base_url)
            }
        )?;
    }

    writeln!(out, "root         {}", root.display())?;
    writeln!(out, "index        {}", db_path.display())?;
    if db_path.is_file() {
        let stats = Store::open(&db_path)?.stats()?;
        writeln!(
            out,
            "index stats  {} file(s), {} chunk(s)",
            stats.files, stats.chunks
        )?;
        if let Some(model) = &stats.embed_model {
            writeln!(
                out,
                "built with   {model}{}",
                stats
                    .embed_dim
                    .map(|d| format!(", {d} dimensions"))
                    .unwrap_or_default()
            )?;
        }
    } else {
        writeln!(
            out,
            "index stats  not created yet — run `npurag index <dir>`"
        )?;
    }

    let known: Vec<&str> = config.backends.keys().map(String::as_str).collect();
    writeln!(out, "presets      {}", known.join(", "))?;

    emit(&out)
}
