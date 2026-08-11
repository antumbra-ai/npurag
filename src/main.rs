//! Command-line entry point.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use npurag::ask::{ask, AskOptions};
use npurag::backend::{Backend, MockBackend, OpenAiBackend};
use npurag::config::{self, Config, Overrides};
use npurag::http::{self, HttpOptions, Service};
use npurag::index::{index_dir_with_progress, IndexOptions, Progress};
use npurag::mcp::{serve, McpServer};
use npurag::rerank::RerankMode;
use npurag::search::{search, search_payload, Scores, SearchMode, SearchOptions};
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
    /// Search the index by meaning and by wording.
    Search {
        query: String,
        #[arg(short = 'k', long, default_value_t = 8)]
        top_k: usize,
        /// Only return hits whose path matches this glob.
        #[arg(long, value_name = "GLOB")]
        path: Option<String>,
        /// Which retrievers to run; defaults to the configured mode.
        #[arg(long, value_enum, value_name = "MODE")]
        mode: Option<SearchMode>,
        /// How to rerank the shortlist; defaults to the configured setting.
        #[arg(long, value_enum, value_name = "MODE")]
        rerank: Option<RerankMode>,
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
        /// Which retrievers to run; defaults to the configured mode.
        #[arg(long, value_enum, value_name = "MODE")]
        mode: Option<SearchMode>,
        /// How to rerank the shortlist; defaults to the configured setting.
        #[arg(long, value_enum, value_name = "MODE")]
        rerank: Option<RerankMode>,
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
    /// Serve the index to an assistant over the Model Context Protocol.
    ///
    /// Speaks JSON-RPC on stdin and stdout; the client launches this. Nothing
    /// listens on a port.
    Mcp {
        /// Indexed directory to serve; defaults to the current one.
        path: Option<PathBuf>,
    },
    /// Answer HTTP requests against the index.
    ///
    /// For callers that are not an assistant — a script, a service, a cron job.
    /// Listens on loopback unless told otherwise, and refuses to listen
    /// anywhere else without a token.
    Serve {
        /// Indexed directory to serve; defaults to the current one.
        path: Option<PathBuf>,
        /// Address to listen on.
        #[arg(long, value_name = "ADDR", default_value = http::DEFAULT_BIND)]
        bind: String,
        /// Bearer token callers must present. NPURAG_TOKEN is preferred: an
        /// argument is visible to anyone who can list processes.
        #[arg(long, value_name = "TOKEN")]
        token: Option<String>,
    },
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
        rerank_model: None,
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
            mode,
            rerank,
            json,
        } => search_cmd(
            &config,
            cli.mock,
            query,
            retrieval(&config, *top_k, path.clone(), *mode, *rerank),
            *json,
        ),
        Command::Ask {
            question,
            top_k,
            path,
            mode,
            rerank,
            model,
            no_sources,
            json,
        } => ask_cmd(
            &config,
            cli.mock,
            question,
            AskOptions {
                search: retrieval(&config, *top_k, path.clone(), *mode, *rerank),
                model: model.clone(),
                ..Default::default()
            },
            !*no_sources,
            *json,
        ),
        Command::Mcp { path } => mcp_cmd(&config, cli.mock, path.as_deref()),
        Command::Serve { path, bind, token } => serve_cmd(
            &config,
            cli.mock,
            path.as_deref(),
            HttpOptions {
                bind: bind.clone(),
                // The environment wins: a token on the command line shows up in
                // `ps` for every other user on the machine.
                token: std::env::var("NPURAG_TOKEN").ok().or_else(|| token.clone()),
            },
        ),
        Command::Prune => prune_cmd(&config),
        Command::Watch { path, debounce } => watch_cmd(&config, cli.mock, path, *debounce),
    }
}

/// Retrieval settings for one command: the configured ones, with whatever the
/// command line said about them applied on top.
fn retrieval(
    config: &Config,
    top_k: usize,
    path: Option<String>,
    mode: Option<SearchMode>,
    rerank: Option<RerankMode>,
) -> SearchOptions {
    let mut options = config.search_options();
    options.top_k = top_k;
    options.path = path;
    if let Some(mode) = mode {
        options.mode = mode;
    }
    if let Some(rerank) = rerank {
        options.rerank.mode = rerank;
    }
    options
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
        backend: Box::new(
            OpenAiBackend::new(
                resolved.name.clone(),
                resolved.base_url,
                resolved.embed_model.clone(),
                resolved.chat_model,
            )
            .with_rerank_model(resolved.rerank_model),
        ),
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
        let payload = search_payload(&store, query, &hits)?;
        return emit(&format!("{}\n", serde_json::to_string_pretty(&payload)?));
    }

    if hits.is_empty() {
        return emit("no matches\n");
    }

    let root = store.stats()?.root_path;
    let mut out = String::new();
    for (rank, hit) in hits.iter().enumerate() {
        // With one retriever the tag would say only what the command already
        // said; with two it says which of them found this.
        let tag = if options.mode == SearchMode::Hybrid || hit.scores.rerank.is_some() {
            format!(" [{}]", provenance(&hit.scores))
        } else {
            String::new()
        };
        writeln!(
            out,
            "{:>2}. {:.3}{tag}  {}#{}",
            rank + 1,
            hit.score,
            display_path(&hit.path, root.as_deref()),
            hit.ord
        )?;
        writeln!(out, "    {}", snippet(&hit.text, 140))?;
    }
    emit(&out)
}

/// Which stages put a hit where it is: `v` for the vector search, `l` for BM25,
/// `r` when the reranker had the last word.
fn provenance(scores: &Scores) -> String {
    let mut parts = Vec::new();
    if scores.vector.is_some() {
        parts.push("v");
    }
    if scores.lexical.is_some() {
        parts.push("l");
    }
    if scores.rerank.is_some() {
        parts.push("r");
    }
    parts.join("+")
}

/// A progress bar on stderr, so piping stdout stays clean. indicatif hides it
/// automatically when stderr is not a terminal, which keeps cron logs readable.
fn new_progress_bar(total: u64) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template("{spinner} {pos}/{len} {wide_msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    bar
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

/// Open the index for a directory named on the command line.
///
/// The serving commands take the directory as an argument rather than reading
/// the working directory the way `search` does: whatever launches them —
/// an assistant, a service manager — chooses that directory, and the user does
/// not.
fn named_index(
    config: &Config,
    mock: bool,
    path: Option<&Path>,
) -> Result<(Store, PathBuf, Active)> {
    let root = match path {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir()?,
    };
    let root = root.canonicalize().unwrap_or(root);
    let db_path = config.db_path_for(&root)?;
    if !db_path.is_file() {
        bail!(
            "no index for {} yet — run `npurag index {}` before serving it",
            root.display(),
            root.display()
        );
    }

    let active = activate(config, mock)?;
    let store = Store::open(&db_path)?;
    store.ensure_model(&active.embed_model)?;
    Ok((store, db_path, active))
}

/// Answer HTTP requests against one index.
fn serve_cmd(config: &Config, mock: bool, path: Option<&Path>, options: HttpOptions) -> Result<()> {
    let (store, db_path, active) = named_index(config, mock, path)?;

    let service = Service {
        store: &store,
        backend: active.backend.as_ref(),
        defaults: config.search_options(),
        token: options.token.clone(),
    };

    http::serve(&service, &options, |address| {
        // stderr, so that redirecting stdout to a log leaves the banner where a
        // person can still see it.
        eprintln!(
            "npurag {} serving {} on http://{address}{}\nstop with Ctrl-C",
            env!("CARGO_PKG_VERSION"),
            db_path.display(),
            if options.token.is_some() {
                " (bearer token required)"
            } else {
                " (no token; loopback only)"
            }
        );
    })
}

/// Serve one index over MCP, on stdin and stdout.
fn mcp_cmd(config: &Config, mock: bool, path: Option<&Path>) -> Result<()> {
    let (store, db_path, active) = named_index(config, mock, path)?;

    // Diagnostics go to stderr: stdout carries protocol messages and nothing
    // else, and a stray line there would break the client's parser.
    eprintln!(
        "npurag {} serving {} over MCP on stdin/stdout",
        env!("CARGO_PKG_VERSION"),
        db_path.display()
    );

    let server = McpServer {
        store: &store,
        backend: active.backend.as_ref(),
        defaults: config.search_options(),
    };
    serve(&server, std::io::stdin().lock(), std::io::stdout().lock())
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
    let bar = std::cell::RefCell::new(None::<ProgressBar>);
    let report = index_dir_with_progress(
        &mut store,
        active.backend.as_ref(),
        &root,
        &walk_options,
        &config.chunk_options(),
        &options,
        &|event| match event {
            Progress::Planned { total } => {
                *bar.borrow_mut() = Some(new_progress_bar(total as u64));
            }
            Progress::Advanced { done, path, .. } => {
                if let Some(bar) = bar.borrow().as_ref() {
                    bar.set_position(done as u64);
                    bar.set_message(
                        path.file_name()
                            .unwrap_or(path.as_os_str())
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        },
    )?;
    if let Some(bar) = bar.borrow().as_ref() {
        bar.finish_and_clear();
    }

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
        writeln!(
            out,
            "rerank model {}",
            resolved
                .rerank_model
                .as_deref()
                .unwrap_or("none (set rerank_model, or use --rerank llm)")
        )?;

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

    writeln!(
        out,
        "retrieval    {} search, rerank {}",
        config.search.mode, config.search.rerank
    )?;

    let known: Vec<&str> = config.backends.keys().map(String::as_str).collect();
    writeln!(out, "presets      {}", known.join(", "))?;

    emit(&out)
}
