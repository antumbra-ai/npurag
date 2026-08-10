//! Command-line entry point.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use npurag::backend::{Backend, MockBackend, OpenAiBackend};
use npurag::config::{self, Config, Overrides};
use npurag::index::{index_dir, IndexOptions};
use npurag::store::Store;

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
        #[arg(long)]
        json: bool,
    },
    /// Answer a question grounded in the indexed files.
    Ask {
        question: String,
        #[arg(short = 'k', long, default_value_t = 8)]
        top_k: usize,
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
                ..Default::default()
            },
            include,
            exclude,
            *max_size,
            *follow_symlinks,
        ),
        Command::Search { .. } => bail!("`search` arrives in M2"),
        Command::Ask { .. } => bail!("`ask` arrives in M3"),
        Command::Prune => bail!("`prune` arrives in M5"),
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
        + report.unreadable;
    if skipped > 0 {
        writeln!(
            out,
            "skipped      {skipped} ({} binary, {} needing an extractor, {} too large, {} unreadable)",
            report.skipped_binary,
            report.skipped_unsupported,
            report.skipped_too_large,
            report.unreadable
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
