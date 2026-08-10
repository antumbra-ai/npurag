//! Command-line entry point.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use npurag::backend::{Backend, MockBackend, OpenAiBackend};
use npurag::config::{self, Config, Overrides};

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
        Command::Index { .. } => bail!("`index` arrives in M1"),
        Command::Search { .. } => bail!("`search` arrives in M2"),
        Command::Ask { .. } => bail!("`ask` arrives in M3"),
        Command::Prune => bail!("`prune` arrives in M5"),
    }
}

fn build_backend(config: &Config, mock: bool) -> Result<Box<dyn Backend>> {
    if mock {
        return Ok(Box::new(MockBackend::new()));
    }
    let resolved = config.resolve_backend()?;
    Ok(Box::new(OpenAiBackend::new(
        resolved.name,
        resolved.base_url,
        resolved.embed_model,
        resolved.chat_model,
    )))
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

fn status(
    config: &Config,
    mock: bool,
    path: Option<&std::path::Path>,
    config_flag: Option<&std::path::Path>,
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

        let backend = build_backend(config, false)?;
        writeln!(
            out,
            "health       {}",
            if backend.health() {
                "reachable".to_string()
            } else {
                format!("unreachable at {}/models", resolved.base_url)
            }
        )?;
    }

    writeln!(out, "root         {}", root.display())?;
    writeln!(out, "index        {}", db_path.display())?;
    if db_path.is_file() {
        // Chunk and file counts land here once the store exists in M1.
        writeln!(out, "index stats  present")?;
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
