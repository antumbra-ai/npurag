//! Configuration: file, environment and CLI flags, layered in that order.
//!
//! Every hardware-specific value lives here and nowhere else. Switching from AMD
//! to Intel is a change of preset, never a change of code.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::chunk::ChunkOptions;
use crate::extract::ExtractOptions;
use crate::walk::WalkOptions;

pub const AMD_FLM: &str = "amd-flm";
pub const INTEL_OVMS: &str = "intel-ovms";

/// A single backend target: where to reach it and what to ask it for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPreset {
    /// Base URL including the API version prefix. FastFlowLM uses `/v1`; newer
    /// OpenVINO Model Server builds use `/v3`, which is exactly why this is
    /// configuration rather than a constant in the HTTP client.
    pub base_url: String,
    pub embed_model: String,
    pub chat_model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Name of the active entry in `backends`.
    pub backend: String,
    pub max_file_size_mb: u64,
    pub chunk_tokens: usize,
    pub chunk_overlap: usize,
    pub exclude: Vec<String>,
    /// Allow falling back to locally installed `pdftotext` / `pandoc` for
    /// formats this build has no extractor for. Nothing leaves the machine;
    /// set it to false to keep npurag from spawning any process at all.
    pub external_extractors: bool,
    /// Explicit index location; when unset a per-root path under the user's data
    /// directory is used.
    pub db: Option<PathBuf>,
    /// Tables must be serialised after scalars, so this field stays last.
    pub backends: BTreeMap<String, BackendPreset>,
}

impl Default for Config {
    fn default() -> Self {
        let mut backends = BTreeMap::new();
        backends.insert(
            AMD_FLM.to_string(),
            BackendPreset {
                base_url: "http://localhost:52625/v1".to_string(),
                embed_model: "embeddinggemma-300m".to_string(),
                chat_model: "gemma3:4b".to_string(),
            },
        );
        backends.insert(
            INTEL_OVMS.to_string(),
            BackendPreset {
                base_url: "http://localhost:8000/v3".to_string(),
                embed_model: "embeddinggemma-300m".to_string(),
                chat_model: "gemma3:4b-int4-ov".to_string(),
            },
        );
        Self {
            backend: AMD_FLM.to_string(),
            max_file_size_mb: 5,
            chunk_tokens: 400,
            chunk_overlap: 60,
            external_extractors: true,
            exclude: vec![
                ".git/**".to_string(),
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "**/*.min.js".to_string(),
            ],
            db: None,
            backends,
        }
    }
}

/// Values supplied on the command line, which win over everything else.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub backend: Option<String>,
    pub base_url: Option<String>,
    pub embed_model: Option<String>,
    pub chat_model: Option<String>,
    pub db: Option<PathBuf>,
}

/// The active backend after all layers have been applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBackend {
    pub name: String,
    pub base_url: String,
    pub embed_model: String,
    pub chat_model: String,
}

impl Config {
    /// Parse a config file. Missing keys fall back to [`Config::default`].
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).context("could not parse the config file")
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("could not serialise the config")
    }

    /// Load the config file if it exists, then overlay environment variables and
    /// command-line flags. A missing file is not an error — the defaults are a
    /// working configuration.
    pub fn load(path: Option<&Path>, overrides: &Overrides) -> Result<Self> {
        let mut config = match path {
            Some(p) => {
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("could not read the config file {}", p.display()))?;
                Self::from_toml(&text)?
            }
            None => match default_config_path() {
                Some(p) if p.is_file() => {
                    let text = std::fs::read_to_string(&p).with_context(|| {
                        format!("could not read the config file {}", p.display())
                    })?;
                    Self::from_toml(&text)?
                }
                _ => Self::default(),
            },
        };
        config.apply_env();
        config.apply_overrides(overrides);
        Ok(config)
    }

    pub fn apply_env(&mut self) {
        self.apply_env_with(|key| std::env::var(key).ok());
    }

    /// Environment overlay, parameterised over the lookup so tests need not
    /// mutate the real process environment.
    pub fn apply_env_with<F>(&mut self, get: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        self.apply_overrides(&Overrides {
            backend: get("NPURAG_BACKEND"),
            base_url: get("NPURAG_BASE_URL"),
            embed_model: get("NPURAG_EMBED_MODEL"),
            chat_model: get("NPURAG_CHAT_MODEL"),
            db: get("NPURAG_DB").map(PathBuf::from),
        });
    }

    /// Apply overrides in place. Per-preset values are written into the active
    /// preset, creating it when the name is not one of the built-ins — that is
    /// what makes `--backend foo --base-url …` work without a config file.
    pub fn apply_overrides(&mut self, overrides: &Overrides) {
        if let Some(name) = &overrides.backend {
            self.backend = name.clone();
        }
        if let Some(db) = &overrides.db {
            self.db = Some(db.clone());
        }

        let touches_preset = overrides.base_url.is_some()
            || overrides.embed_model.is_some()
            || overrides.chat_model.is_some();
        if !touches_preset {
            return;
        }

        let fallback = Self::default();
        let template = fallback
            .backends
            .get(&self.backend)
            .cloned()
            .unwrap_or_else(|| BackendPreset {
                base_url: String::new(),
                embed_model: String::new(),
                chat_model: String::new(),
            });
        let preset = self
            .backends
            .entry(self.backend.clone())
            .or_insert(template);

        if let Some(base_url) = &overrides.base_url {
            preset.base_url = base_url.clone();
        }
        if let Some(embed_model) = &overrides.embed_model {
            preset.embed_model = embed_model.clone();
        }
        if let Some(chat_model) = &overrides.chat_model {
            preset.chat_model = chat_model.clone();
        }
    }

    pub fn resolve_backend(&self) -> Result<ResolvedBackend> {
        let preset = self.backends.get(&self.backend).ok_or_else(|| {
            let known: Vec<&str> = self.backends.keys().map(String::as_str).collect();
            anyhow!(
                "unknown backend `{}` (configured presets: {})",
                self.backend,
                known.join(", ")
            )
        })?;
        if preset.base_url.trim().is_empty() {
            return Err(anyhow!(
                "backend `{}` has no base_url; set it in the config file or pass --base-url",
                self.backend
            ));
        }
        Ok(ResolvedBackend {
            name: self.backend.clone(),
            base_url: preset.base_url.trim_end_matches('/').to_string(),
            embed_model: preset.embed_model.clone(),
            chat_model: preset.chat_model.clone(),
        })
    }

    pub fn extract_options(&self) -> ExtractOptions {
        ExtractOptions {
            external_tools: self.external_extractors,
        }
    }

    pub fn chunk_options(&self) -> ChunkOptions {
        ChunkOptions {
            target_tokens: self.chunk_tokens,
            overlap_tokens: self.chunk_overlap,
            ..ChunkOptions::default()
        }
    }

    /// Build the traversal filters, letting command-line globs extend — never
    /// replace — the excludes configured in the file.
    pub fn walk_options(
        &self,
        include: &[String],
        exclude: &[String],
        max_size_mb: Option<u64>,
        follow_symlinks: bool,
    ) -> WalkOptions {
        let mut excludes = self.exclude.clone();
        excludes.extend(exclude.iter().cloned());
        WalkOptions {
            max_file_size: max_size_mb.unwrap_or(self.max_file_size_mb) * 1024 * 1024,
            follow_symlinks,
            include: include.to_vec(),
            exclude: excludes,
            ..WalkOptions::default()
        }
    }

    /// Where the index for `root` lives. One index per root, keyed by a digest of
    /// the absolute path so that two roots never share a database.
    pub fn db_path_for(&self, root: &Path) -> Result<PathBuf> {
        if let Some(db) = &self.db {
            return Ok(db.clone());
        }
        let dir = default_data_dir()
            .ok_or_else(|| anyhow!("could not determine a data directory; pass --db"))?;
        Ok(dir.join(root_key(root)).join("index.db"))
    }
}

/// A short, stable, filesystem-safe key for a root path.
///
/// blake3 arrives with the indexer in M1; until then a small inline hash keeps
/// M0 free of dependencies it does not yet need.
fn root_key(root: &Path) -> String {
    let text = root.to_string_lossy();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

pub fn default_config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "npurag")
        .map(|dirs| dirs.config_dir().join("config.toml"))
}

pub fn default_data_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "npurag").map(|dirs| dirs.data_dir().to_path_buf())
}
