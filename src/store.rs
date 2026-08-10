//! The SQLite index: schema, and the reads and writes the pipeline needs.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::chunk::Chunk;

pub const SCHEMA_VERSION: &str = "1";

pub mod meta_keys {
    pub const SCHEMA_VERSION: &str = "schema_version";
    pub const EMBED_MODEL: &str = "embed_model";
    pub const EMBED_DIM: &str = "embed_dim";
    pub const CHAT_MODEL: &str = "chat_model";
    pub const BACKEND: &str = "backend";
    pub const ROOT_PATH: &str = "root_path";
    pub const CREATED_AT: &str = "created_at";
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS files (
  id         INTEGER PRIMARY KEY,
  path       TEXT UNIQUE NOT NULL,
  mtime      REAL    NOT NULL,
  size       INTEGER NOT NULL,
  blake3     TEXT    NOT NULL,
  n_chunks   INTEGER NOT NULL,
  indexed_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id       INTEGER PRIMARY KEY,
  file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ord      INTEGER NOT NULL,
  text     TEXT    NOT NULL,
  n_tokens INTEGER NOT NULL,
  vec      BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
"#;

#[derive(Debug, Clone, PartialEq)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub mtime: f64,
    pub size: i64,
    pub blake3: String,
    pub n_chunks: i64,
    pub indexed_at: f64,
}

/// The parts of a chunk that are only read once ranking has picked it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkText {
    pub ord: i64,
    pub text: String,
    pub n_tokens: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub files: i64,
    pub chunks: i64,
    pub embed_model: Option<String>,
    pub embed_dim: Option<usize>,
    pub backend: Option<String>,
    pub root_path: Option<String>,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("could not create {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("could not open the index at {}", path.display()))?;
        Self::from_connection(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "foreign_keys", true)?;
        // Ignore the returned mode: an in-memory database stays in `memory`.
        let _: Option<String> = conn
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .optional()?;
        conn.execute_batch(SCHEMA)
            .context("could not apply the index schema")?;

        let store = Self { conn };
        match store.meta_get(meta_keys::SCHEMA_VERSION)? {
            None => {
                store.meta_set(meta_keys::SCHEMA_VERSION, SCHEMA_VERSION)?;
                store.meta_set(meta_keys::CREATED_AT, &now_seconds().to_string())?;
            }
            Some(found) if found != SCHEMA_VERSION => {
                return Err(anyhow!(
                    "this index was written by schema version {found}, this build speaks {SCHEMA_VERSION}; \
                     delete it and run `npurag index` again"
                ));
            }
            Some(_) => {}
        }
        Ok(store)
    }

    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Fail if this index holds vectors from a different embedding model.
    ///
    /// Embeddings from two models occupy unrelated spaces, so comparing across
    /// them produces confident nonsense rather than an obvious error.
    pub fn ensure_model(&self, embed_model: &str) -> Result<()> {
        match self.meta_get(meta_keys::EMBED_MODEL)? {
            Some(found) if found != embed_model => Err(anyhow!(
                "this index was built with embedding model `{found}`, but `{embed_model}` is \
                 configured; vectors from different models are not comparable — rerun with \
                 --reindex to rebuild it"
            )),
            _ => Ok(()),
        }
    }

    /// Record what produced this index, after checking it is compatible.
    pub fn bind_to_model(&self, backend: &str, embed_model: &str, root: &Path) -> Result<()> {
        self.ensure_model(embed_model)?;
        if self.meta_get(meta_keys::EMBED_MODEL)?.is_none() {
            self.meta_set(meta_keys::EMBED_MODEL, embed_model)?;
        }
        self.meta_set(meta_keys::BACKEND, backend)?;
        self.meta_set(meta_keys::ROOT_PATH, &root.to_string_lossy())?;
        Ok(())
    }

    /// Stream every stored vector with the path it came from.
    ///
    /// Chunk text is deliberately left behind: it dwarfs the vectors, and only
    /// the handful of rows that survive ranking ever need to be read.
    pub fn scan_vectors<F>(&self, mut visit: F) -> Result<()>
    where
        F: FnMut(i64, &str, Vec<f32>),
    {
        let mut stmt = self.conn.prepare(
            "SELECT chunks.id, files.path, chunks.vec
             FROM chunks JOIN files ON files.id = chunks.file_id",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            visit(id, &path, blob_to_vec(&blob)?);
        }
        Ok(())
    }

    /// Fetch the full text of specific chunks, keyed by chunk id.
    pub fn chunk_texts(&self, ids: &[i64]) -> Result<HashMap<i64, ChunkText>> {
        let mut found = HashMap::with_capacity(ids.len());
        let mut stmt = self.conn.prepare(
            "SELECT chunks.id, chunks.ord, chunks.text, chunks.n_tokens
             FROM chunks WHERE chunks.id = ?1",
        )?;
        for id in ids {
            let row = stmt
                .query_row(params![id], |row| {
                    Ok(ChunkText {
                        ord: row.get(1)?,
                        text: row.get(2)?,
                        n_tokens: row.get(3)?,
                    })
                })
                .optional()?;
            if let Some(row) = row {
                found.insert(*id, row);
            }
        }
        Ok(found)
    }

    /// Remember the embedding width the first time one is seen, and reject any
    /// later vector that disagrees.
    fn note_embed_dim(&self, dim: usize) -> Result<()> {
        match self.meta_get(meta_keys::EMBED_DIM)? {
            Some(found) if found != dim.to_string() => Err(anyhow!(
                "the backend returned a {dim}-dimensional embedding but this index holds \
                 {found}-dimensional vectors; rerun with --reindex"
            )),
            Some(_) => Ok(()),
            None => self.meta_set(meta_keys::EMBED_DIM, &dim.to_string()),
        }
    }

    pub fn file_record(&self, path: &str) -> Result<Option<FileRecord>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, path, mtime, size, blake3, n_chunks, indexed_at
                 FROM files WHERE path = ?1",
                params![path],
                |row| {
                    Ok(FileRecord {
                        id: row.get(0)?,
                        path: row.get(1)?,
                        mtime: row.get(2)?,
                        size: row.get(3)?,
                        blake3: row.get(4)?,
                        n_chunks: row.get(5)?,
                        indexed_at: row.get(6)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn all_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files ORDER BY path")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Replace a file and all of its chunks in one transaction, so a failure
    /// mid-write can never leave a file half-indexed.
    pub fn replace_file(
        &mut self,
        path: &str,
        mtime: f64,
        size: u64,
        blake3: &str,
        chunks: &[Chunk],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.len() != vectors.len() {
            return Err(anyhow!(
                "got {} vectors for {} chunks of {path}",
                vectors.len(),
                chunks.len()
            ));
        }
        if let Some(first) = vectors.first() {
            self.note_embed_dim(first.len())?;
        }

        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        tx.execute(
            "INSERT INTO files (path, mtime, size, blake3, n_chunks, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                path,
                mtime,
                size as i64,
                blake3,
                chunks.len() as i64,
                now_seconds()
            ],
        )?;
        let file_id = tx.last_insert_rowid();
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (file_id, ord, text, n_tokens, vec)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (chunk, vector) in chunks.iter().zip(vectors) {
                stmt.execute(params![
                    file_id,
                    chunk.ord as i64,
                    chunk.text,
                    chunk.n_tokens as i64,
                    vec_to_blob(vector)
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Refresh the cheap metadata of a file whose content hash was unchanged, so
    /// the next run can take the fast path instead of re-hashing it.
    pub fn touch_file(&self, path: &str, mtime: f64, size: u64) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET mtime = ?2, size = ?3 WHERE path = ?1",
            params![path, mtime, size as i64],
        )?;
        Ok(())
    }

    pub fn delete_file(&self, path: &str) -> Result<bool> {
        Ok(self
            .conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])?
            > 0)
    }

    /// Drop every indexed file that the walk no longer sees.
    pub fn delete_missing(&mut self, seen: &HashSet<String>) -> Result<usize> {
        let gone: Vec<String> = self
            .all_paths()?
            .into_iter()
            .filter(|p| !seen.contains(p))
            .collect();
        if gone.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.transaction()?;
        for path in &gone {
            tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        }
        tx.commit()?;
        Ok(gone.len())
    }

    /// Drop indexed files that no longer exist on disk.
    ///
    /// Unlike [`Store::delete_missing`] this consults the filesystem directly,
    /// so it works without walking the tree — which is what makes `prune` cheap
    /// enough to run on its own.
    pub fn prune_missing(&mut self) -> Result<Vec<String>> {
        let gone: Vec<String> = self
            .all_paths()?
            .into_iter()
            .filter(|path| !Path::new(path).exists())
            .collect();
        if gone.is_empty() {
            return Ok(gone);
        }
        let tx = self.conn.transaction()?;
        for path in &gone {
            tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        }
        tx.commit()?;
        Ok(gone)
    }

    pub fn stats(&self) -> Result<Stats> {
        Ok(Stats {
            files: self
                .conn
                .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?,
            chunks: self
                .conn
                .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?,
            embed_model: self.meta_get(meta_keys::EMBED_MODEL)?,
            embed_dim: self
                .meta_get(meta_keys::EMBED_DIM)?
                .and_then(|v| v.parse().ok()),
            backend: self.meta_get(meta_keys::BACKEND)?,
            root_path: self.meta_get(meta_keys::ROOT_PATH)?,
        })
    }

    /// Wipe the indexed content, keeping the database and its identity metadata.
    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM files", [])?;
        Ok(())
    }
}

/// Vectors are stored as little-endian `f32`, which is what the plan specifies
/// and what a future SIMD path can read without conversion.
pub fn vec_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

pub fn blob_to_vec(blob: &[u8]) -> Result<Vec<f32>> {
    if !blob.len().is_multiple_of(4) {
        return Err(anyhow!(
            "a stored vector is {} bytes, which is not a whole number of f32 values",
            blob.len()
        ));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
