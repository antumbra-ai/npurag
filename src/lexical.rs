//! The lexical half of retrieval: BM25 over the full-text index.
//!
//! Embeddings are good at meaning and bad at literals. A part number, an error
//! code, a surname, a flag like `--follow-symlinks` — these carry almost no
//! semantic signal, and a nearest-neighbour search will happily return a chunk
//! that is *about* the same subject while missing the one that actually says the
//! word. BM25 is the opposite: it knows nothing about meaning and everything
//! about rare terms. Running both and fusing them is what makes retrieval hold
//! up on real notes, which mix prose and identifiers on the same page.
//!
//! SQLite's FTS5 does the work, so the lexical index lives in the same file as
//! everything else and needs no new dependency and no second daemon.

use anyhow::Result;

use crate::search::{PathFilter, Scored};
use crate::store::Store;

/// A ceiling on how many distinct terms one query contributes.
///
/// Long questions are common and each extra term costs another posting list to
/// walk; the terms past this point are almost never the ones that matter.
const MAX_TERMS: usize = 32;

/// Turn free text into an FTS5 `MATCH` expression.
///
/// FTS5 has a query language of its own — quotes, `NEAR`, `AND`/`OR`, `*`, `-`,
/// column filters — and a question typed by a human is not written in it. A bare
/// `npurag search "what about --path?"` would be a syntax error at best and a
/// silently different query at worst. So the text is taken apart into
/// alphanumeric terms and put back together as an explicit `OR` of quoted
/// literals: whatever the user typed is treated as words, never as operators.
/// Because the terms are alphanumeric by construction, quoting them cannot be
/// escaped out of.
///
/// Returns `None` for a query with no usable terms, which means there is nothing
/// to look up rather than that everything matches.
pub fn match_expression(query: &str) -> Option<String> {
    let mut terms: Vec<String> = Vec::new();
    for word in query.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let word = word.to_lowercase();
        if terms.contains(&word) {
            continue;
        }
        terms.push(word);
        if terms.len() == MAX_TERMS {
            break;
        }
    }
    if terms.is_empty() {
        return None;
    }
    let quoted: Vec<String> = terms.iter().map(|term| format!("\"{term}\"")).collect();
    Some(quoted.join(" OR "))
}

/// The best `limit` chunks for `query` by BM25, most relevant first.
pub fn candidates(
    store: &Store,
    query: &str,
    limit: usize,
    filter: Option<&PathFilter>,
) -> Result<Vec<Scored>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let Some(expression) = match_expression(query) else {
        return Ok(Vec::new());
    };

    let mut found: Vec<Scored> = Vec::with_capacity(limit);
    store.scan_lexical(&expression, |id, path, score| {
        if filter.is_some_and(|f| !f.matches(path)) {
            // Filtered out, but the scan continues: the next row may qualify.
            return true;
        }
        found.push(Scored {
            id,
            path: path.to_string(),
            score,
        });
        found.len() < limit
    })?;
    Ok(found)
}
