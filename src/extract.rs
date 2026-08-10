//! Turning a file on disk into indexable text.
//!
//! M1 covers what Rust handles without argument: plain text, code and Markdown.
//! Formats that need a real parser are recognised and reported as skipped rather
//! than silently mangled; the extractors for them arrive in M4 behind feature
//! flags.

use std::path::Path;

/// How many leading bytes are inspected when deciding whether a file is binary.
const SNIFF_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Extraction {
    Text(String),
    Skipped(SkipReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// A NUL byte near the start; almost certainly not text.
    Binary,
    /// A format npurag knows about but cannot read yet.
    UnsupportedFormat,
}

/// Extensions that carry text but need a dedicated parser (M4).
const NEEDS_EXTRACTOR: &[&str] = &[
    "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "odt", "odp", "ods", "rtf", "epub",
];

pub fn extract(path: &Path, bytes: &[u8]) -> Extraction {
    if needs_extractor(path) {
        return Extraction::Skipped(SkipReason::UnsupportedFormat);
    }
    if looks_binary(bytes) {
        return Extraction::Skipped(SkipReason::Binary);
    }
    Extraction::Text(String::from_utf8_lossy(bytes).into_owned())
}

pub fn needs_extractor(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| NEEDS_EXTRACTOR.contains(&e.as_str()))
}

pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(SNIFF_BYTES).any(|b| *b == 0)
}
