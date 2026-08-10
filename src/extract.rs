//! Turning a file on disk into indexable text.
//!
//! Plain text, code and Markdown need no help. Everything else needs a parser,
//! and those live behind Cargo features because they are heavy and most indexes
//! do not want them: build with `--features extractors` to get PDF, HTML and
//! Office documents. A format whose extractor is absent is *skipped and
//! counted*, never treated as an error — an index that quietly drops a third of
//! a directory would be worse than one that says so.
//!
//! When a built-in extractor is missing or fails, npurag can fall back to
//! `pdftotext` or `pandoc` if they happen to be installed. Both are local
//! programs, so nothing leaves the machine; set `external_extractors = false`
//! in the config to keep npurag from spawning anything at all.

use std::path::Path;
use std::process::{Command, Stdio};

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
    /// A format npurag recognises but has no extractor for in this build.
    UnsupportedFormat,
    /// An extractor ran and could not make sense of the file.
    ExtractionFailed,
}

#[derive(Debug, Clone)]
pub struct ExtractOptions {
    /// Allow falling back to `pdftotext` / `pandoc` when they are installed.
    pub external_tools: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            external_tools: true,
        }
    }
}

/// What kind of parsing a file needs, decided by extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Plain,
    Pdf,
    Html,
    /// Zip-based Office and OpenDocument files.
    Office,
    /// Legacy or exotic formats only an external converter is likely to read.
    ExternalOnly,
}

pub fn format_of(path: &Path) -> Format {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "pdf" => Format::Pdf,
        "html" | "htm" | "xhtml" => Format::Html,
        "docx" | "pptx" | "xlsx" | "odt" | "odp" => Format::Office,
        // .ods has no built-in reader in dotext, so it only ever goes to pandoc.
        "doc" | "ppt" | "xls" | "ods" | "rtf" | "epub" => Format::ExternalOnly,
        _ => Format::Plain,
    }
}

/// True when this build can read the format without an external program.
pub fn has_builtin_extractor(format: Format) -> bool {
    match format {
        Format::Plain => true,
        Format::Pdf => cfg!(feature = "pdf"),
        Format::Html => cfg!(feature = "html"),
        Format::Office => cfg!(feature = "office"),
        Format::ExternalOnly => false,
    }
}

pub fn extract(path: &Path, bytes: &[u8], options: &ExtractOptions) -> Extraction {
    match format_of(path) {
        Format::Plain => {
            if looks_binary(bytes) {
                Extraction::Skipped(SkipReason::Binary)
            } else {
                Extraction::Text(String::from_utf8_lossy(bytes).into_owned())
            }
        }
        Format::Pdf => finish(extract_pdf(path, bytes, options)),
        Format::Html => finish(extract_html(bytes)),
        Format::Office => finish(extract_office(path, options)),
        Format::ExternalOnly => finish(if options.external_tools {
            run_pandoc(path)
        } else {
            None
        }),
    }
}

/// Empty output means the extractor produced nothing usable, which is a skip
/// rather than an empty document in the index.
fn finish(text: Option<String>) -> Extraction {
    match text {
        Some(text) if !text.trim().is_empty() => Extraction::Text(text),
        Some(_) => Extraction::Skipped(SkipReason::ExtractionFailed),
        None => Extraction::Skipped(SkipReason::UnsupportedFormat),
    }
}

fn extract_pdf(path: &Path, bytes: &[u8], options: &ExtractOptions) -> Option<String> {
    #[cfg(feature = "pdf")]
    {
        // pdf-extract can panic on malformed input, and one bad file in a large
        // directory must not take the whole run down with it.
        let parsed = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem(bytes));
        if let Ok(Ok(text)) = parsed {
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
    }
    #[cfg(not(feature = "pdf"))]
    let _ = bytes;

    if options.external_tools {
        return run_tool("pdftotext", &["-q", "-layout"], path, true);
    }
    None
}

fn extract_html(bytes: &[u8]) -> Option<String> {
    #[cfg(feature = "html")]
    {
        html2text::from_read(bytes, 100).ok()
    }
    #[cfg(not(feature = "html"))]
    {
        let _ = bytes;
        None
    }
}

fn extract_office(path: &Path, options: &ExtractOptions) -> Option<String> {
    #[cfg(feature = "office")]
    {
        if let Some(text) = read_with_dotext(path) {
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
    }

    if options.external_tools {
        return run_pandoc(path);
    }
    None
}

#[cfg(feature = "office")]
fn read_with_dotext(path: &Path) -> Option<String> {
    use std::io::Read;

    use dotext::doc::OpenOfficeDoc;
    use dotext::*;

    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let mut buffer = String::new();
    let read = match extension.as_str() {
        "docx" => Docx::open(path).ok()?.read_to_string(&mut buffer),
        "pptx" => Pptx::open(path).ok()?.read_to_string(&mut buffer),
        "xlsx" => Xlsx::open(path).ok()?.read_to_string(&mut buffer),
        "odt" => Odt::open(path).ok()?.read_to_string(&mut buffer),
        "odp" => Odp::open(path).ok()?.read_to_string(&mut buffer),
        _ => return None,
    };
    read.ok().map(|_| buffer)
}

fn run_pandoc(path: &Path) -> Option<String> {
    run_tool("pandoc", &["-t", "plain"], path, false)
}

/// Run a local converter over `path` and collect its stdout.
///
/// `stdout_arg` covers tools like pdftotext that need to be told explicitly to
/// write to stdout rather than to a file next to the input.
fn run_tool(program: &str, args: &[&str], path: &Path, stdout_arg: bool) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args).arg(path);
    if stdout_arg {
        command.arg("-");
    }
    let output = command
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    (!text.trim().is_empty()).then_some(text)
}

pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(SNIFF_BYTES).any(|b| *b == 0)
}
