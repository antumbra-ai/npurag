//! Extractor behaviour, both when a parser is compiled in and when it is not.
//!
//! The feature-gated tests only run under `--features extractors`; the rest
//! assert the graceful-skip contract that holds in every build.

use std::path::{Path, PathBuf};

use npurag::extract::{
    extract, format_of, has_builtin_extractor, looks_binary, ExtractOptions, Extraction, Format,
    SkipReason,
};

fn docs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/docs")
}

/// Read a fixture and extract it with external tools disabled, so the result
/// depends only on what is compiled in and never on the host's installed
/// programs.
fn extract_fixture(name: &str) -> Extraction {
    let path = docs().join(name);
    let bytes = std::fs::read(&path).expect("fixture is readable");
    extract(
        &path,
        &bytes,
        &ExtractOptions {
            external_tools: false,
        },
    )
}

#[allow(dead_code)] // only the feature-gated tests below call this
fn text_of(extraction: Extraction) -> String {
    match extraction {
        Extraction::Text(text) => text,
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn formats_are_recognised_by_extension() {
    assert_eq!(format_of(Path::new("a.pdf")), Format::Pdf);
    assert_eq!(format_of(Path::new("a.PDF")), Format::Pdf);
    assert_eq!(format_of(Path::new("a.html")), Format::Html);
    assert_eq!(format_of(Path::new("a.htm")), Format::Html);
    assert_eq!(format_of(Path::new("a.docx")), Format::Office);
    assert_eq!(format_of(Path::new("a.odt")), Format::Office);
    assert_eq!(format_of(Path::new("a.doc")), Format::ExternalOnly);
    assert_eq!(format_of(Path::new("a.epub")), Format::ExternalOnly);
    assert_eq!(format_of(Path::new("a.md")), Format::Plain);
    assert_eq!(format_of(Path::new("noextension")), Format::Plain);
}

#[test]
fn plain_text_needs_no_extractor_in_any_build() {
    assert!(has_builtin_extractor(Format::Plain));
    assert_eq!(
        extract(
            Path::new("n.md"),
            "zażółć gęślą jaźń".as_bytes(),
            &ExtractOptions::default()
        ),
        Extraction::Text("zażółć gęślą jaźń".to_string())
    );
}

#[test]
fn binary_files_are_still_detected_by_content() {
    assert!(looks_binary(&[b'a', 0, b'b']));
    assert_eq!(
        extract(
            Path::new("blob.dat"),
            &[0, 1, 2],
            &ExtractOptions::default()
        ),
        Extraction::Skipped(SkipReason::Binary)
    );
}

#[test]
fn legacy_formats_are_skipped_when_no_external_tool_may_run() {
    assert!(!has_builtin_extractor(Format::ExternalOnly));
    assert_eq!(
        extract(
            Path::new("old.doc"),
            b"whatever",
            &ExtractOptions {
                external_tools: false
            }
        ),
        Extraction::Skipped(SkipReason::UnsupportedFormat)
    );
}

#[test]
fn a_missing_extractor_skips_instead_of_failing() {
    // Whatever this build supports, an unsupported format must come back as a
    // skip the indexer can count — never an error that aborts the run.
    for name in ["invoice.pdf", "meeting.docx", "page.html"] {
        let format = format_of(&docs().join(name));
        let extraction = extract_fixture(name);
        if has_builtin_extractor(format) {
            assert!(matches!(extraction, Extraction::Text(_)), "{name}");
        } else {
            assert_eq!(
                extraction,
                Extraction::Skipped(SkipReason::UnsupportedFormat),
                "{name}"
            );
        }
    }
}

#[cfg(feature = "pdf")]
#[test]
fn pdf_text_is_extracted() {
    let text = text_of(extract_fixture("invoice.pdf"));
    assert!(text.contains("importer"), "got {text:?}");
    assert!(text.contains("412"), "got {text:?}");
}

#[cfg(feature = "office")]
#[test]
fn docx_text_is_extracted() {
    let text = text_of(extract_fixture("meeting.docx"));
    assert!(text.contains("warehouse rollout"), "got {text:?}");
    assert!(text.contains("September"), "got {text:?}");
}

#[cfg(feature = "html")]
#[test]
fn html_is_reduced_to_readable_text() {
    let text = text_of(extract_fixture("page.html"));
    assert!(text.contains("Deployment runbook"), "got {text:?}");
    assert!(text.contains("Drain the queue"), "got {text:?}");
    // Markup and code must not end up in the index as content.
    assert!(!text.contains("<h1>"), "got {text:?}");
    assert!(!text.contains("var x"), "got {text:?}");
}

#[cfg(feature = "pdf")]
#[test]
fn a_corrupt_pdf_is_skipped_rather_than_fatal() {
    let extraction = extract(
        Path::new("broken.pdf"),
        b"%PDF-1.4\nthis is not really a pdf",
        &ExtractOptions {
            external_tools: false,
        },
    );
    assert!(
        matches!(extraction, Extraction::Skipped(_)),
        "got {extraction:?}"
    );
}

#[cfg(feature = "extractors")]
#[test]
fn every_document_fixture_yields_text_when_all_extractors_are_built_in() {
    for name in ["invoice.pdf", "meeting.docx", "page.html"] {
        let text = text_of(extract_fixture(name));
        assert!(!text.trim().is_empty(), "{name} produced nothing");
    }
}

#[test]
fn invalid_utf8_degrades_instead_of_failing() {
    match extract(
        Path::new("odd.txt"),
        &[b'a', 0xff, b'b'],
        &ExtractOptions::default(),
    ) {
        Extraction::Text(text) => assert!(text.contains('a') && text.contains('b')),
        other => panic!("expected lossy text, got {other:?}"),
    }
}

#[test]
fn a_nul_far_past_the_sniff_window_does_not_condemn_the_file() {
    let mut bytes = vec![b'a'; 4096];
    bytes.push(0);
    assert!(!looks_binary(&bytes));
}
