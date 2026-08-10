use npurag::chunk::{chunk_text, estimate_tokens, ChunkOptions};

fn options(target: usize, overlap: usize) -> ChunkOptions {
    ChunkOptions {
        target_tokens: target,
        overlap_tokens: overlap,
        max_tokens: 2048,
    }
}

#[test]
fn short_text_stays_a_single_chunk() {
    let chunks = chunk_text("one short paragraph", &ChunkOptions::default());
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "one short paragraph");
    assert_eq!(chunks[0].ord, 0);
}

#[test]
fn empty_and_blank_input_produce_nothing() {
    assert!(chunk_text("", &ChunkOptions::default()).is_empty());
    assert!(chunk_text("   \n\n  \t\n", &ChunkOptions::default()).is_empty());
}

#[test]
fn long_text_is_split_and_ordinals_are_sequential() {
    let text = (0..200)
        .map(|i| format!("line number {i} with a little padding text\n"))
        .collect::<String>();
    let chunks = chunk_text(&text, &options(100, 20));

    assert!(chunks.len() > 3, "got {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.ord, i);
        assert!(!chunk.text.is_empty());
    }
}

#[test]
fn chunks_overlap_so_a_split_passage_stays_findable() {
    let text = (0..40)
        .map(|i| format!("sentence {i} carries some words\n"))
        .collect::<String>();
    let chunks = chunk_text(&text, &options(60, 20));
    assert!(chunks.len() >= 2);

    let first_lines: Vec<&str> = chunks[0].text.lines().collect();
    let second_lines: Vec<&str> = chunks[1].text.lines().collect();
    let tail = first_lines.last().expect("first chunk has lines");
    assert!(
        second_lines.contains(tail),
        "the second chunk should repeat the tail of the first"
    );
}

#[test]
fn overlap_never_stalls_the_walk() {
    // Every line is itself larger than the overlap budget allows to carry, so a
    // naive implementation could loop forever re-emitting the same tail.
    let text = (0..30)
        .map(|i| format!("{i} {}\n", "x".repeat(200)))
        .collect::<String>();
    let chunks = chunk_text(&text, &options(60, 55));
    assert!(chunks.len() >= 2);

    let joined_len: usize = chunks.iter().map(|c| c.text.len()).sum();
    assert!(
        joined_len < text.len() * 3,
        "overlap should not multiply the corpus"
    );
}

#[test]
fn zero_overlap_repeats_nothing() {
    let text = (0..20)
        .map(|i| format!("line {i} of the document\n"))
        .collect::<String>();
    let chunks = chunk_text(&text, &options(40, 0));
    assert!(chunks.len() >= 2);

    let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(rejoined, text, "without overlap the chunks tile the input");
}

#[test]
fn a_single_oversized_line_is_broken_up_under_the_hard_ceiling() {
    let options = ChunkOptions {
        target_tokens: 50,
        overlap_tokens: 0,
        max_tokens: 100,
    };
    let text = "y".repeat(100 * 4 * 3);
    let chunks = chunk_text(&text, &options);

    assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
    for chunk in &chunks {
        assert!(
            chunk.n_tokens <= options.max_tokens,
            "chunk of {} tokens breaches the ceiling",
            chunk.n_tokens
        );
    }
}

#[test]
fn splitting_an_oversized_line_respects_character_boundaries() {
    let options = ChunkOptions {
        target_tokens: 8,
        overlap_tokens: 0,
        max_tokens: 8,
    };
    // Multi-byte throughout, so a byte-wise split would produce invalid UTF-8
    // and the concatenation below would not match.
    let text = "ąęśćżźółń".repeat(40);
    let chunks = chunk_text(&text, &options);

    let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(rejoined, text);
}

#[test]
fn token_estimate_is_the_documented_heuristic() {
    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("abcd"), 1);
    assert_eq!(estimate_tokens("abcde"), 2);
}

#[test]
fn reported_token_counts_match_the_chunk_text() {
    let text = (0..50).map(|i| format!("row {i}\n")).collect::<String>();
    for chunk in chunk_text(&text, &options(30, 5)) {
        assert_eq!(chunk.n_tokens, estimate_tokens(&chunk.text));
    }
}
