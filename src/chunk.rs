//! Splitting a document into overlapping pieces small enough to embed.
//!
//! Chunks are grown line by line, so paragraph and code-block boundaries survive.
//! Token counts are the `len / 4` heuristic from the plan — good enough to stay
//! under a model's input limit, and cheap. A real tokenizer is the fallback if
//! retrieval quality suffers.

#[derive(Debug, Clone, Copy)]
pub struct ChunkOptions {
    /// Size a chunk grows towards before being closed.
    pub target_tokens: usize,
    /// How much of the previous chunk is repeated at the start of the next one,
    /// so a passage split across a boundary is still retrievable.
    pub overlap_tokens: usize,
    /// Hard ceiling; embeddinggemma-300m on FastFlowLM accepts 2048 tokens.
    pub max_tokens: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            target_tokens: 400,
            overlap_tokens: 60,
            max_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub ord: usize,
    pub text: String,
    pub n_tokens: usize,
}

/// The `len / 4` estimate. Deliberately crude; see the module docs.
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

pub fn chunk_text(text: &str, options: &ChunkOptions) -> Vec<Chunk> {
    let units = split_units(text, options.max_tokens);
    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_tokens = 0usize;

    for unit in units {
        let unit_tokens = estimate_tokens(&unit);

        if !current.is_empty() && current_tokens + unit_tokens > options.target_tokens {
            let carry = overlap_tail(&current, options.overlap_tokens);
            push_chunk(&mut chunks, &current);

            let carry_tokens: usize = carry.iter().map(|u| estimate_tokens(u)).sum();
            // Dropping the carry keeps the next chunk under the hard ceiling in
            // the rare case where overlap plus the incoming unit would breach it.
            if carry_tokens + unit_tokens > options.max_tokens {
                current = Vec::new();
                current_tokens = 0;
            } else {
                current = carry;
                current_tokens = carry_tokens;
            }
        }

        current.push(unit);
        current_tokens += unit_tokens;
    }

    push_chunk(&mut chunks, &current);
    chunks
}

fn push_chunk(chunks: &mut Vec<Chunk>, units: &[String]) {
    if units.is_empty() {
        return;
    }
    let text = units.concat();
    if text.trim().is_empty() {
        return;
    }
    chunks.push(Chunk {
        ord: chunks.len(),
        n_tokens: estimate_tokens(&text),
        text,
    });
}

/// Break the document into lines, then break apart any single line that is too
/// long to ever fit a chunk. Newlines are kept so chunks rejoin verbatim.
fn split_units(text: &str, max_tokens: usize) -> Vec<String> {
    let max_bytes = max_tokens.saturating_mul(4).max(1);
    let mut units = Vec::new();

    for line in text.split_inclusive('\n') {
        if line.len() <= max_bytes {
            units.push(line.to_string());
            continue;
        }
        let mut start = 0;
        while start < line.len() {
            // Never split inside a multi-byte character.
            let mut end = (start + max_bytes).min(line.len());
            while end < line.len() && !line.is_char_boundary(end) {
                end -= 1;
            }
            units.push(line[start..end].to_string());
            start = end;
        }
    }
    units
}

/// The trailing units whose combined size stays within the overlap budget.
fn overlap_tail(units: &[String], overlap_tokens: usize) -> Vec<String> {
    if overlap_tokens == 0 {
        return Vec::new();
    }
    let mut tail = Vec::new();
    let mut tokens = 0usize;
    for unit in units.iter().rev() {
        let unit_tokens = estimate_tokens(unit);
        if tokens + unit_tokens > overlap_tokens {
            break;
        }
        tokens += unit_tokens;
        tail.push(unit.clone());
    }
    // Carrying every unit would mean the next chunk starts where this one did,
    // and the walk would never advance.
    if tail.len() == units.len() {
        tail.pop();
    }
    tail.reverse();
    tail
}
