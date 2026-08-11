# npurag — how to use it

npurag turns a folder on your machine into a searchable index. You can then search it by
meaning rather than by exact words, and ask questions that are answered from your own
files, with the excerpts each answer came from.

Nothing is uploaded anywhere. The index is a single SQLite file on your disk, and the
model that reads your text runs on your own machine.

## Before you start

npurag does not run a model itself. It talks to a local server over HTTP, so you need one
running:

- **AMD Ryzen AI (FastFlowLM)** — `flm serve gemma3:4b --embed 1`, which listens on
  `http://localhost:52625/v1`. This is the default.
- **Intel (OpenVINO Model Server)** — listens on port 8000. Note that its API prefix may
  be `/v3` rather than `/v1`; check your version and set `base_url` accordingly.
- **Anything else that speaks the OpenAI API** — point `--base-url` at it.

Check what npurag sees:

```bash
npurag status
```

This prints the active backend, the URL it will call, whether that URL answered, and how
big the index is. If it says `unreachable`, the server is not running or is on another
port — fix that before indexing.

Want to try npurag without any server at all? Add `--mock`. It uses a small built-in
stand-in that needs no hardware. Results are crude, but every command works, which makes
it a good way to see the shape of things.

## The four commands

### Index a folder

```bash
npurag index ~/Documents
```

The first run reads everything. Later runs only re-read what changed: a file whose size
and timestamp are untouched is never opened, and one that was merely re-saved without
edits is recognised by its content and not re-processed. Running this on a timer is
therefore cheap.

Useful options:

| Option | What it does |
|---|---|
| `--reindex` | Rebuild from scratch, ignoring what the index already knows |
| `--include GLOB` | Only index files matching this pattern; can be repeated |
| `--exclude GLOB` | Skip files matching this pattern, on top of the configured excludes |
| `--max-size MB` | Skip files larger than this (default 5 MB) |
| `--follow-symlinks` | Follow symbolic links instead of skipping them |

The summary at the end reports anything skipped and why — binary files, files too large,
formats needing a parser that is not installed. A run that indexed less than you expected
will say so rather than looking successful.

### Search by meaning

```bash
npurag search "how did I set up the backup"
```

You get the best-matching passages with a score, the file each came from and a preview.
Words do not have to match: a note about "nightly archive retention" can answer a question
about backups.

Two searches actually run, and their rankings are merged. One matches by **meaning**, using
the embeddings; the other matches by **wording**, using BM25 over a full-text index. The
tag beside each score says which found it — `[v]` by meaning, `[l]` by its words, `[v+l]`
by both, with `+r` when a reranker had the final say. The two cover each other's blind
spots: meaning alone struggles with an invoice number or an error code, and wording alone
misses a paraphrase.

| Option | What it does |
|---|---|
| `-k N` | How many results to return (default 8) |
| `--path GLOB` | Only search files whose path matches, e.g. `--path '*.md'` |
| `--mode MODE` | `hybrid` (default), `vector` for meaning only, `lexical` for words only |
| `--rerank MODE` | `auto` (default), `off`, `endpoint`, `llm` — see below |
| `--json` | Machine-readable output, for scripting |

`--mode lexical` is worth knowing about for two reasons: it is the mode to reach for when
you remember the exact string, and it is the only one that needs no server at all, so it
still works when the backend is down.

### Reranking

Retrieval has to be quick, because it scores the whole index. Reranking takes the twenty
passages that survived and looks at each one against your question properly, which usually
improves the top few results.

| `--rerank` | What happens |
|---|---|
| `auto` | Rerank if the backend has a reranking model, otherwise skip it. The default. |
| `off` | Rank by retrieval score alone. |
| `endpoint` | Insist on the backend's reranking model, and fail if it has none. |
| `llm` | Score the passages with the chat model. Works on any backend, costs a generation. |

`auto` does nothing until you give a backend a `rerank_model` in the config — npurag will
not assume a third model is loaded. `npurag status` shows whether one is configured.

### Ask a question

```bash
npurag ask "what did I decide about project X?"
```

npurag finds the relevant passages, gives them to the model, and prints the answer
followed by a **Sources** section: which collection it drew on, and which file and
fragment each excerpt came from. The model is told to answer only from those excerpts and
to say so when they do not contain the answer — so treat a confident-looking answer with
no sources as a warning.

| Option | What it does |
|---|---|
| `-k N` | How many passages to draw on (default 8) |
| `--path GLOB` | Only draw on files whose path matches |
| `--mode MODE` | How the passages are found; same modes as `search` |
| `--rerank MODE` | How the shortlist is reranked; same modes as `search` |
| `--model NAME` | Use a different chat model for this question |
| `--no-sources` | Print only the answer |
| `--json` | Answer, sources and origin as JSON |

Ask in whatever language you like; the model is instructed to answer in the language of
the question.

### Keep it fresh

```bash
npurag watch ~/Documents   # re-index as files change; stop with Ctrl-C
npurag prune               # drop entries for files that no longer exist
```

`watch` waits for editing to settle before re-indexing, so saving a file once causes one
update rather than several. If you would rather not keep a program running, schedule
`npurag index` instead — on Linux the project ships systemd user units for exactly that.

## Where things live

- **Index** — `~/.local/share/npurag/<id>/index.db` on Linux,
  `%LOCALAPPDATA%\npurag\data\<id>\index.db` on Windows. One index per indexed folder.
  Deleting it loses nothing but the time to rebuild.
- **Config** — `~/.config/npurag/config.toml` on Linux,
  `%APPDATA%\npurag\config\config.toml` on Windows. It is optional; the defaults work.

A config file looks like this:

```toml
backend = "amd-flm"          # which preset below to use

max_file_size_mb = 5
chunk_tokens     = 400       # roughly how big each indexed passage is
chunk_overlap    = 60        # how much neighbouring passages share
exclude = [".git/**", "node_modules/**", "target/**", "**/*.min.js"]
external_extractors = true   # may call pdftotext / pandoc if installed

[search]
mode           = "hybrid"    # hybrid | vector | lexical
rerank         = "auto"      # auto | off | endpoint | llm
rerank_top     = 20          # how many passages the reranker is shown
candidates     = 0           # per-search candidates before merging; 0 = from -k
rrf_k          = 60.0        # how flatly the two rankings are weighed against each other
vector_weight  = 1.0         # raise to trust meaning more
lexical_weight = 1.0         # raise to trust exact wording more

[backends.amd-flm]
base_url    = "http://localhost:52625/v1"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b"

[backends.intel-ovms]
base_url    = "http://localhost:8000/v3"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b-int4-ov"
# rerank_model = "bge-reranker-base"   # if your server has one
```

Switch backends per command with `--backend intel-ovms`, or override just the address with
`--base-url`. Any of `NPURAG_BACKEND`, `NPURAG_BASE_URL`, `NPURAG_EMBED_MODEL`,
`NPURAG_CHAT_MODEL`, `NPURAG_RERANK_MODEL` and `NPURAG_DB` work as environment variables
too. Command-line flags win over environment variables, which win over the config file.

## Which file types

Text, source code and Markdown always work. PDF, HTML and Office documents (DOCX, PPTX,
XLSX, ODT, ODP) work in the downloadable builds, which include those parsers. Anything
npurag cannot read is skipped and counted, never silently dropped.

If `pdftotext` or `pandoc` are installed, npurag will use them for formats it cannot read
itself, including older ones like `.doc` and `.ods`. Both are local programs; set
`external_extractors = false` if you would rather npurag never started another process.

## When something looks wrong

**`status` says the backend is unreachable.** The server is not running, or it is on a
different port, or its API prefix differs — OpenVINO Model Server often uses `/v3` where
FastFlowLM uses `/v1`. The URL shown by `status` is exactly what npurag will call.

**"this index was built with embedding model X".** An index only makes sense alongside the
model that built it; measurements from two different models are not comparable. Run
`npurag index <folder> --reindex` to rebuild with the model you have configured now.

**"no index for … yet".** `search` and `ask` use the index for the folder you are currently
in. Either `cd` into the folder you indexed, or pass `--db` with the path to its index.

**Results seem poor.** Try a longer, more specific question — a full sentence gives far
more to match on than a single word. If you remember the exact wording, `--mode lexical`
searches for it literally; if you remember only the gist, `--mode vector` ignores wording
entirely. `--rerank llm` will take a closer look at the shortlist, at the cost of a
generation. Passage size is adjustable with `chunk_tokens`, and `--path` narrows the search
when you know roughly where the answer lives.

**An index built by an older version.** It is upgraded in place the first time you open it:
the full-text half is built from text the index already holds, so nothing is sent to the
backend and nothing needs re-embedding.

---

Licensed under the Apache License 2.0. Issues and questions:
<https://github.com/antumbra-ai/npurag/issues>
