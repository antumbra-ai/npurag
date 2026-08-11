# npurag

[![CI](https://github.com/antumbra-ai/npurag/actions/workflows/ci.yml/badge.svg)](https://github.com/antumbra-ai/npurag/actions/workflows/ci.yml)

**On-device semantic search & RAG for any folder — running on your NPU.**

`npurag` turns a directory on your machine into an always-fresh semantic index you can
search by meaning and ask questions against — fully local, private, and energy-efficient.
Embeddings and generation run on your laptop's NPU (AMD Ryzen AI via FastFlowLM, or any
OpenAI-compatible backend), so continuous indexing costs just a few watts and never leaves
your device.

npurag is the **memory layer** of **Antumbra** — an open, local-first AI platform where the
NPU is the always-on background tier and your GPU or the cloud handle the heavy generation.

> **Status: early.** Indexing, search and RAG work today against any OpenAI-compatible
> backend, and against a built-in mock with no hardware at all. Not yet exercised on real
> NPU silicon — reports welcome.

New here? [`USAGE.md`](./USAGE.md) is the full manual — every command, the config file and
what to do when something looks wrong. Po polsku: [`USAGE_PL.md`](./USAGE_PL.md).

## What it does

- `npurag index <dir>` — index a whole folder (any file types), incrementally.
- `npurag search "<query>"` — hybrid search over your files: meaning (embeddings) and
  wording (BM25) ranked together, so an invoice number is as findable as a paraphrase.
  Optionally reranked, by a reranking model or by the chat model itself.
- `npurag ask "<question>"` — answer questions grounded in your files (RAG), with the
  excerpts each answer was built from.
- `npurag watch <dir>` — re-index as files change; `npurag prune` drops entries whose
  files are gone.
- `npurag mcp <dir>` — serve the index to an assistant over the Model Context Protocol,
  on stdin and stdout. No port, no daemon: the client launches it as a child process.
- `npurag serve <dir>` — the same searches over HTTP, for callers that are not an
  assistant. Loopback by default, and it refuses to bind wider without a token.

For a scheduled refresh instead of a running process, see the systemd user units in
[`contrib/systemd`](./contrib/systemd).

## Requirements

- Rust (stable).
- A running OpenAI-compatible backend:
  - **AMD Ryzen AI** — FastFlowLM (`flm serve … --embed 1`, default `localhost:52625`).
  - **Intel (Lunar Lake)** — OpenVINO Model Server (default `localhost:8000`).
- No NPU? Development and tests run against a built-in **mock backend** — no hardware required.

## Install

Prebuilt downloads are attached to each [release](https://github.com/antumbra-ai/npurag/releases):

- **Linux** — one statically linked binary, no dependencies and no installer:

  ```bash
  curl -LO https://github.com/antumbra-ai/npurag/releases/latest/download/npurag-<version>-linux-x86_64
  chmod +x npurag-<version>-linux-x86_64
  sudo mv npurag-<version>-linux-x86_64 /usr/local/bin/npurag
  ```

- **Windows** — run the `-setup.exe` installer. It installs per user, needs no
  administrator, and can add npurag to your `PATH`.

Every file ships with a `.sha256` beside it. Both builds include the PDF, HTML and Office
extractors, since a downloaded binary cannot have Cargo features turned on afterwards.

## Build

Built in milestones (M0–M9): skeleton & backend → index → search → ask (RAG) →
extractors → freshness → scale → hybrid retrieval and reranking → MCP server →
HTTP endpoint.

```bash
cargo build
cargo test                 # runs against the mock backend — no NPU needed
cargo install --path .     # installs npurag to ~/.cargo/bin
```

Text, code and Markdown are indexed out of the box. PDF, HTML and Office documents need
their parsers compiled in, since they pull in heavy dependencies most indexes do not want:

```bash
cargo build --features extractors      # or: --features pdf,html,office
```

Without them, those files are skipped and counted rather than failing the run. npurag can
also fall back to `pdftotext` or `pandoc` when they are installed locally; set
`external_extractors = false` in the config to stop it spawning any process at all.

| Feature | What it adds |
|---|---|
| `pdf` | PDF text extraction |
| `html` | HTML to readable text |
| `office` | DOCX, PPTX, XLSX, ODT, ODP |
| `extractors` | all three of the above |
| `simd` | simsimd kernels for cosine similarity |

Search is an exact scan over every stored vector. That holds up well past fifty thousand
chunks — the test suite builds an index that size and searches it — so `simd` is a knob
for unusually large collections rather than something you need.

## Contributing

Bug reports and backend compatibility reports are welcome — see
[`CONTRIBUTING.md`](./CONTRIBUTING.md). Commits must be signed off (`git commit -s`).

## License

Licensed under the [Apache License 2.0](./LICENSE). See [`NOTICE`](./NOTICE) for
attribution requirements when redistributing.
