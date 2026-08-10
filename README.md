# npurag

**On-device semantic search & RAG for any folder — running on your NPU.**

`npurag` turns a directory on your machine into an always-fresh semantic index you can
search by meaning and ask questions against — fully local, private, and energy-efficient.
Embeddings and generation run on your laptop's NPU (AMD Ryzen AI via FastFlowLM, or any
OpenAI-compatible backend), so continuous indexing costs just a few watts and never leaves
your device.

npurag is the **memory layer** of **Antumbra** — an open, local-first AI platform where the
NPU is the always-on background tier and your GPU or the cloud handle the heavy generation.

> **Status: early / planning.** No code yet — the design is being built out in milestones
> (M0–M6).

## What it does

- `npurag index <dir>` — index a whole folder (any file types), incrementally.
- `npurag search "<query>"` — semantic search over your files.
- `npurag ask "<question>"` — answer questions grounded in your files (RAG).

## Requirements

- Rust (stable).
- A running OpenAI-compatible backend:
  - **AMD Ryzen AI** — FastFlowLM (`flm serve … --embed 1`, default `localhost:52625`).
  - **Intel (Lunar Lake)** — OpenVINO Model Server (default `localhost:8000`).
- No NPU? Development and tests run against a built-in **mock backend** — no hardware required.

## Build

Built in milestones (M0–M6): skeleton & backend → index → search → ask (RAG) →
extractors → freshness → scale.

```bash
cargo build
cargo test        # runs against the mock backend — no NPU needed
```

## Contributing

Bug reports and backend compatibility reports are welcome — see
[`CONTRIBUTING.md`](./CONTRIBUTING.md). Commits must be signed off (`git commit -s`).

## License

Licensed under the [Apache License 2.0](./LICENSE). See [`NOTICE`](./NOTICE) for
attribution requirements when redistributing.
