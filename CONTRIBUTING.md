# Contributing to npurag

Thanks for your interest. npurag is an early-stage project — the design is being built
out in milestones (M0–M6), so the most useful contributions right now are bug reports,
backend compatibility reports, and small focused patches.

## Licensing of contributions

npurag is licensed under the [Apache License 2.0](./LICENSE). As stated in section 5 of
the License:

> Unless You explicitly state otherwise, any Contribution intentionally submitted for
> inclusion in the Work by You to the Licensor shall be under the terms and conditions
> of this License, without any additional terms or conditions.

By submitting a pull request you agree that your contribution is licensed under Apache
2.0. Please do not submit code you are not entitled to license this way — in particular,
code copied from projects under a copyleft or source-available license.

If your change adds a dependency or vendored code that carries its own attribution
requirement, add it to the third-party section of [`NOTICE`](./NOTICE) in the same pull
request.

## Developer Certificate of Origin (DCO)

Every commit must be signed off, certifying that you wrote the patch or otherwise have
the right to submit it under Apache 2.0 (see the full
[DCO 1.1](https://developercertificate.org/) text).

Sign off by adding a trailer to each commit:

```bash
git commit -s -m "index: skip binary files"
```

which appends:

```
Signed-off-by: Random J Developer <random@developer.example.org>
```

The name and email must match your `git config user.name` / `user.email` and be a real
identity — anonymous or pseudonymous sign-offs are not accepted. To fix a commit you
already made, use `git commit --amend -s`; for a whole branch,
`git rebase --signoff main`.

## Development setup

No NPU is required to work on npurag. Everything runs against a built-in mock backend.

```bash
cargo build
cargo test          # mock backend — no hardware, no running server
```

Before opening a pull request, all three must be green:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

Tests must never require an NPU, a GPU, or a live inference server. If you are adding a
feature that talks to a backend, put it behind the `Backend` trait and cover it with the
mock implementation.

## Pull requests

- Keep changes small and focused; one concern per pull request.
- Match the existing style — small modules, no large framework abstractions.
- Optional file-format extractors go behind Cargo feature flags, and a missing extractor
  must degrade to skipping the file with a warning, never to a hard error.
- Do not commit index databases or build output (`*.db`, `/target`); see
  [`.gitignore`](./.gitignore).

## Reporting backend compatibility

npurag talks to any OpenAI-compatible server, so reports about backends other than
AMD FastFlowLM are especially welcome. When filing one, please include the server and
its version, the configured `base_url` (including the API prefix — `/v1` and `/v3` both
occur in the wild), the embedding and chat model names, and the output of
`npurag status`.

## Bug reports

Include your OS, `rustc --version`, the exact command you ran, what you expected, and
what happened. If the problem involves a backend, add the details listed above.
