# Developing cord-rs

## Prerequisites

- Stable Rust ≥ 1.95 (edition 2024): `rustup update stable`.
- `rustup target add wasm32-unknown-unknown`, so the pre-commit gate can
  compile the crate and the tests for a 32-bit target.
- A nightly toolchain with Miri and `rust-src`, for the required pre-push
  soundness checks: `rustup toolchain install nightly --component miri rust-src`.

Nightly is a **development** requirement only: building and using the crate
needs nothing beyond stable ≥ 1.95.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate docs, re-exports, hidden `internal` test hooks |
| `src/cord.rs`, `buffer.rs`, `iter.rs`, `source.rs`, `io.rs` | Public API (`Cord`, `CordBuffer`, iterators/cursor, input & comparison traits, std io/fmt) |
| `src/inline_data.rs` | The 16-byte inline/tree union behind `Cord` |
| `src/rep.rs`, `src/rep/*.rs` | The `unsafe` rep layer ported from abseil (`flat`, `external`, `btree`, `navigator`, `reader`, `analysis`) |
| `src/rep/*_tests.rs`, `src/rep/test_util.rs` | Ports of abseil's internal test suites (unit tests: they need `pub(crate)` access) |
| `src/bytes_impl.rs`, `src/serde_impl.rs` | Feature-gated integrations |
| `tests/` | Public-API suites: `abseil_cord.rs` / `abseil_cord_buffer.rs` (ports of `cord_test.cc` / `cord_buffer_test.cc`), `basic.rs`, `model.rs` (proptest vs `Vec<u8>` oracle), `features.rs` |
| `benches/cord.rs` | Criterion benchmarks |
| `scripts/` | `check.sh` (fast gate), `miri.sh`, `sanitize.sh`, `hooks/pre-commit` |

## Before every commit

Run the fast gate — it takes about a minute and is exactly what CI's `test` and
`lint` jobs run:

```sh
scripts/check.sh
```

It runs, in order: `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings`
with all features and with no features (the crate enables `clippy::pedantic`,
so **zero warnings** is the bar); `cargo test` with all features and with no
features; `cargo doc` with `-D warnings`; and, for `wasm32`, a compile of the
crate plus every test binary and a clippy run, so 32-bit-only errors (such as
constants that only fit in 64 bits) are caught before CI's `i686` job.

To make this automatic, install the hook once per clone:

```sh
git config core.hooksPath scripts/hooks
```

`git commit --no-verify` (or `SKIP_CHECKS=1 git commit`) skips it for one
commit — use that only for work-in-progress commits on a branch.

Fix formatting with `cargo fmt`; most clippy findings can be auto-applied with
`cargo clippy --fix --all-features --all-targets`.

## Before pushing / opening a PR

The `unsafe` core must stay sound, so the soundness checks are required in
addition to the gate (CI enforces all of them; they are faster to iterate on
locally):

```sh
scripts/miri.sh        # Miri with strict provenance: unit, API, model and feature tests (~15-20 min)
scripts/sanitize.sh    # AddressSanitizer + ThreadSanitizer builds of the whole suite (~5 min)
PROPTEST_CASES=3000 cargo test --all-features --release --test model   # longer model run (~10 s)
```

Useful narrowing while iterating:

```sh
# One test under Miri (name filters work as with cargo test):
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --all-features --lib -- navigator
# Miri on the public API tests only:
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --all-features --test basic
# Loops in the internal suites are scaled down under `cfg(miri)`; keep that
# pattern when adding heavy tests.
```

## Benchmarks

```sh
cargo bench                                   # everything (several minutes)
cargo bench -- append/                        # one group (regex filter on the benchmark id)
cargo bench -- --save-baseline before         # record a baseline ...
cargo bench -- --baseline before              # ... and compare after a change
cargo bench -- --warm-up-time 0.5 --measurement-time 1.5   # quick smoke run
```

Results (with history) are under `target/criterion/`. Benchmarks are
Criterion-based and exercise construction, clone, append/prepend, slicing,
iteration, comparison, search, flattening, hashing and two "diabolical"
workloads, each with a `Vec<u8>` baseline where a comparison is meaningful.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- `test`: `cargo test` with and without features on Linux, macOS and Windows, plus a 2000-case model run in release.
- `lint`: `cargo fmt --check`, pedantic clippy (both feature configurations), docs with `-D warnings`, and an MSRV (1.95) check.
- `cross`: the full test suite on `i686` (32-bit), plus `powerpc64` (big-endian) and `wasm32` builds.
- `miri` and `sanitizers`: `scripts/miri.sh` and `scripts/sanitize.sh` on nightly. Required, like every other job.

## Conventions

- **Faithful port.** The rep layer mirrors abseil's `absl/strings/internal/cord_*`
  (layouts, constants, algorithms, reference-counting rules: functions taking a
  rep adopt a reference, functions returning one transfer it). When changing
  it, diff against the C++ source and keep the ported tests aligned; test
  names mirror the C++ test names in `snake_case`.
- **`unsafe` discipline.** All raw-pointer code lives under `src/rep/` and in the
  `unsafe` helper blocks of `cord.rs`/`iter.rs`/`buffer.rs`. Every `unsafe {}`
  block in the public layer carries a `// SAFETY:` comment. Derive data
  pointers from the allocation pointer, never from a reference to the header
  (Stacked Borrows), and derive pointers into owned buffers only after the
  owner reached its final address.
- **Lints.** `clippy::pedantic` is on. Justify deliberate casts with
  `#[expect(clippy::..., reason = "...")]` (checked helpers such as
  `small_u8`, `height_to_isize`), not blanket `allow`s. Test files may allow the
  small-integer cast lints at file level with a reason.
- **API.** Idiomatic Rust modeled on the `bytes` crate; out-of-range indices and
  ranges panic, with `get` / `try_slice` as the non-panicking forms. New public
  items need docs and, where useful, a doctest.
- **Portability.** Keep 32-bit and big-endian targets working: no assumptions
  about pointer width beyond what `InlineData`/`CordBuffer` encode explicitly.

## Releasing

1. Bump `version` in `Cargo.toml`; update the README if the API changed.
2. `scripts/check.sh && scripts/miri.sh && scripts/sanitize.sh`.
3. `cargo publish --dry-run`, then `cargo publish`.
4. Tag: `git tag -a vX.Y.Z -m "cord-rs X.Y.Z" && git push --tags`.
