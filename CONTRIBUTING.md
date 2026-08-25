# Developing cord-rs

## Prerequisites

- Stable Rust ≥ 1.95 (edition 2024): `rustup update stable`.
- `rustup target add wasm32-unknown-unknown`, so the pre-commit gate can
  compile the crate and the tests for a 32-bit target.
- `rustup target add aarch64-unknown-none`, so the pre-commit gate can check
  the `no_std` build against a target with no `std` at all.
- A nightly toolchain with Miri and `rust-src`, for the required pre-push
  soundness checks: `rustup toolchain install nightly --component miri rust-src`.

Nightly is a **development** requirement only: building and using the crate
needs nothing beyond stable ≥ 1.95.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate docs, re-exports, hidden `__internal` test hooks |
| `src/cord.rs`, `buffer.rs`, `iter.rs`, `source.rs`, `io.rs` | Public API (`Cord`, `CordBuffer`, iterators/cursor, input & comparison traits, std io/fmt) |
| `src/inline_data.rs` | The 16-byte inline/tree union behind `Cord` |
| `src/rep.rs`, `src/rep/*.rs` | The `unsafe` rep layer ported from abseil (`flat`, `external`, `btree`, `navigator`, `reader`, `analysis`) |
| `src/rep/*_tests.rs`, `src/rep/test_util.rs` | Ports of abseil's internal test suites (unit tests: they need `pub(crate)` access) |
| `src/bytes_impl.rs`, `src/serde_impl.rs` | Feature-gated integrations |
| `tests/` | Public-API suites: `abseil_cord.rs` / `abseil_cord_buffer.rs` (ports of `cord_test.cc` / `cord_buffer_test.cc`), `basic.rs`, `model.rs` (proptest vs `Vec<u8>` oracle), `features.rs` |
| `benches/cord.rs` | Criterion benchmarks |
| `scripts/` | `check.sh` (fast gate), `miri.sh`, `sanitize.sh`, `hooks/pre-commit` |
| `ci/no_std_bin/` | A standalone crate (own `[workspace]`, not a member of this one): a real linked bare-metal `no_std` binary, see "The `no_std` binary" below |

## Before every commit

Run the fast gate — it takes about a minute and is exactly what CI's `test` and
`lint` jobs run:

```sh
scripts/check.sh
```

It runs, in order: `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings`
(the crate enables `clippy::pedantic`, so **zero warnings** is the bar) and
`cargo check`/`cargo doc` with `-D warnings`, each across all five
combinations of the `std` default feature and the two optional features
(`bytes`, `serde`): none, `bytes` alone, `serde` alone, default features
(`std` alone — what most users build), and all features; `cargo test` with
all features, default features, and no features; and, for `wasm32`, a
compile of the crate plus every test binary and a clippy run, so 32-bit-only
errors (such as constants that only fit in 64 bits) are caught before CI's
`i686` job. `wasm32` still links `std`, though, so it proves nothing about
`no_std`; a further step checks the lib — across the four feature
combinations with `std` off — and runs clippy against
`aarch64-unknown-none`, a target with no `std` at all, so a `no_std`
regression is caught locally before CI's `no_std` job. CI's `features` job
goes further still, running `cargo hack --feature-powerset` (with
`--no-dev-deps` for the `check` leg, since `serde` here is both an optional
feature and a dev-dependency and could otherwise mask lib code that only
compiles because the dev-dependency happens to be present) across the full
feature powerset, not just the two features in isolation.

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

`src/buffer.rs` has the crate's only `cfg(target_endian)`-conditional layout,
and CI's `cross` job only *builds* big-endian (`powerpc64`), never runs it.
Miri can interpret a foreign target without that target's toolchain
component installed (it never compiles for the target, only interprets), so
`scripts/miri.sh` takes an optional `MIRI_TARGET` environment variable that
additionally runs the lib and `tests/basic.rs` legs for a second target:

```sh
MIRI_TARGET=s390x-unknown-linux-gnu scripts/miri.sh   # + big-endian lib/basic legs (~5 min more)
```

This is not part of the required set above (it would roughly double every
local pre-push run); CI runs it unconditionally instead, as a dedicated step
in the `miri` job.

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

## The `no_std` binary (`ci/no_std_bin/`)

`cargo check --lib --target aarch64-unknown-none` (what the `no_std` CI job
and `scripts/check.sh`'s bare-metal step otherwise run) proves the crate
*compiles* against `core` + `alloc` alone, but `check` never links, never
monomorphizes generic code that nothing instantiates, and never involves an
allocator — so it can't prove the crate actually *works* with no `std`.

`ci/no_std_bin/` closes that gap: a small standalone crate (its own
`[workspace]` table, so it is not a member of this repository's workspace,
is excluded from `cargo package`'s output, and is invisible to a plain
`cargo clippy --all-targets`/`cargo test` run at the root) that builds a real
`#![no_std] #![no_main]` executable for `aarch64-unknown-none`, a target with
no operating system at all. It supplies its own `#[panic_handler]` (loops)
and `#[global_allocator]` (a bump allocator over a static byte array — the
simplest allocator that is still a real one), then its entry point
(`src/exercise.rs`) exercises a broad slice of the public API — construction
from a slice and an owned `Vec`, `append`/`prepend` large enough to force the
inline → flat → btree transitions, `slice`/`get`, `chunks()`/`bytes()` sums,
comparison, `find`, `CordBuffer` `put_slice`/`append`,
`estimated_memory_usage`, `Hash`, and (behind the crate's own `bytes`/`serde`
features) `Buf`/`Bytes` and a hand-rolled `no_std` serializer — and writes a
checksum plus a failure count into a `static` the linker cannot strip, so
the optimizer cannot delete the work.

Build it (from the repository root):

```sh
cargo build --manifest-path ci/no_std_bin/Cargo.toml --target aarch64-unknown-none            # debug
cargo build --manifest-path ci/no_std_bin/Cargo.toml --target aarch64-unknown-none --release   # release
cargo build --manifest-path ci/no_std_bin/Cargo.toml --features bytes,serde --target aarch64-unknown-none            # + bytes/serde
cargo build --manifest-path ci/no_std_bin/Cargo.toml --features bytes,serde --target aarch64-unknown-none --release
```

A successful build already proves the interesting part: rustc had to
resolve every symbol `cord-rs` references (allocation, panicking, atomics)
against this crate's own `no_std` shims, with zero undefined symbols in the
linked ELF (`nm -u target/aarch64-unknown-none/release/no_std_bin`). Actually
*running* the image (e.g. under QEMU) is not part of this gate.

Its `Cargo.lock` is committed, like the crate's own.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- `test`: `cargo test` with all features, default features, and no features on Linux, macOS and Windows, plus a 2000-case model run in release.
- `lint`: `cargo fmt --check`, pedantic clippy (both feature configurations), docs with `-D warnings`, an MSRV (1.95) check (all features, no features, and no-default-features with `bytes,serde`), and, on nightly, `cargo doc` with `--cfg docsrs` (what `[package.metadata.docs.rs]` passes) so a docs.rs-only failure — the nightly-only `doc_cfg` feature that badges `#[cfg(feature = "std")]` items, say — is caught in CI instead of after publishing; the plain stable `cargo doc` step above can't reproduce it.
- `features`: `cargo hack --feature-powerset` for `check` (`--no-dev-deps`), `clippy -- -D warnings` and `doc` (`RUSTDOCFLAGS=-D warnings`), covering every combination of the optional features rather than just none/all.
- `cross`: the full test suite on `i686` (32-bit), plus `powerpc64` (big-endian) and `wasm32` builds.
- `no_std`: `cargo check --lib --no-default-features` across the four combinations of `bytes`/`serde` with `std` off, plus a clippy run with both, on `aarch64-unknown-none`, a target with no `std` at all (unlike the `wasm32` build above, which still links it) — then, on top of that, four real linked builds of `ci/no_std_bin/` (debug/release × default/`bytes,serde`; see "The `no_std` binary" above).
- `miri` and `sanitizers`: `scripts/miri.sh` and `scripts/sanitize.sh` on nightly, plus a step running the lib and `tests/basic.rs` legs under Miri for `s390x-unknown-linux-gnu` (big-endian) — see `scripts/miri.sh`'s `MIRI_TARGET` option above. Required, like every other job.

## Conventions

- **A port, not a fidelity claim.** The rep layer started from abseil's
  `absl/strings/internal/cord_*` (layouts, constants, algorithms, the
  adopt/transfer reference-counting convention: functions taking a rep adopt
  a reference, functions returning one transfer it) and stays close to it by
  default — but the crate is a port *with changes*, not a bit-for-bit clone.
  Tree *shape* may diverge from abseil's when that minimizes the tree or is
  neutral (never more nodes, sharing preserved or improved, content
  identical, `validate()` still holds); the properties — O(1) clone, cheap
  slicing, balance, sharing — are the contract, not the shape (see README's
  "Relationship to abseil" section). When changing the rep layer, diff
  against the C++ source for intent, keep the ported tests aligned, and note
  any resulting structural divergence in the commit; test names mirror the
  C++ test names in `snake_case`.
- **`unsafe` discipline.** Raw-pointer code lives under `src/rep/` and in
  `src/rep.rs`'s typed handle layer; the `unsafe` remaining in
  `cord.rs`/`iter.rs`/`buffer.rs`/`inline_data.rs` is now almost entirely
  calls into that layer's safe methods (see "Typed handles carry the
  invariant" below). `unsafe_op_in_unsafe_fn` is `deny`d, so every unsafe
  operation needs an explicit `unsafe {}` block. Derive data pointers from
  the allocation pointer, never from a reference to the header (Stacked
  Borrows), and derive pointers into owned buffers only after the owner
  reached its final address.
  - **Typed handles carry the invariant.** `src/rep.rs` defines `RepRef<'a>`
    (Copy borrowed view), `OwnedRep` (RAII refcount owner), `RepView<'a>`
    (checked tag dispatch), `UniqueRep<'a>` (refcount==1 mutation witness),
    and per-kind `FlatRef`/`ExternalRef`/`BtreeRef`. Each has exactly *one*
    `unsafe fn from_raw` constructor documenting the type's invariant
    (liveness, well-formedness, and — for `UniqueRep` — exclusivity); every
    other method on the type is safe because it needs nothing beyond that
    invariant. New code that reads or transfers a rep should go through
    these types (or add a safe method to one) rather than reach for a raw
    `*mut CordRep` / the low-level `RepPtr` trait directly — `RepPtr` still
    exists, but it's the handles' implementation detail and deep btree
    surgery's escape hatch, not a general-purpose API.
  - **`UniqueRep<'a>` is `&mut`-only.** It may only be constructed from a
    `&mut` path that has already proven exclusivity — never from a `Copy`
    `RepRef` (two copies of the same handle could each observe
    `ref_is_one()` and each mint a "unique" view of the same node). Exactly
    three call sites are permitted: `OwnedRep::try_unique(&mut self)`,
    `InlineData::tree_unique(&mut self)`, and `CordBuffer`'s internal
    `Rep::view_mut` (sound without a dynamic check because its flat is
    unconditionally exclusive by construction). A fourth call site needs the
    same `&mut`-borrow argument as the existing three, not just a passing
    refcount check.
  - **Handles vs. raw pointers.** Use the handle types for anything that
    reads a rep, transfers ownership, or mutates through a proven-exclusive
    witness. Deep btree surgery (`StackOperations`, merge/split/rebuild,
    copy_prefix/suffix) stays on raw `NonNull`/`RepPtr` on purpose: it
    tracks share-depth *dynamically* (the same node is exclusive below
    `share_depth`, shared above it), which no static witness type can
    express without lying — forcing it through a handle would be
    relabeling, not a safety win.
  - **Hot-path rule: no enum dispatch or `unwrap` in per-chunk/per-byte
    loops.** Use the debug-asserted *unchecked* downcasts there, not
    `RepRef::view()`'s checked match. This isn't hypothetical: an early
    draft ran `view()`'s tag dispatch on the clone path and cost +211% on
    `clone/inline` in A/B benchmarking; reverting to the direct tag-test
    fast path restored parity. Benchmark (`cargo bench -- <filter>` against
    a saved baseline) before landing anything on a path `benches/cord.rs`
    covers.
  - Every `unsafe fn` gets a `/// # Safety` doc section, but *centralize*:
    when several methods share an invariant (e.g. a trait's "self must point
    to a live, well-formed node"), state it once at the trait/module level
    and have individual methods reference it, spelling out only the delta a
    given method adds (an extra bound, an exclusivity requirement). Don't
    duplicate the trait's contract onto its impls — an impl method with no
    doc comment of its own is documented by its trait declaration already.
  - `// SAFETY:` comments are mandatory in safe fns that call into unsafe
    code, and in unsafe fns only where a specific operation needs a local
    argument that isn't already covered by the fn's own `# Safety` section
    (a loop invariant, an arithmetic bound, a tag-based cast, an aliasing
    argument) — never as a bare echo of the doc a few lines above. If an
    `unsafe fn`'s whole body is one block discharging exactly its own
    documented contract, it needs no block comment at all.
  - Never write `# Safety: None`. If a function genuinely has no
    precondition, either make it a safe fn that wraps its own unsafe
    operations internally (the common case for test helpers), or, if it must
    stay `unsafe` for signature uniformity with sibling methods, drop the
    `# Safety` heading and say so in a plain sentence instead.
  - Tests: SAFETY comments on ordinary `unsafe {}` blocks inside `#[test] fn`
    bodies are optional — add one only for a genuinely non-obvious setup
    (e.g. overlapping-buffer bounds); a routine "freshly allocated, unreffed
    once" note is not worth writing, the test demonstrates it by construction
    and runs under Miri.
  - **Current unsafe footprint.** Exact counts here have drifted stale twice
    already, so this note gives the shape of the result and how to measure
    it yourself, not a snapshot. Qualitatively: upper-layer call sites
    (`cord.rs`, `iter.rs`, `io.rs`, `buffer.rs`, `inline_data.rs`, `lib.rs`)
    essentially never need to *be* `unsafe fn` anymore — nearly all of that
    surface now lives in the rep layer (`rep.rs` plus `rep/*.rs`),
    concentrated exactly where the effort intended: audited once, in one
    place, with deep btree surgery (`btree.rs`) as the largest single piece.
    `unsafe {}` *block* counts in the upper layers didn't shrink the same
    way, and that's expected, not a miss — the raw operations moved rather
    than vanished, now living inside the handle constructors and
    `InlineData`'s editing helpers in `rep.rs`/`inline_data.rs`, often split
    into more, smaller, individually-justified blocks than the one big block
    they replaced.

    To reproduce the comparison, count `unsafe fn` and `unsafe {` per file —
    as a whole, including any inline `#[cfg(test)]` module it carries (e.g.
    `inline_data.rs`'s) — excluding the dedicated `*_tests.rs` /
    `test_util.rs` files under `src/rep/`:

    ```sh
    for f in src/cord.rs src/iter.rs src/io.rs src/buffer.rs src/inline_data.rs src/lib.rs \
             src/rep.rs src/rep/analysis.rs src/rep/btree.rs src/rep/external.rs src/rep/flat.rs \
             src/rep/navigator.rs src/rep/reader.rs; do
      printf '%-24s fn=%-4s block=%-4s\n' "$f" "$(grep -c 'unsafe fn' "$f")" "$(grep -c 'unsafe {' "$f")"
    done
    ```

    The first six files are the upper layers; the rest are the rep layer.
- **Lints.** `clippy::pedantic` is on. Justify deliberate casts with
  `#[expect(clippy::..., reason = "...")]` (checked helpers such as
  `small_u8`, `height_to_isize`), not blanket `allow`s. Test files may allow the
  small-integer cast lints at file level with a reason.
- **API.** Idiomatic Rust modeled on the `bytes` crate; out-of-range indices and
  ranges panic, with `get` as the non-panicking form (covering both indices and
  ranges via the sealed `CordIndex` trait). New public items need docs and,
  where useful, a doctest.
- **Portability.** Keep 32-bit and big-endian targets working: no assumptions
  about pointer width beyond what `InlineData`/`CordBuffer` encode explicitly.
- **`no_std`.** `std` is a default feature, but library code must compile
  without it — the crate is `#![no_std]` plus `alloc`. Use `core::`/`alloc::`
  paths, not `std::`, in code that isn't itself behind
  `#[cfg(feature = "std")]`. Gate any `std::io` (or other `std`-only)
  integration with `#[cfg(feature = "std")]`, paired with the
  `#[cfg_attr(docsrs, doc(cfg(feature = "std")))]` badge so docs.rs shows the
  requirement. Keep intra-doc links to `std` items out of docs that aren't
  themselves feature-gated on `std` — they dangle when the crate is
  documented without it. Integration tests that need `std` (`tests/*.rs`)
  must gate that code with `#[cfg(feature = "std")]` too, so the suite still
  builds with `--no-default-features`.

## Before a release

Manual, occasional checks — not part of day-to-day `scripts/check.sh`, and
the tools aren't installed by default:

- `cargo semver-checks check-release` — diff the public API against the
  previous published version for accidental breakage.
- `cargo public-api diff` — review the public surface for anything that
  shouldn't have changed.
- `cargo +nightly minimal-versions check --all-features` (or
  `cargo +nightly -Zminimal-versions check --all-features` if
  `cargo-minimal-versions` isn't installed) — verify the declared dependency
  floors actually build.
- `cargo package --list` — compare the packaged file list against the
  `include` allowlist in `Cargo.toml`.
- `cargo publish --dry-run` — a final packaging sanity check.

## Releasing

1. Bump `version` in `Cargo.toml`; update the README if the API changed.
2. `scripts/check.sh && scripts/miri.sh && scripts/sanitize.sh`.
3. `cargo publish --dry-run`, then `cargo publish`.
4. Tag: `git tag -a vX.Y.Z -m "cord-rs X.Y.Z" && git push --tags`.
