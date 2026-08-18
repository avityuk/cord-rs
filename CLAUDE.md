# cord-rs — notes for Claude Code sessions

Rust port of abseil's `absl::Cord` (source of truth for the rep layer:
`~/Projects/abseil-cpp/absl/strings/{cord.h,cord.cc,internal/cord_*}`) — a
port with changes, not a fidelity claim. Design decisions: typed-handle rep
layer (RepRef/OwnedRep/RepView/UniqueRep over a raw-pointer btree-surgery
core), idiomatic `bytes`-style API, no Cordz/CRC, panic on out-of-range,
64+32-bit / LE+BE. Tree shape may diverge from abseil's when that minimizes
the tree or is neutral — properties (O(1) clone, cheap slicing, balance,
sharing) are the contract, not the shape. See CONTRIBUTING.md for the
layout, conventions and the full command reference.

## Before every commit

```sh
scripts/check.sh   # fmt, pedantic clippy (both feature configs), tests (both), docs, wasm32 build
```

Zero clippy warnings is the bar (`clippy::pedantic` is enabled crate-wide).
Install the hook with `git config core.hooksPath scripts/hooks` so this runs
automatically.

## Before pushing

```sh
scripts/miri.sh        # Miri, strict provenance (required; nightly)
scripts/sanitize.sh    # ASan + TSan (required; nightly + rust-src)
```

Nightly is required for validation only; the crate itself must keep building
and being usable on stable (MSRV 1.95).

## Conventions that matter here

- Rep functions adopt/transfer references exactly like abseil; validate trees
  in tests with `cord_rs::__internal::validate` / `dump`.
- Derive data pointers from allocation pointers, not header references; derive
  pointers into owned buffers after the owner is at its final address.
- Justify casts with `#[expect(clippy::..., reason = "...")]`, not `allow`.
- Heavy test loops must be scaled down under `cfg(miri)`.
