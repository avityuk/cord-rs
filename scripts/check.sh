#!/usr/bin/env bash
# Fast pre-commit gate (about a minute): formatting, strict (pedantic) clippy,
# checks and docs across all four feature combinations, tests in both feature
# configurations, plus a 32-bit build check.
# Everything here runs on stable Rust. The optional nightly-only extras
# (scripts/miri.sh, scripts/sanitize.sh) are not part of this gate.
set -euo pipefail
cd "$(dirname "$0")/.."

export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "cargo fmt --check"
cargo fmt --all --check

# The crate's two optional features (`bytes`, `serde`) are independent, so
# clippy/check/docs run across all four combinations below -- not just none
# and all -- to catch things (like a doc link that only resolves with one
# feature on) that combining them the naive way misses.
feature_sets=(
  "--no-default-features"
  "--no-default-features --features bytes"
  "--no-default-features --features serde"
  "--all-features"
)

for flags in "${feature_sets[@]}"; do
  step "clippy [$flags] (pedantic, warnings are errors)"
  cargo clippy $flags --all-targets -- -D warnings
done

# `--all-targets` above builds the lib target too, but folded in among bins,
# examples, tests and benches; a plain `cargo check` isolates that exact
# lib-only, no-`cfg(test)` build on its own, matching what CI's dedicated
# MSRV step (`cargo check --all-features` on the pinned toolchain) runs. Keep
# both: this is a different, narrower check that fails faster and reads
# unambiguously when it does. `cargo check` has no trailing `-- <rustc args>`
# (unlike `build`/`test`/`clippy`), so `-D warnings` goes through RUSTFLAGS.
for flags in "${feature_sets[@]}"; do
  step "check [$flags] (lib only, no test cfg)"
  RUSTFLAGS="${RUSTFLAGS:--D warnings}" cargo check $flags
done

step "tests, all features"
cargo test --all-features

step "tests, no features"
cargo test --no-default-features

# rustdoc's own lints (broken intra-doc links above all) are warnings by
# default, so without `-D warnings` a dangling link would not fail this gate.
for flags in "${feature_sets[@]}"; do
  step "docs [$flags] (warnings are errors)"
  RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}" cargo doc $flags --no-deps
done

step "32-bit (wasm32): compile the crate and every test binary"
if rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
  cargo test --all-features --target wasm32-unknown-unknown --no-run
  cargo clippy --all-features --all-targets --target wasm32-unknown-unknown -- -D warnings
else
  echo "skipped: install with 'rustup target add wasm32-unknown-unknown'"
fi

printf '\n\033[1;32mAll checks passed.\033[0m\n'
