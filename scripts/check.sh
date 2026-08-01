#!/usr/bin/env bash
# Fast pre-commit gate (about a minute): formatting, strict (pedantic) clippy,
# tests and docs in both feature configurations, plus a 32-bit build check.
# Everything here runs on stable Rust. The optional nightly-only extras
# (scripts/miri.sh, scripts/sanitize.sh) are not part of this gate.
set -euo pipefail
cd "$(dirname "$0")/.."

export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "cargo fmt --check"
cargo fmt --all --check

step "clippy, all features (pedantic, warnings are errors)"
cargo clippy --all-features --all-targets -- -D warnings

step "clippy, no features"
cargo clippy --no-default-features --all-targets -- -D warnings

step "tests, all features"
cargo test --all-features

step "tests, no features"
cargo test --no-default-features

step "docs"
cargo doc --all-features --no-deps

step "32-bit (wasm32): compile the crate and every test binary"
if rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
  cargo test --all-features --target wasm32-unknown-unknown --no-run
  cargo clippy --all-features --all-targets --target wasm32-unknown-unknown -- -D warnings
else
  echo "skipped: install with 'rustup target add wasm32-unknown-unknown'"
fi

printf '\n\033[1;32mAll checks passed.\033[0m\n'
