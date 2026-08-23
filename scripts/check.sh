#!/usr/bin/env bash
# Fast pre-commit gate (about a minute): formatting, strict (pedantic) clippy,
# checks and docs across five feature combinations, tests in three feature
# configurations, a 32-bit build check, and a bare-metal no_std check.
# Everything here runs on stable Rust. The optional nightly-only extras
# (scripts/miri.sh, scripts/sanitize.sh) are not part of this gate.
set -euo pipefail
cd "$(dirname "$0")/.."

export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Labels the empty `feature_sets` entry (default features, i.e. `std` alone)
# for step headers; every other entry is printed as `[<flags>]`.
describe_flags() {
  if [[ -z "$1" ]]; then
    printf 'default features (std)'
  else
    printf '[%s]' "$1"
  fi
}

step "cargo fmt --check"
cargo fmt --all --check

# The crate's two optional features (`bytes`, `serde`) are independent, and
# `std` is itself a default feature, so clippy/check/docs run across all five
# combinations below -- not just none and all -- to catch things (like a doc
# link that only resolves with one feature on) that combining them the naive
# way misses. The empty entry is default features (`std` alone, no `bytes` or
# `serde`) -- what most users build -- sitting between the no_std
# combinations and `--all-features`. An empty entry word-splits to nothing
# when `$flags` is expanded unquoted below, so it cleanly becomes a bare
# `cargo <cmd>` with no extra flags.
feature_sets=(
  "--no-default-features"
  "--no-default-features --features bytes"
  "--no-default-features --features serde"
  ""
  "--all-features"
)

for flags in "${feature_sets[@]}"; do
  step "clippy $(describe_flags "$flags") (pedantic, warnings are errors)"
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
  step "check $(describe_flags "$flags") (lib only, no test cfg)"
  RUSTFLAGS="${RUSTFLAGS:--D warnings}" cargo check $flags
done

step "tests, all features"
cargo test --all-features

step "tests, default features (std)"
cargo test

step "tests, no features"
cargo test --no-default-features

# rustdoc's own lints (broken intra-doc links above all) are warnings by
# default, so without `-D warnings` a dangling link would not fail this gate.
for flags in "${feature_sets[@]}"; do
  step "docs $(describe_flags "$flags") (warnings are errors)"
  RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}" cargo doc $flags --no-deps
done

# This target has `std` available (it's not `no_std` itself), so it proves
# the crate's 32-bit layout and arithmetic, not the `no_std` build -- the
# bare-metal step below does that.
step "32-bit (wasm32): compile the crate and every test binary"
if rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
  cargo test --all-features --target wasm32-unknown-unknown --no-run
  cargo clippy --all-features --all-targets --target wasm32-unknown-unknown -- -D warnings
else
  echo "skipped: install with 'rustup target add wasm32-unknown-unknown'"
fi

# `aarch64-unknown-none` has no `std` at all, so unlike wasm32 above, a lib
# check here only succeeds if the crate is genuinely usable with `core` +
# `alloc` -- this is what actually proves `no_std` support, across every
# combination of the optional features with `std` off.
step "bare metal (aarch64-unknown-none): no_std lib check"
if rustup target list --installed | grep -q '^aarch64-unknown-none$'; then
  for flags in \
    "--no-default-features" \
    "--no-default-features --features bytes" \
    "--no-default-features --features serde" \
    "--no-default-features --features bytes,serde"
  do
    RUSTFLAGS="${RUSTFLAGS:--D warnings}" cargo check --lib $flags --target aarch64-unknown-none
  done
  cargo clippy --lib --no-default-features --features bytes,serde --target aarch64-unknown-none -- -D warnings
else
  echo "skipped: install with 'rustup target add aarch64-unknown-none'"
fi

printf '\n\033[1;32mAll checks passed.\033[0m\n'
