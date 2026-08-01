#!/usr/bin/env bash
# Runs the unit tests and the deterministic API tests under Miri with strict
# provenance checking. Requires: rustup +nightly component add miri.
set -euo pipefail
cd "$(dirname "$0")/.."
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-strict-provenance}"
export PROPTEST_CASES="${PROPTEST_CASES:-4}"
cargo +nightly miri test --all-features --lib "$@"
cargo +nightly miri test --all-features --test basic "$@"
cargo +nightly miri test --all-features --test model "$@"
cargo +nightly miri test --all-features --test features "$@"
