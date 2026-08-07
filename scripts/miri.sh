#!/usr/bin/env bash
# Runs the unit, API, model and feature tests under Miri with strict
# provenance. Required before pushing (CI enforces it). Needs a nightly
# toolchain (development only -- using the crate needs stable):
#   rustup toolchain install nightly --component miri
set -euo pipefail
cd "$(dirname "$0")/.."
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-strict-provenance}"
export PROPTEST_CASES="${PROPTEST_CASES:-4}"
cargo +nightly miri test --all-features --lib "$@"
cargo +nightly miri test --all-features --test basic "$@"
# proptest calls getcwd during startup, which Miri's isolation forbids
# (persistence is already off under Miri; the call happens regardless), so
# this one leg runs with isolation disabled, strict provenance kept.
MIRIFLAGS="$MIRIFLAGS -Zmiri-disable-isolation" cargo +nightly miri test --all-features --test model "$@"
cargo +nightly miri test --all-features --test features "$@"
