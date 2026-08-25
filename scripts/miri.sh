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

# Optional: additionally execute the lib and public-API (`tests/basic.rs`)
# legs under Miri for a second, foreign target -- e.g. a big-endian one, to
# interpret (not just build) src/buffer.rs's cfg(target_endian)-conditional
# layout, the crate's only one. Miri can run foreign targets without that
# target's toolchain component installed, since it interprets rather than
# compiles for it. Off by default: it roughly doubles the run, so it is not
# part of the required set above; opt in locally with, e.g.:
#   MIRI_TARGET=s390x-unknown-linux-gnu scripts/miri.sh
if [[ -n "${MIRI_TARGET:-}" ]]; then
  cargo +nightly miri test --all-features --lib --target "$MIRI_TARGET" "$@"
  cargo +nightly miri test --all-features --test basic --target "$MIRI_TARGET" "$@"
fi
