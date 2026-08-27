#!/usr/bin/env bash
# Runs the unit, API, model and feature tests under Miri with strict
# provenance, plus a release-mode rerun of the unit tests (see below).
# Required before pushing (CI enforces it); takes about 35-45 min. Needs a
# nightly toolchain (development only -- using the crate needs stable):
#   rustup toolchain install nightly --component miri
set -euo pipefail
cd "$(dirname "$0")/.."
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-strict-provenance}"
export PROPTEST_CASES="${PROPTEST_CASES:-4}"
cargo +nightly miri test --all-features --lib "$@"
cargo +nightly miri test --all-features --test cord "$@"
cargo +nightly miri test --all-features --test cord_buffer "$@"
# proptest calls getcwd during startup, which Miri's isolation forbids
# (persistence is already off under Miri; the call happens regardless), so
# this one leg runs with isolation disabled, strict provenance kept.
MIRIFLAGS="$MIRIFLAGS -Zmiri-disable-isolation" cargo +nightly miri test --all-features --test model "$@"
cargo +nightly miri test --all-features --test features "$@"
cargo +nightly miri test --all-features --test panics "$@"

# Required: a release-mode rerun of the unit tests, which exercise the rep
# layer's debug_assert!-guarded contracts. In every leg above, a violated
# debug_assert! panics before Miri ever reaches the unsafe operation it
# guards, so the actual undefined behavior stays invisible to debug Miri;
# release mode compiles those assertions away, so Miri reports the UB
# itself instead of a panic.
cargo +nightly miri test --release --all-features --lib "$@"

# Optional: additionally execute the lib and public-API (`tests/cord/`,
# `tests/cord_buffer.rs`) legs under Miri for a second, foreign target -- e.g.
# a big-endian one, to interpret (not just build) src/buffer.rs's
# cfg(target_endian)-conditional layout, the crate's only one. Miri can run
# foreign targets without that target's toolchain component installed, since
# it interprets rather than compiles for it. Off by default: it roughly
# doubles the run, so it is not part of the required set above; opt in
# locally with, e.g.:
#   MIRI_TARGET=s390x-unknown-linux-gnu scripts/miri.sh
if [[ -n "${MIRI_TARGET:-}" ]]; then
  cargo +nightly miri test --all-features --lib --target "$MIRI_TARGET" "$@"
  cargo +nightly miri test --all-features --test cord --target "$MIRI_TARGET" "$@"
  cargo +nightly miri test --all-features --test cord_buffer --target "$MIRI_TARGET" "$@"
fi
