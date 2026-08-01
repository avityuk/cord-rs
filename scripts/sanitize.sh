#!/usr/bin/env bash
# Runs the full test suite under AddressSanitizer and ThreadSanitizer.
# Required before pushing (CI enforces it). Needs a nightly toolchain with
# rust-src (development only -- using the crate needs stable):
#   rustup toolchain install nightly --component rust-src
set -euo pipefail
cd "$(dirname "$0")/.."
target="${TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
for san in address thread; do
  echo "=== ${san} sanitizer (${target}) ==="
  RUSTFLAGS="-Zsanitizer=${san}" RUSTDOCFLAGS="-Zsanitizer=${san}" \
    cargo +nightly test -Zbuild-std --target "${target}" --all-features "$@"
done
