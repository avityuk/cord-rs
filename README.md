# cord-rs

A rope-like byte sequence for building, slicing and sharing large binary
data: O(log n) append, prepend and slicing, O(1) cloning, and a 16-byte
handle that stores up to 15 bytes inline without allocating.

```toml
[dependencies]
cord-rs = "0.1"
```

```rust
use cord_rs::{Cord, CordBuffer};

// Small values live inline; no allocation.
let mut cord = Cord::from("hello");

// Append / prepend / slice are O(log n) and share memory.
cord.append(" world");
cord.prepend(">> ");
let world = cord.slice(9..);
assert_eq!(world, "world");

// Large owned buffers are adopted, not copied.
cord.append(vec![b'!'; 4096]);

// Zero-copy building through CordBuffer (e.g. read from a socket).
let mut buffer = CordBuffer::with_capacity(1024);
buffer.put_slice(b" tail");
cord.append(buffer);

// Iterate chunks for bulk processing, bytes for convenience.
let total: usize = cord.chunks().map(<[u8]>::len).sum();
assert_eq!(total, cord.len());
assert!(cord.ends_with(" tail"));
```

## Why a cord?

`Vec<u8>` concatenation, insertion and slicing are O(n) copies; sharing means
cloning. A `Cord` stores bytes either inline or in a reference-counted B-tree
of immutable buffers, so it is designed for byte sequences that change over
their lifetime or cross API boundaries:

- **Cheap edits**: `append`, `prepend`, `slice`, `split_off`/`split_to`,
  `advance`, `truncate` are O(log n) and reuse existing buffers — e.g.
  prepending a header to a wire-format message, or repeatedly appending
  payloads.
- **Cheap sharing**: `clone` is a reference-count bump; sub-cords share the
  underlying memory.
- **Zero-copy ingestion**: large `Vec<u8>`, `String`, `Box<[u8]>`,
  `Arc<[u8]>`, `&'static [u8]`, `Cow::Owned` and `bytes::Bytes` values are
  adopted rather than copied; `CordBuffer` lets I/O write into memory that
  becomes part of the cord directly.
- **Small-data friendly**: values up to 15 bytes live inline in the 16-byte
  handle; small appends fill spare capacity in existing buffers.

For small, contiguous, rarely modified data, prefer `Vec<u8>` or
`bytes::Bytes` — cords pay indirection for their flexibility, and random
access (`cord[i]`) is O(log n).

## API sketch

Naming follows the [`bytes`] crate where a convention exists: `append` /
`prepend` (accepting slices, strings, cords, owned buffers — see
[`CordSource`]), `advance`, `truncate`, `slice`, `split_off`,
`split_to`, `as_contiguous`, `make_contiguous`, `chunks()`, `bytes()`, `cursor()` (chunked
reading, skipping, sub-cord extraction), `find`, `starts_with` / `ends_with` /
`contains`, `compare` plus `PartialEq`/`Ord` against slices, strings
and cords, `Hash` (chunk-layout independent), `Index`, `Extend`,
`FromIterator`, `io::Write`/`fmt::Write`, and `take_append_buffer` for
reusing a cord's spare capacity. Out-of-range indices and ranges panic, like
`Vec` and `bytes::Bytes`; `get` covers both.

Cargo features:

- `bytes`: `bytes::Buf` for `Cord` and `Cursor`, `bytes::BufMut` for
  `CordWriter` and `CordBuffer`, and zero-copy conversions with
  `bytes::Bytes`.
- `serde`: `Serialize` / `Deserialize` for `Cord` as a byte sequence.

The crate has one required dependency, [`memchr`], used for substring search
(`find`, `contains`, `starts_with`/`ends_with`).

## Relationship to abseil's `absl::Cord`

`cord-rs` is a port of [`absl::Cord`][absl-cord] — the same data structure
Google uses for protobuf `bytes`/`string` fields and RPC payloads — with some
changes along the way: an idiomatic Rust API, and internals that started from
abseil's design (16-byte handle with inline storage, size-classed buffers, a
shallow b-tree, the same sharing thresholds and growth policy) and evolve
independently where that makes the crate simpler or the trees smaller. The
Cordz sampling / profiling layer and the CRC checksum node are not ported.

<details>
<summary>abseil → cord-rs API mapping</summary>

| abseil                                | cord-rs                                         |
| ------------------------------------- | ----------------------------------------------- |
| `Cord(string_view)`                   | `Cord::from(&[u8])`, `From<&str>`, `copy_from_slice` |
| `Cord(std::string&&)`                 | `From<Vec<u8>>`, `From<String>`, `From<Box<[u8]>>`, `From<Arc<[u8]>>` |
| `MakeCordFromExternal(...)`           | `Cord::from_static`, `From<Arc<[u8]>>`, `From<bytes::Bytes>` |
| `Append` / `Prepend`                  | `append` / `prepend` (any [`CordSource`])       |
| `Append(CordBuffer)`                  | `append(buffer)`                                |
| `GetAppendBuffer`                     | `take_append_buffer`, `take_append_buffer_with` |
| `RemovePrefix(n)`                     | `advance(n)`                                    |
| `RemoveSuffix(n)`                     | `truncate(len - n)`                             |
| `Subcord(pos, n)`                     | `slice(range)`, `get(range)`, `split_off`, `split_to` |
| `TryFlat` / `Flatten`                 | `as_contiguous` / `make_contiguous`             |
| `Chunks()` / `Chars()`                | `chunks()` / `bytes()`                          |
| `CharIterator` + `AdvanceAndRead`     | `cursor()` + `Cursor::read_cord`                |
| `Compare` / `StartsWith` / ...        | `compare`, `starts_with`, `ends_with`, `contains`, `find`, `PartialEq`/`Ord` |
| `operator[]`                          | `cord[i]`, `get(i)`                             |
| `CopyCordToString` / `CopyCordToSpan` | `to_vec`, `copy_prefix_to`, `io::Read`/`Seek` on `Cursor` |
| `AbslFormatFlush` / `operator<<`      | `fmt::Write`, `io::Write`                       |

</details>

## Safety and testing

The core is a reference-counted tree in the abseil tradition and uses
`unsafe` internally; the public API is safe. Verification:

- abseil's own test suites are ported: `cord_test.cc`, `cord_buffer_test.cc`,
  `cord_rep_btree_test.cc`, `cord_rep_btree_navigator_test.cc`,
  `cord_rep_btree_reader_test.cc` and `cord_data_edge_test.cc`.
- A [proptest] model test runs random operation sequences (append, prepend,
  slice, split, advance, truncate, clone, flatten, cursor reads, ...) on
  several cords sharing structure, against `Vec<u8>` oracles, validating the
  tree after every step.
- The test suite runs under [Miri] with strict provenance and under
  AddressSanitizer / ThreadSanitizer; both are required CI jobs.
- 64-bit and 32-bit targets, little and big endian, are supported (CI covers
  `i686`, `wasm32` and a `powerpc64` build).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). `scripts/check.sh` is the pre-commit
gate (rustfmt; pedantic clippy, checks and docs with warnings as errors
across all four feature combinations; tests in both feature configurations;
and a 32-bit compile of the crate and tests);
`scripts/miri.sh` and `scripts/sanitize.sh` are the required pre-push
soundness checks (nightly — a development-only requirement); `cargo bench`
runs the Criterion benchmarks.

## Minimum supported Rust version

1.95 (edition 2024) to build and use the crate; validation tooling (Miri,
sanitizers) additionally needs a nightly toolchain.

## License

Apache-2.0, like abseil. See `LICENSE` and `NOTICE`; the algorithms, data
layouts and much of the documentation derive from the Abseil C++ Common
Libraries, Copyright The Abseil Authors.

[absl-cord]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
[`bytes`]: https://crates.io/crates/bytes
[`memchr`]: https://crates.io/crates/memchr
[`CordSource`]: https://docs.rs/cord-rs/latest/cord_rs/trait.CordSource.html
[proptest]: https://crates.io/crates/proptest
[Miri]: https://github.com/rust-lang/miri
