# cord-rs

A Rust port of abseil's [`absl::Cord`][absl-cord]: a rope-like byte sequence
with O(log n) append, prepend and slicing, O(1) cloning, and a 16 byte
footprint with 15 bytes of inline storage.

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
let mut buffer = CordBuffer::with_default_limit(1024);
buffer.put_slice(b" tail");
cord.append(buffer);

// Iterate chunks for bulk processing, bytes for convenience.
let total: usize = cord.chunks().map(<[u8]>::len).sum();
assert_eq!(total, cord.len());
assert!(cord.ends_with(" tail"));
```

## When to use a `Cord`

`Cord` is designed for large byte sequences that change over their lifetime
or are shared across API boundaries: wire-format messages that get headers
prepended or payloads appended, buffers assembled from many pieces, or data
that is sliced and passed around without copying. For small, contiguous,
rarely modified data prefer `Vec<u8>` or `bytes::Bytes`.

## What is ported

The representation and the optimizations of `absl::Cord` are preserved:

| abseil                                   | cord-rs                                              |
| ---------------------------------------- | ---------------------------------------------------- |
| 16 byte `Cord`, 15 byte inline data      | `size_of::<Cord>() == 16`, 15 byte inline data       |
| Tag byte dispatch instead of vtables     | Same node layouts (`FLAT`, `EXTERNAL`, `SUBSTRING`, `BTREE`) |
| Size-classed flat buffers (32 B – 256 KiB) | Identical size classes and tag encoding             |
| B-tree of 6 edges / 64 bytes per node    | Same fan-out, node size and max height (12)          |
| In-place append into private buffers     | `append` reuses spare capacity when unshared         |
| Copy-vs-share threshold (511 bytes)      | Same threshold for cords, `Vec<u8>`, `String`, `Arc` |
| Amortized 10 % growth of new flats       | Same                                                 |
| `CordBuffer` with SSO                    | `CordBuffer` (15 byte inline, `Vec`-style uninit API) |
| `GetAppendBuffer`                        | `take_append_buffer`                                 |
| `EstimatedMemoryUsage` (3 modes)         | `estimated_memory_usage(MemoryAccounting)`           |

Not ported: the Cordz sampling / profiling layer and the CRC checksum node.

### API mapping

The public API is idiomatic Rust, modeled on the [`bytes`] crate where a
convention exists:

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
| `Subcord(pos, n)`                     | `slice(range)`, `try_slice`, `split_off`, `split_to` |
| `TryFlat` / `Flatten`                 | `as_flat` / `flatten`                           |
| `Chunks()` / `Chars()`                | `chunks()` / `bytes()`                          |
| `CharIterator` + `AdvanceAndRead`     | `cursor()` + `Cursor::read`                     |
| `Compare` / `StartsWith` / ...        | `compare`, `starts_with`, `ends_with`, `contains`, `find`, `PartialEq`/`Ord` |
| `operator[]`                          | `cord[i]`, `get(i)`                             |
| `CopyCordToString` / `CopyCordToSpan` | `to_vec`, `copy_prefix_to`, `io::Read` on `Cursor` |
| `AbslFormatFlush` / `operator<<`      | `fmt::Write`, `io::Write`, `Display`            |

Bounds are checked like `Vec` / `bytes::Bytes`: out-of-range indices and
ranges panic; `get` and `try_slice` are the non-panicking forms.

## Features

- `bytes`: `bytes::Buf` for `Cord` and `Cursor`, `bytes::BufMut` for
  `CordWriter`, and zero-copy conversions with `bytes::Bytes`.
- `serde`: `Serialize` / `Deserialize` for `Cord` as a byte sequence.

## Safety and testing

The core is a faithful port of abseil's raw-pointer, reference-counted tree
and is therefore `unsafe` internally; the public API is safe. Verification:

- abseil's own test suites are ported: `cord_test.cc`, `cord_buffer_test.cc`,
  `cord_rep_btree_test.cc`, `cord_rep_btree_navigator_test.cc`,
  `cord_rep_btree_reader_test.cc` and `cord_data_edge_test.cc`.
- A [proptest] model test runs random operation sequences (append, prepend,
  slice, split, advance, truncate, clone, flatten, cursor reads, ...) on
  several cords sharing structure, against `Vec<u8>` oracles, validating the
  tree after every step.
- The unit and API tests run under [Miri] with strict provenance
  (`scripts/miri.sh`) and under AddressSanitizer / ThreadSanitizer
  (`scripts/sanitize.sh`).
- 64-bit and 32-bit targets, little and big endian, are supported (CI covers
  `i686`, `wasm32` and a `powerpc64` build).

Benchmarks: `cargo bench` (Criterion).

## Minimum supported Rust version

1.95 (edition 2024).

## License

Apache-2.0, like abseil. See `LICENSE` and `NOTICE`; the algorithms, data
layouts and much of the documentation derive from the Abseil C++ Common
Libraries, Copyright The Abseil Authors.

[absl-cord]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
[`bytes`]: https://crates.io/crates/bytes
[proptest]: https://crates.io/crates/proptest
[Miri]: https://github.com/rust-lang/miri
