# cord-rs

A rope-like byte sequence for building, slicing and sharing large binary
data: O(log n) append, prepend and slicing, O(1) cloning, and a 16-byte
handle that stores up to 15 bytes inline without allocating. Works in
`no_std` environments with `alloc`.

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
- **Cheap sharing**: cloning is O(1) — inline values copy the 16-byte handle,
  while tree-backed values increment one reference count; sub-cords share the
  underlying memory.
- **Zero-copy ingestion**: large `Vec<u8>`, `String`, `Box<[u8]>`,
  `Arc<[u8]>`, `Cow::Owned` and `bytes::Bytes` values are adopted rather than
  copied, as is `&'static [u8]` via `Cord::from_static`; `CordBuffer` lets I/O
  write into memory that becomes part of the cord directly.
- **Small-data friendly**: values up to 15 bytes live inline in the 16-byte
  handle; small appends fill spare capacity in existing buffers.
- **Allocator-friendly**: chunks are allocated in size classes — 8-byte steps
  up to 512 B, 64-byte steps to 8 KiB, 4 KiB steps beyond — so a stream of
  differently-sized appends produces a handful of distinct allocation sizes
  rather than a wide spread, which keeps fragmentation and per-allocation
  slop low. Chunks are 32 B at the smallest and 4 KiB by default (64 KiB when
  a `CordBuffer` asks for a larger block). The tree over them is a B-tree
  with up to 6 edges per node, built by appending and prepending only: nodes
  fill densely and it never needs rebalancing.

For small, contiguous, rarely modified data, prefer `Vec<u8>` or
`bytes::Bytes` — cords pay indirection for their flexibility, and random
access (`cord[i]`) is O(log n).

## Sharing and mutation

A `Cord` is a value, like `String` or `Vec<u8>`: it owns what it contains, and
mutating one cord never changes another. What is unusual is how cheaply that
is implemented — the underlying buffers are reference counted and shared, and
are only copied when sharing would otherwise become visible.

- **Cloning is O(1).** Inline cords copy their 16-byte handle; tree-backed
  cords increment one reference count. Cloning never allocates or copies a
  tree-backed cord's payload.
- **Mutation is copy-on-write, at chunk granularity.** `append` writes into a
  buffer's spare capacity only when this cord is the sole owner of that buffer
  and of every tree node above it; if anything on the path is shared, the
  append allocates a new chunk and links it in instead. Shared chunks are
  never written to. The same rule governs `advance`/`truncate` (a uniquely
  owned chunk is trimmed in place, a shared one gets a new substring node
  over it) and `take_append_buffer`, which returns a fresh empty buffer
  rather than stealing a shared tail.
- **Slices share memory.** `slice`, `get(range)`, `split_off` and `split_to`
  return cords referencing the same buffers as their source; results of 15
  bytes or fewer are copied into the 16-byte handle instead. Sharing means a
  sub-cord keeps its *whole* backing buffer alive: a 90-byte slice of a single
  100 KB buffer still accounts for 100 KB, and `make_contiguous` will not
  change that (the slice is already contiguous). To release the rest, copy out
  with `Cord::from(slice.to_vec())`.
- **Thread safety.** `Cord` is `Send + Sync` and has the thread-safety of any
  Rust value: shared references from many threads are fine, mutation needs
  `&mut`. Reference counts are atomic, shared chunks are immutable, and
  external buffers must be owned by a `Send + Sync` value, so passing a clone
  to another thread needs no lock and no copy.
- **Accounting.** `cord.estimated_memory_usage(mode)` reports approximate
  bytes held, with three modes: `Total` charges every cord for all memory it
  can reach (two cords sharing one 8 KiB buffer are charged ~8 KiB each),
  `FairShare` divides shared memory by the number of sharers (~4 KiB each),
  and `TotalMorePrecise` counts memory reachable twice from the *same* cord
  only once, at the cost of deduplicating every reference.

```rust
use cord_rs::{Cord, MemoryAccounting};

let mut a = Cord::from(vec![b'x'; 4096]);
let b = a.clone();          // no copy: a refcount bump
a.append("!");              // `b` is unaffected
assert_eq!(b.len(), 4096);
assert_eq!(a.len(), 4097);

let tail = a.slice(2048..); // shares memory with `a`
assert_eq!(tail.len(), 2049);

// `b` shares its buffer, so its fair share is well under its total.
assert!(
    b.estimated_memory_usage(MemoryAccounting::FairShare)
        < b.estimated_memory_usage(MemoryAccounting::Total)
);
```

## Zero-copy ingestion with `CordBuffer`

A `CordBuffer` is writable storage in the cord's native chunk format. Once it
is filled, appending a heap-backed buffer transfers its allocation directly
into the cord without copying the payload. A sufficiently large `Vec<u8>` can
also be adopted without copying, but it remains an external buffer and needs a
separate metadata allocation; `CordBuffer` becomes a flat cord chunk directly
and can reuse spare capacity from the cord's existing tail.

```rust
use std::io::{self, Read};
use cord_rs::{Cord, CordBuffer};

/// Reads up to `n` bytes from `src` directly into cord storage.
fn read_into_cord<R: Read>(mut src: R, mut n: usize) -> io::Result<Cord> {
    let mut cord = Cord::new();
    while n > 0 {
        let mut buffer = CordBuffer::with_capacity(n);
        let take = buffer.available().min(n);

        // `Read` wants initialized memory: zero the region, read into it,
        // then trim back to what was actually read.
        buffer.extend(std::iter::repeat_n(0u8, take));
        let read = src.read(buffer.as_mut_slice())?;
        if read == 0 {
            break;
        }

        buffer.truncate(read);
        cord.append(buffer); // transfers the chunk without copying its payload
        n -= read;
    }
    Ok(cord)
}

let input = vec![b'x'; 10_000];
let cord = read_into_cord(&input[..], input.len()).unwrap();
assert_eq!(cord, input);
```

`buffer.extend(...)` + `truncate` keeps the loop entirely safe at the cost of
one `memset` per read. A producer whose API explicitly accepts uninitialized
output memory (or otherwise guarantees that it only writes to the destination)
can instead use `spare_capacity_mut` and commit the initialized prefix with the
`unsafe` `set_len`, exactly as with `Vec`; a general `std::io::Read` cannot.
For sources that hand you bytes rather than take a destination buffer, the safe
`put_slice`, `Extend`, `std::io::Write` (and `bytes::BufMut` with the `bytes`
feature) all work.

Buffers created with `CordBuffer::with_capacity` are capped at
`CordBuffer::DEFAULT_MAX_CAPACITY` (4083 bytes on 64-bit: a 4 KiB allocation
minus the 13-byte chunk header). That default trades CPU efficiency against
memory overhead and fragmentation. If — and only if — you have measurements
showing your data is many times larger, `with_capacity_and_block_size` goes up
to `CordBuffer::MAX_BLOCK_SIZE` (64 KiB):

```rust
use cord_rs::CordBuffer;
// A 64 KiB block: capacity is the block size minus the chunk header.
let buffer = CordBuffer::with_capacity_and_block_size(1 << 20, 64 << 10);
assert_eq!(buffer.capacity(), CordBuffer::max_capacity_for(64 << 10));
// Smaller requests round *down* to a power-of-two block to keep the
// distribution of allocation sizes narrow; capacity is again the block
// size minus the chunk header.
let buffer = CordBuffer::with_capacity_and_block_size(19_586, 64 << 10);
assert_eq!(buffer.capacity(), CordBuffer::max_capacity_for(16 << 10));
```

## Reading and parsing

Three ways to read a cord, in increasing order of control:

- `cord.chunks()` yields the contiguous slices the cord is made of — the
  fastest way to process every byte.
- `cord.bytes()` yields bytes, for when a chunk loop would be awkward.
- `cord.cursor()` is a position inside the cord: peek, advance, seek, and —
  the interesting one — `read_cord(n)`, which splits off the next `n` bytes
  as their own cord that *shares* the source's memory.

```rust
use cord_rs::Cord;

/// Splits a `[kind: u8][len: u32 BE][payload]` frame without copying the payload.
fn parse_frame(cord: &Cord) -> Option<(u8, Cord)> {
    let mut cursor = cord.cursor();
    let mut header = [0u8; 5];
    std::io::Read::read_exact(&mut cursor, &mut header).ok()?;
    let len = u32::from_be_bytes(header[1..].try_into().unwrap()) as usize;
    if cursor.remaining() < len {
        return None;
    }
    Some((header[0], cursor.read_cord(len)))
}

let mut framed = Cord::from(&[1u8, 0, 0, 0, 3][..]);
framed.append("abc");
assert_eq!(parse_frame(&framed), Some((1, Cord::from("abc"))));
```

A `Cursor` implements `std::io::Read`, `BufRead` and `Seek` (and `bytes::Buf`
with the `bytes` feature), so existing byte-oriented code can read a cord
directly. It is cheap to clone, which makes speculative parsing easy: clone,
try, and keep the cursor that succeeded. With the `bytes` feature,
`bytes::Buf` gives the same cursor `get_u8`/`get_u32` and friends.

## API sketch

Naming follows the [`bytes`] crate where a convention exists: `append` /
`prepend` (accepting slices, strings, cords, owned buffers — see
[`CordSource`]), `advance`, `truncate`, `slice`, `split_off`, `split_to`,
`as_contiguous`, `make_contiguous`, `chunks()`, `bytes()`, `cursor()`
(chunked reading, skipping, sub-cord extraction — the returned iterator types
live in [`cord_rs::iter`]), `find`, `starts_with` / `ends_with` / `contains`,
`compare` plus `PartialEq`/`Ord` against slices, strings and cords, `Hash`
(chunk-layout independent), `Index`, `Extend`, `FromIterator`,
`io::Write`/`fmt::Write`. Out-of-range indices and ranges panic, like `Vec`
and `bytes::Bytes`; `get` is the non-panicking form for both.

Cargo features:

- `std` (default): `std::io` integration: `Read`/`BufRead`/`Seek` on `Cursor`,
  `Write` on `Cord`, `CordBuffer` and `CordWriter`, plus the dependencies' own
  `std` features (`memchr`'s runtime SIMD detection). Disable it for a
  `no_std` + `alloc` build: `cord-rs = { version = "0.1", default-features =
  false }`; everything else, including `core::fmt::Write` for `Cord`, stays
  available. `no_std` targets need 32-bit and pointer-width atomics.
- `bytes`: `bytes::Buf` for `Cord` and `Cursor`, `bytes::BufMut` for
  `CordWriter` and `CordBuffer`, and zero-copy conversions with
  `bytes::Bytes`.
- `serde`: `Serialize` / `Deserialize` for `Cord` as a byte sequence.

The crate has one required dependency, [`memchr`], used for substring search
(`find` and `contains`).

## Relationship to Abseil's `absl::Cord`

`cord-rs` is a Rust port of [`absl::Cord`][absl-cord], a rope-like byte
container used in systems that assemble and share large byte sequences. It has
an idiomatic Rust API, while its internals started from Abseil's design (a
16-byte handle with inline storage, size-classed buffers, a shallow b-tree, and
the same sharing thresholds and growth policy) and evolve independently where
that makes the crate simpler or the trees smaller. The Cordz sampling /
profiling layer and the CRC checksum node are not ported.

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
| `CreateWithDefaultLimit(n)`           | `CordBuffer::with_capacity(n)`                  |
| `CreateWithCustomLimit(block, n)`     | `CordBuffer::with_capacity_and_block_size(n, block)` (note the argument order) |
| `MaximumPayload()` / `(block_size)`   | `CordBuffer::DEFAULT_MAX_CAPACITY` / `max_capacity_for(block_size)` |
| `IncreaseLengthBy` / `SetLength`      | `CordBuffer::set_len` (unsafe), `truncate`, `clear` |
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
| `EstimatedMemoryUsage(mode)`          | `estimated_memory_usage(MemoryAccounting::…)`   |
| `SetExpectedChecksum` / `ExpectedChecksum` | not ported (no CRC node)                   |
| `swap`                                | `std::mem::swap`                                |

`Subcord(pos, n)` clamps out-of-range arguments; `slice`/`split_off`/`split_to`
panic and `get` returns `None`, following `Vec` and `bytes::Bytes`.

`MakeCordFromExternal` takes a releaser callback, and abseil warns that a
releaser which does nothing is "likely a bug". cord-rs instead takes an owning
value — `&'static [u8]`, `Arc<[u8]>`, `Vec<u8>`, `bytes::Bytes` — so the
lifetime is carried by the type and that mistake cannot be made.

</details>

## Safety and testing

The core is a reference-counted tree in the abseil tradition and uses
`unsafe` internally; the public API is safe. Verification:

- The suite is organized by area — construction, editing, slicing,
  comparison, iteration, buffers, accounting and a set of stress workloads —
  and includes the cases from abseil's own suites (`cord_test.cc`,
  `cord_buffer_test.cc`, `cord_rep_btree_test.cc`,
  `cord_rep_btree_navigator_test.cc`, `cord_rep_btree_reader_test.cc`,
  `cord_data_edge_test.cc`), rewritten against the Rust API; like the rest of
  the crate they derive from Apache-2.0 licensed Abseil code (see `NOTICE`).
- A [proptest] model test runs random operation sequences (append, prepend,
  slice, split, advance, truncate, clone, make_contiguous, cursor reads, ...) on
  several cords sharing structure, against `Vec<u8>` oracles, validating the
  tree after every step.
- The test suite runs under [Miri] with strict provenance and under
  AddressSanitizer / ThreadSanitizer; both are required CI jobs.
- 64-bit and 32-bit targets, little and big endian, are supported (CI covers
  `i686`, `wasm32` and a `powerpc64` build). CI also links a real `no_std`
  binary — allocator, panic handler and all — for a bare-metal target
  (`aarch64-unknown-none`).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). `scripts/check.sh` is the pre-commit
gate (rustfmt; pedantic clippy, checks and docs with warnings as errors
across all five feature combinations; tests in three feature configurations;
a 32-bit compile of the crate and tests; and a bare-metal `no_std` check on
`aarch64-unknown-none`, skipped with a message when the target isn't
installed — `rustup target add aarch64-unknown-none`);
`scripts/miri.sh` and `scripts/sanitize.sh` are the required pre-push
soundness checks (nightly — a development-only requirement); `cargo bench`
runs the Criterion benchmarks.

## Minimum supported Rust version

1.95 (edition 2024) to build and use the crate; validation tooling (Miri,
sanitizers) additionally needs a nightly toolchain.

## License

Licensed under the Apache License, Version 2.0.

This crate is an independent Rust port of portions of Abseil's `absl::Cord`.
Portions of the implementation and documentation are derived from the
Apache-2.0-licensed Abseil C++ Common Libraries, Copyright The Abseil Authors.
This project is not affiliated with or endorsed by the Abseil project. See
`LICENSE` and `NOTICE`.

[absl-cord]: https://github.com/abseil/abseil-cpp/blob/master/absl/strings/cord.h
[`bytes`]: https://crates.io/crates/bytes
[`memchr`]: https://crates.io/crates/memchr
[`CordSource`]: https://docs.rs/cord-rs/latest/cord_rs/trait.CordSource.html
[`cord_rs::iter`]: https://docs.rs/cord-rs/latest/cord_rs/iter/index.html
[proptest]: https://crates.io/crates/proptest
[Miri]: https://github.com/rust-lang/miri
