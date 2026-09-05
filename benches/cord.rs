//! Criterion micro-benchmarks for the hot paths of `Cord`.
//!
//! Run with `cargo bench`. Groups cover construction, cloning, append /
//! prepend growth, slicing, iteration, comparison, search, flattening and
//! hashing, with `Vec<u8>` baselines where a comparison is meaningful.
#![expect(clippy::cast_possible_truncation, reason = "benchmarks juggle small integers freely")]

use std::hash::{Hash, Hasher};
use std::hint::black_box;

use cord_rs::{__internal as internal, Cord, CordBuffer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const SIZES: [usize; 5] = [8, 100, 4 << 10, 64 << 10, 1 << 20];

struct BenchOwner(Box<[u8]>);

impl AsRef<[u8]> for BenchOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

fn data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 % 251) as u8).collect()
}

/// Deterministic owned strings with a mixed, boundary-heavy size distribution.
fn mixed_strings(total: usize) -> Vec<String> {
    let mut seed = 0x4d59_5df4_d0f3_3173_u64;
    let mut remaining = total;
    let mut pieces = Vec::new();
    while remaining != 0 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let bucket = (seed >> 32) % 100;
        let (min, max) = match bucket {
            0..50 => (1usize, 15),
            50..80 => (16, 511),
            80..97 => (512, 4095),
            _ => (4096, 64 << 10),
        };
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let len = (min + (seed as usize % (max - min + 1))).min(remaining);
        pieces.push(String::from_utf8(vec![b'a' + (pieces.len() % 26) as u8; len]).unwrap());
        remaining -= len;
    }
    pieces
}

/// A cord of `n` bytes built from `chunk` sized appends (fragmented tree).
fn fragmented(n: usize, chunk: usize) -> Cord {
    let bytes = data(n);
    let mut cord = Cord::new();
    for piece in bytes.chunks(chunk) {
        cord.append(piece);
    }
    cord
}

/// A cord built from `chunk` sized pieces, each its own external node,
/// joined by `prepend` (like the test suite's `make_fragmented_cord`).
/// Plain `Cord::prepend`/`append` of a raw slice below `MIN_FLAT_LENGTH`
/// still rounds up to a size-classed flat and can absorb later small
/// pieces into that same spare capacity; giving each piece its own
/// external node guarantees it stays exactly `chunk`-sized and never
/// coalesces with its neighbors.
fn fragmented_external(bytes: &[u8], chunk: usize) -> Cord {
    let mut cord = Cord::new();
    for piece in bytes.chunks(chunk) {
        let mut tmp = internal::make_external(piece);
        tmp.prepend(&cord);
        cord = tmp;
    }
    cord
}

fn bench_construct(c: &mut Criterion) {
    let mut g = c.benchmark_group("construct");
    for size in SIZES {
        let bytes = data(size);
        g.throughput(Throughput::Bytes(size as u64));
        // Criterion drops values returned by these first four routines after
        // timing, so they isolate construction latency.
        g.bench_with_input(BenchmarkId::new("from_slice", size), &bytes, |b, bytes| {
            b.iter(|| Cord::from(black_box(&bytes[..])));
        });
        g.bench_with_input(BenchmarkId::new("from_vec", size), &bytes, |b, bytes| {
            b.iter_batched(|| bytes.clone(), |v| Cord::from(black_box(v)), criterion::BatchSize::SmallInput);
        });
        g.bench_with_input(BenchmarkId::new("from_box", size), &bytes, |b, bytes| {
            b.iter_batched(
                || bytes.clone().into_boxed_slice(),
                |v| Cord::from(black_box(v)),
                criterion::BatchSize::SmallInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("from_owner", size), &bytes, |b, bytes| {
            b.iter_batched(
                || BenchOwner(bytes.clone().into_boxed_slice()),
                |owner| Cord::from_owner(black_box(owner)),
                criterion::BatchSize::SmallInput,
            );
        });
        // Explicitly dropping inside the routine measures the complete
        // construction-and-release lifecycle for owned inputs.
        g.bench_with_input(BenchmarkId::new("from_vec_and_drop", size), &bytes, |b, bytes| {
            b.iter_batched(
                || bytes.clone(),
                |v| drop(black_box(Cord::from(black_box(v)))),
                criterion::BatchSize::SmallInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("from_box_and_drop", size), &bytes, |b, bytes| {
            b.iter_batched(
                || bytes.clone().into_boxed_slice(),
                |v| drop(black_box(Cord::from(black_box(v)))),
                criterion::BatchSize::SmallInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("from_owner_and_drop", size), &bytes, |b, bytes| {
            b.iter_batched(
                || BenchOwner(bytes.clone().into_boxed_slice()),
                |owner| drop(black_box(Cord::from_owner(black_box(owner)))),
                criterion::BatchSize::SmallInput,
            );
        });
        g.bench_with_input(BenchmarkId::new("vec_from_slice", size), &bytes, |b, bytes| {
            b.iter(|| black_box(&bytes[..]).to_vec());
        });
    }
    g.finish();
}

fn bench_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("clone");
    let inline = Cord::from("inline");
    let flat = Cord::from(&data(1000)[..]);
    let tree = fragmented(1 << 20, 1000);
    g.bench_function("inline", |b| b.iter(|| black_box(&inline).clone()));
    g.bench_function("flat", |b| b.iter(|| black_box(&flat).clone()));
    g.bench_function("btree_1MiB", |b| b.iter(|| black_box(&tree).clone()));
    g.finish();
}

fn bench_drop(c: &mut Criterion) {
    let mut g = c.benchmark_group("drop");
    let tree = fragmented(1 << 20, 1000);
    let bytes = data(64 << 10);

    g.bench_function("shared_btree_1MiB", |b| {
        b.iter_batched(|| tree.clone(), drop, criterion::BatchSize::SmallInput);
    });
    g.bench_function("unique_btree_1MiB", |b| {
        b.iter_batched(|| fragmented(1 << 20, 1000), drop, criterion::BatchSize::SmallInput);
    });
    g.bench_function("unique_global_64KiB", |b| {
        b.iter_batched(|| Cord::from(bytes.clone()), drop, criterion::BatchSize::SmallInput);
    });
    g.bench_function("unique_substring_64KiB", |b| {
        b.iter_batched(
            || {
                let base = Cord::from(bytes.clone());
                base.slice(1..base.len() - 1)
            },
            drop,
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_convert(c: &mut Criterion) {
    let mut g = c.benchmark_group("convert");
    let bytes = data(1 << 20);
    g.throughput(Throughput::Bytes(bytes.len() as u64));
    g.bench_function("into_vec_global_unique", |b| {
        b.iter_batched(
            || Cord::from(bytes.clone()),
            |cord| Vec::<u8>::from(black_box(cord)),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("into_string_global_unique", |b| {
        b.iter_batched(
            || Cord::from(String::from_utf8(vec![b'x'; 1 << 20]).unwrap()),
            |cord| String::try_from(black_box(cord)).unwrap(),
            criterion::BatchSize::SmallInput,
        );
    });

    let flat = Cord::copy_from_slice(&bytes[..4000]);
    g.throughput(Throughput::Bytes(flat.len() as u64));
    g.bench_function("into_vec_flat", |b| {
        b.iter_batched(
            || flat.clone(),
            |cord| Vec::<u8>::from(black_box(cord)),
            criterion::BatchSize::SmallInput,
        );
    });

    let fragmented = fragmented(1 << 20, 1000);
    g.throughput(Throughput::Bytes(fragmented.len() as u64));
    g.bench_function("into_vec_fragmented", |b| {
        b.iter_batched(
            || fragmented.clone(),
            |cord| Vec::<u8>::from(black_box(cord)),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("to_vec_fragmented", |b| b.iter(|| black_box(&fragmented).to_vec()));
    g.finish();
}

fn bench_append(c: &mut Criterion) {
    let mut g = c.benchmark_group("append");
    for chunk in [1usize, 16, 100, 1000, 4096] {
        let total = 1 << 20;
        let bytes = data(total);
        g.throughput(Throughput::Bytes(total as u64));
        g.bench_with_input(BenchmarkId::new("slices_to_1MiB", chunk), &bytes, |b, bytes| {
            b.iter(|| {
                let mut cord = Cord::new();
                for piece in bytes.chunks(chunk) {
                    cord.append(black_box(piece));
                }
                cord
            });
        });
        g.bench_with_input(BenchmarkId::new("vec_extend_to_1MiB", chunk), &bytes, |b, bytes| {
            b.iter(|| {
                let mut v = Vec::new();
                for piece in bytes.chunks(chunk) {
                    v.extend_from_slice(black_box(piece));
                }
                v
            });
        });
    }
    // Exercise inline transitions, copied flats, adopted external buffers,
    // B-tree packing, and occasional large nodes in a single stable workload.
    let mixed = mixed_strings(1 << 20);
    g.throughput(Throughput::Bytes(1 << 20));
    g.bench_function("owned_strings_mixed_to_1MiB", |b| {
        b.iter_batched(
            || mixed.clone(),
            |pieces| {
                let mut cord = Cord::new();
                for piece in pieces {
                    cord.append(black_box(piece));
                }
                cord
            },
            criterion::BatchSize::SmallInput,
        );
    });
    // Appending cords: shared vs owned.
    let piece = fragmented(64 << 10, 4000);
    g.throughput(Throughput::Bytes((64 << 10) * 16));
    g.bench_function("cords_shared_16x64KiB", |b| {
        b.iter(|| {
            let mut cord = Cord::new();
            for _ in 0..16 {
                cord.append(black_box(&piece));
            }
            cord
        });
    });
    g.bench_function("cords_owned_16x64KiB", |b| {
        b.iter(|| {
            let mut cord = Cord::new();
            for _ in 0..16 {
                cord.append(black_box(piece.clone()));
            }
            cord
        });
    });
    // Zero-copy building through CordBuffer.
    let total = 1 << 20;
    let bytes = data(total);
    g.throughput(Throughput::Bytes(total as u64));
    g.bench_function("cord_buffer_to_1MiB", |b| {
        b.iter(|| {
            let mut cord = Cord::new();
            let mut src = &bytes[..];
            while !src.is_empty() {
                let mut buffer = CordBuffer::with_capacity(src.len());
                let n = buffer.put_slice_partial(src);
                src = &src[n..];
                cord.append(buffer);
            }
            cord
        });
    });
    g.finish();
}

fn bench_prepend(c: &mut Criterion) {
    let mut g = c.benchmark_group("prepend");
    for chunk in [16usize, 1000] {
        let total = 256 << 10;
        let bytes = data(total);
        g.throughput(Throughput::Bytes(total as u64));
        g.bench_with_input(BenchmarkId::new("slices_to_256KiB", chunk), &bytes, |b, bytes| {
            b.iter(|| {
                let mut cord = Cord::new();
                for piece in bytes.rchunks(chunk) {
                    cord.prepend(black_box(piece));
                }
                cord
            });
        });
    }
    g.finish();
}

fn bench_slice(c: &mut Criterion) {
    let mut g = c.benchmark_group("slice");
    let flat = Cord::from(&data(4000)[..]);
    let tree = fragmented(1 << 20, 1000);
    g.bench_function("flat_middle", |b| b.iter(|| black_box(&flat).slice(1000..3000)));
    g.bench_function("btree_small_inline", |b| b.iter(|| black_box(&tree).slice(500_000..500_010)));
    g.bench_function("btree_middle_64KiB", |b| {
        b.iter(|| black_box(&tree).slice(500_000..(500_000 + (64 << 10))));
    });
    g.bench_function("btree_advance_1KiB", |b| {
        b.iter_batched(|| tree.clone(), |mut c| c.advance(1024), criterion::BatchSize::SmallInput);
    });
    g.bench_function("btree_truncate_1KiB", |b| {
        b.iter_batched(|| tree.clone(), |mut c| c.truncate(c.len() - 1024), criterion::BatchSize::SmallInput);
    });
    g.bench_function("btree_split_off_middle", |b| {
        b.iter_batched(|| tree.clone(), |mut c| c.split_off(500_000), criterion::BatchSize::SmallInput);
    });
    g.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let mut g = c.benchmark_group("iterate");
    let tree = fragmented(1 << 20, 1000);
    let vec = tree.to_vec();
    g.throughput(Throughput::Bytes(vec.len() as u64));
    g.bench_function("chunks_sum", |b| {
        b.iter(|| {
            black_box(&tree).chunks().map(|c| c.iter().map(|&x| u64::from(x)).sum::<u64>()).sum::<u64>()
        });
    });
    g.bench_function("chunks_rev_sum", |b| {
        b.iter(|| {
            black_box(&tree).chunks().rev().map(|c| c.iter().map(|&x| u64::from(x)).sum::<u64>()).sum::<u64>()
        });
    });
    g.bench_function("bytes_sum", |b| b.iter(|| black_box(&tree).bytes().map(u64::from).sum::<u64>()));
    g.bench_function("bytes_rev_sum", |b| {
        b.iter(|| black_box(&tree).bytes().rev().map(u64::from).sum::<u64>());
    });
    g.bench_function("index_every_4KiB", |b| {
        b.iter(|| (0..tree.len()).step_by(4096).map(|i| u64::from(tree[i])).sum::<u64>());
    });
    g.bench_function("cursor_read_4KiB_pieces", |b| {
        b.iter(|| {
            let mut cursor = black_box(&tree).cursor();
            let mut n = 0;
            while cursor.has_remaining() {
                n += cursor.read_cord(4096.min(cursor.remaining())).len();
            }
            n
        });
    });
    g.bench_function("vec_sum_baseline", |b| {
        b.iter(|| black_box(&vec).iter().map(|&x| u64::from(x)).sum::<u64>());
    });
    g.bench_function("to_vec", |b| b.iter(|| black_box(&tree).to_vec()));
    g.finish();
}

fn bench_compare(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare");
    let a = fragmented(1 << 20, 1000);
    let b_same = fragmented(1 << 20, 777);
    let flat = Cord::from(a.to_vec());
    let vec = a.to_vec();
    g.throughput(Throughput::Bytes(vec.len() as u64));
    g.bench_function("eq_fragmented_vs_fragmented", |b| b.iter(|| black_box(&a) == black_box(&b_same)));
    g.bench_function("eq_fragmented_vs_flat", |b| b.iter(|| black_box(&a) == black_box(&flat)));
    g.bench_function("eq_fragmented_vs_slice", |b| b.iter(|| black_box(&a) == black_box(&vec[..])));
    g.bench_function("cmp_shared_clone", |b| {
        let clone = a.clone();
        b.iter(|| black_box(&a).cmp(black_box(&clone)));
    });
    // Two distinct small inline cords: guards the only regression path of
    // 0351c79 (the `is_same` shared-pointer check added ahead of the
    // existing inline fast path costs a little extra for cords that were
    // never going to be `is_same` anyway).
    let inline_a = Cord::from("inline cord a");
    let inline_b = Cord::from("inline cord b");
    g.bench_function("cmp_inline_vs_inline", |b| {
        b.iter(|| black_box(&inline_a).cmp(black_box(&inline_b)));
    });
    let slice_suffix = &vec[vec.len() - 64..];
    let fragmented_suffix = a.slice(a.len() - (64 << 10)..);
    g.bench_function("ends_with_slice_64B", |b| {
        b.iter(|| black_box(&a).ends_with(black_box(slice_suffix)));
    });
    g.bench_function("ends_with_fragmented_64KiB", |b| {
        b.iter(|| black_box(&a).ends_with(black_box(&fragmented_suffix)));
    });
    g.bench_function("vec_eq_baseline", |b| b.iter(|| black_box(&vec) == black_box(&vec)));
    g.finish();
}

fn bench_find(c: &mut Criterion) {
    let mut g = c.benchmark_group("find");
    let mut tree = fragmented(1 << 20, 1000);
    tree.append("needle");
    let vec = tree.to_vec();
    g.throughput(Throughput::Bytes(vec.len() as u64));
    g.bench_function("find_at_end", |b| b.iter(|| black_box(&tree).find("needle")));
    g.bench_function("find_cord_needle_at_end", |b| {
        let needle = Cord::from("needle");
        b.iter(|| black_box(&tree).find(&needle));
    });
    g.bench_function("vec_windows_baseline", |b| {
        b.iter(|| black_box(&vec).windows(6).position(|w| w == b"needle"));
    });

    let adversarial_vec = vec![b'a'; 64 << 10];
    let mut adversarial = Cord::new();
    for chunk in adversarial_vec.chunks(1000) {
        adversarial.append(chunk);
    }
    g.throughput(Throughput::Bytes(adversarial_vec.len() as u64));
    g.bench_function("adversarial_repeated_prefix", |b| {
        b.iter(|| black_box(&adversarial).find(black_box(&b"aaaaab"[..])));
    });
    g.bench_function("adversarial_vec_windows_baseline", |b| {
        b.iter(|| black_box(&adversarial_vec).windows(6).position(|w| w == b"aaaaab"));
    });

    // Fine fragmentation: haystack chunks shorter than the needle, built
    // with `prepend` (see `fragmented_prepend`) so pieces don't coalesce
    // back into larger chunks the way `append` would. The needle matches
    // only at the very end, so almost every position is a false start that
    // has to cross several tiny chunk boundaries before failing — the
    // documented O(n*m) pathology of `find`, otherwise unbenchmarked.
    let fine_needle = b"aaaaaaaaaaab";
    let mut fine_bytes = vec![b'a'; 8 << 10];
    *fine_bytes.last_mut().unwrap() = b'b';
    let fine_haystack = fragmented_external(&fine_bytes, 3);
    assert!(
        fine_haystack.chunks().all(|c| c.len() < fine_needle.len()),
        "fine haystack must stay fragmented into chunks shorter than the needle"
    );
    g.throughput(Throughput::Bytes(fine_bytes.len() as u64));
    g.bench_function("fine_fragmentation_matching_needle", |b| {
        b.iter(|| black_box(&fine_haystack).find(black_box(&fine_needle[..])));
    });
    g.finish();
}

fn bench_flatten(c: &mut Criterion) {
    let mut g = c.benchmark_group("flatten");
    for size in [4 << 10, 64 << 10, 1 << 20] {
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(BenchmarkId::new("fragmented", size), &size, |b, &size| {
            b.iter_batched(
                || fragmented(size, 500),
                |mut c| {
                    let _ = c.make_contiguous();
                    c
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_hash(c: &mut Criterion) {
    let mut g = c.benchmark_group("hash");
    let tree = fragmented(64 << 10, 1000);
    let flat = Cord::from(tree.to_vec());
    let vec = tree.to_vec();
    g.throughput(Throughput::Bytes(vec.len() as u64));
    g.bench_function("fragmented_64KiB", |b| {
        b.iter(|| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            black_box(&tree).hash(&mut h);
            h.finish()
        });
    });
    g.bench_function("flat_64KiB", |b| {
        b.iter(|| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            black_box(&flat).hash(&mut h);
            h.finish()
        });
    });
    g.bench_function("vec_64KiB_baseline", |b| {
        b.iter(|| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            black_box(&vec).hash(&mut h);
            h.finish()
        });
    });
    g.finish();
}

fn bench_diabolical(c: &mut Criterion) {
    let mut g = c.benchmark_group("diabolical");
    g.sample_size(20);
    // Shared-before-every-append growth (worst case for in place appends).
    g.bench_function("shared_single_byte_appends_5000", |b| {
        b.iter(|| {
            let mut cord = Cord::new();
            for i in 0..5000u32 {
                let _shared = cord.clone();
                cord.append(&[(i % 256) as u8][..]);
            }
            cord
        });
    });
    // Repeated split / overwrite / join, hostile to btrees.
    g.bench_function("split_insert_join_x100_on_1MiB", |b| {
        let base = fragmented(1 << 20, 1024);
        let patch = data(500);
        b.iter_batched(
            || base.clone(),
            |mut cord| {
                let mut seed = 12345u64;
                for _ in 0..100 {
                    seed =
                        seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    let offset = (seed >> 33) as usize % (cord.len() - patch.len());
                    let mut suffix = cord.clone();
                    suffix.advance(offset + patch.len());
                    cord.truncate(offset);
                    cord.append(&patch[..]);
                    cord.append(suffix);
                }
                cord
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_access(c: &mut Criterion) {
    let mut g = c.benchmark_group("access");
    let flat = Cord::from(&data(4000)[..]);
    let tree = fragmented(1 << 20, 1000);
    g.bench_function("len_tree", |b| b.iter(|| black_box(&tree).len()));
    g.bench_function("len_inline", |b| {
        let c = Cord::from("hello inline!");
        b.iter(|| black_box(&c).len());
    });
    g.bench_function("index_every_64", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            let mut i = 0;
            while i < tree.len() {
                sum += black_box(&tree)[i] as usize;
                i += 64;
            }
            sum
        });
    });
    g.bench_function("index_flat_every_byte", |b| {
        b.iter(|| (0..flat.len()).map(|i| usize::from(black_box(&flat)[i])).sum::<usize>());
    });
    let flat_substring = flat.slice(100..3900);
    g.bench_function("index_flat_substring_every_byte", |b| {
        b.iter(|| {
            (0..flat_substring.len()).map(|i| usize::from(black_box(&flat_substring)[i])).sum::<usize>()
        });
    });
    g.bench_function("copy_prefix_to_flat", |b| {
        let mut dst = vec![0u8; 4000];
        b.iter(|| black_box(&flat).copy_prefix_to(&mut dst));
    });
    g.bench_function("copy_prefix_to_tree_64KiB", |b| {
        let mut dst = vec![0u8; 64 << 10];
        b.iter(|| black_box(&tree).copy_prefix_to(&mut dst));
    });
    g.bench_function("cursor_read_8B_pieces", |b| {
        let small = tree.slice(0..(16 << 10));
        b.iter(|| {
            let mut cursor = black_box(&small).cursor();
            let mut n = 0usize;
            while cursor.remaining() != 0 {
                n += cursor.read_cord(8.min(cursor.remaining())).len();
            }
            n
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_access,
    bench_construct,
    bench_clone,
    bench_drop,
    bench_convert,
    bench_append,
    bench_prepend,
    bench_slice,
    bench_iterate,
    bench_compare,
    bench_find,
    bench_flatten,
    bench_hash,
    bench_diabolical
);
criterion_main!(benches);
