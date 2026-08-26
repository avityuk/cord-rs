//! Heavy structural / adversarial workloads and validity.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "tests juggle small integers freely"
)]

use crate::common::{self, internal};
use cord_rs::Cord;

/// Creates a cord of at least 128 GB (2 GB on 32-bit) using reference
/// counting.
#[test]
fn a_128_gib_cord_of_shared_external_nodes_stays_valid() {
    let one_gig: usize = 1024 * 1024 * 1024;
    // 128 GiB on 64-bit targets, 2 GiB on 32-bit ones (`checked_mul` keeps
    // the 64-bit constant from being rejected by the overflow lint on
    // 32-bit). Under Miri, `validate()` recurses into every leaf edge
    // (including shared ones), so a tree with ~1M edges would take hours to
    // interpret even though nothing past this point is aliasing-sensitive —
    // it's arithmetic (>4 GiB length accounting), not pointer provenance —
    // so the target size is cut down drastically.
    let max_size: usize =
        if cfg!(miri) { 256 << 20 } else { one_gig.checked_mul(128).unwrap_or(2 * one_gig) };
    let length = 128 * 1024;
    let from = internal::make_external(&vec![b'x'; length]);

    // Grow incrementally and exponentially so the tree needs rebalancing.
    let mut c = Cord::new();
    c.append(&from);
    while c.len() < max_size {
        let copy = c.clone();
        c.append(copy);
        c.append(&from);
        c.append(&from);
        c.append(&from);
        c.append(&from);
    }
    let trailing: u32 = if cfg!(miri) { 64 } else { 1024 };
    for _ in 0..trailing {
        c.append(&from);
    }
    assert!(c.len() >= max_size);
    common::assert_valid(&c);
    assert_eq!(c[c.len() - 1], b'x');
}

struct TestData {
    data: Vec<Vec<u8>>,
}

impl TestData {
    fn make_string(length: usize) -> Vec<u8> {
        let buf = format!("({length})");
        let mut result = Vec::new();
        while result.len() < length {
            result.extend_from_slice(buf.as_bytes());
        }
        result.truncate(length);
        result
    }

    fn new() -> Self {
        // Strings around half of the maximum flat length (32-bit-correct:
        // derived from `MAX_FLAT_LENGTH`, not a hardcoded 64-bit value).
        const MAX_FLAT_LENGTH: usize = internal::MAX_FLAT_LENGTH;
        const HALF: usize = MAX_FLAT_LENGTH / 2;
        let mut data = Vec::new();
        // Short strings increasing in length by one.
        for i in 0..30 {
            data.push(Self::make_string(i));
        }
        for i in -10i64..=10 {
            data.push(Self::make_string((HALF as i64 + i) as usize));
        }
        for i in -10i64..=10 {
            data.push(Self::make_string((MAX_FLAT_LENGTH as i64 + i) as usize));
        }
        Self { data }
    }
}

#[test]
fn append_and_prepend_at_every_interesting_length() {
    let mut d = TestData::new();
    if cfg!(miri) {
        // The inner loop is quadratic in the length list; take every 7th
        // length instead of scaling the whole 72-length x 72-length matrix.
        d.data = d.data.into_iter().step_by(7).collect();
    }
    for a in &d.data {
        {
            // Construct from Cord.
            let tmp = Cord::from(&a[..]);
            let x = tmp.clone();
            assert_eq!(x.to_vec(), *a);
        }
        {
            // Construct from slice.
            let x = Cord::from(&a[..]);
            assert_eq!(x.to_vec(), *a);
        }
        {
            // Append cord to self.
            let mut this = Cord::from(&a[..]);
            let copy = this.clone();
            this.append(copy);
            assert_eq!(this.to_vec(), [&a[..], &a[..]].concat());
        }
        {
            // Prepend cord to self.
            let mut this = Cord::from(&a[..]);
            let copy = this.clone();
            this.prepend(copy);
            assert_eq!(this.to_vec(), [&a[..], &a[..]].concat());
        }
        // Try to append / prepend others.
        for b in &d.data {
            {
                // clone_from Cord.
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.clone_from(&y);
                assert_eq!(x.to_vec(), *b);
            }
            {
                // Assign from slice.
                let mut x = Cord::from(&a[..]);
                assert_eq!(x.to_vec(), *a);
                x = Cord::from(&b[..]);
                assert_eq!(x.to_vec(), *b);
            }
            {
                // append(&Cord)
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.append(&y);
                assert_eq!(x.to_vec(), [&a[..], &b[..]].concat());
            }
            {
                // append(&[u8])
                let mut x = Cord::from(&a[..]);
                x.append(&b[..]);
                assert_eq!(x.to_vec(), [&a[..], &b[..]].concat());
            }
            {
                // prepend(&Cord)
                let mut x = Cord::from(&a[..]);
                let y = Cord::from(&b[..]);
                x.prepend(&y);
                assert_eq!(x.to_vec(), [&b[..], &a[..]].concat());
            }
            {
                // prepend(&[u8])
                let mut x = Cord::from(&a[..]);
                x.prepend(&b[..]);
                assert_eq!(x.to_vec(), [&b[..], &a[..]].concat());
            }
        }
    }
}

/// A diabolical `append(<one byte>)` loop where the cord is shared before
/// every append, producing a terribly fragmented cord.
#[test]
fn single_byte_appends_to_a_reshared_cord_stay_valid() {
    let mut rng = common::Rng::new(9);
    let n: usize = if cfg!(miri) { 300 } else { 5000 };
    let expected = rng.lowercase(n);
    let mut cord = Cord::new();
    for &c in &expected {
        let shared = cord.clone();
        assert_eq!(cord, shared);
        cord.append(&[c][..]);
    }
    assert_eq!(cord.to_vec(), expected);
    common::assert_valid(&cord);
}

/// Mimics an application repeatedly splitting a cord, overwriting a value and
/// recomposing the pieces. This is hostile towards a btree: splits share the
/// boundary nodes and recomposition injects edges, quickly growing the tree
/// to its maximum height.
#[test]
fn repeated_split_and_join_keeps_the_tree_shallow() {
    let mut rng = common::Rng::new(31);
    let data = vec![b'x'; 1 << 10];
    let buffer = Cord::from(&data[..]);
    let mut cord = Cord::new();
    let appends: usize = if cfg!(miri) {
        2000
    } else if cfg!(debug_assertions) {
        100_000
    } else {
        1_000_000
    };
    for _ in 0..appends {
        cord.append(&buffer);
    }

    let rounds: usize = if cfg!(miri) { 50 } else { 1000 };
    for _ in 0..rounds {
        let offset = rng.up_to(cord.len());
        let length = 100 + rng.up_to(data.len() - 100);
        if cord.len() == offset {
            cord.append(&data[..length]);
        } else {
            let mut suffix = Cord::new();
            if offset + length < cord.len() {
                suffix = cord.clone();
                suffix.advance(offset + length);
            }
            if cord.len() > offset {
                cord.truncate(offset);
            }
            cord.append(&data[..length]);
            if !suffix.is_empty() {
                cord.append(suffix);
            }
        }
    }
    common::assert_valid(&cord);
    assert!(cord.len() >= appends * data.len() - rounds * data.len());

    // Even under this adversarial split/join pattern, height must stay
    // within a conservative O(log n) bound instead of degenerating toward a
    // linear chain. The crate doesn't guarantee tight (max-fanout) packing —
    // this workload is deliberately hostile to packing — so assume nodes
    // may be as sparse as half of `BTREE_MAX_CAPACITY`, plus a couple of
    // levels of slack; that's loose enough to tolerate the adversarial
    // workload while still catching gross (e.g. near-linear) degeneracy.
    let chunks = cord.chunks().count();
    let height = internal::btree_height(&cord).expect("a cord this large must be a btree");
    let min_fanout = internal::BTREE_MAX_CAPACITY / 2;
    let mut bound = 0usize;
    let mut reachable = 1usize;
    while reachable < chunks {
        reachable *= min_fanout;
        bound += 1;
    }
    bound += 2; // slack
    assert!(
        height <= bound,
        "btree height {height} exceeds conservative O(log_{min_fanout} n) bound {bound} for {chunks} chunks"
    );
}

// --- Populated cord factories and mutators (from the checksum tests) ------------------
//
// The checksum expectations are out of scope, but the factory / mutator matrix
// is a good structural test: every mutation must leave a valid cord and every
// undoable mutation must restore the original value.

struct PopulatedCordFactory {
    name: &'static str,
    generator: fn() -> Cord,
}

const CORD_FACTORIES: &[PopulatedCordFactory] = &[
    PopulatedCordFactory { name: "sso", generator: || Cord::from("abcde") },
    PopulatedCordFactory {
        name: "flat",
        generator: || {
            // Too large for SSO, small enough to be a single flat.
            let mut flat = Cord::from([b"abcde".as_slice(), &[b'x'; 1000]].concat());
            flat.make_contiguous();
            flat
        },
    },
    PopulatedCordFactory { name: "external", generator: || internal::make_external(b"abcde External!") },
    PopulatedCordFactory {
        name: "external substring",
        generator: || {
            let ext = internal::make_external(b"-abcde External!");
            internal::make_substring(&ext, 1, ext.len() - 1)
        },
    },
    PopulatedCordFactory {
        name: "substring",
        generator: || {
            let mut flat = Cord::from([b"-abcde".as_slice(), &[b'x'; 1000]].concat());
            flat.make_contiguous();
            flat.slice(1..999)
        },
    },
    PopulatedCordFactory {
        name: "fragmented",
        generator: || {
            let fragment = [b"abcde".as_slice(), &[b'x'; 195]].concat();
            // Miri interprets every byte of every chunk `validate()` walks;
            // 200 fragments of a 40 kB cord is minutes, not a check.
            let count: usize = if cfg!(miri) { 20 } else { 200 };
            let cord = common::make_fragmented_cord(std::iter::repeat_n(fragment.clone(), count));
            assert_eq!(cord.len(), fragment.len() * count);
            cord
        },
    },
];

struct CordMutator {
    name: &'static str,
    mutate: fn(&mut Cord),
    undo: Option<fn(&mut Cord)>,
}

const CORD_MUTATORS: &[CordMutator] = &[
    CordMutator { name: "clear", mutate: |c| c.clear(), undo: None },
    CordMutator { name: "overwrite", mutate: |c| *c = Cord::from("overwritten"), undo: None },
    CordMutator {
        name: "append string",
        mutate: |c| c.append("0123456789"),
        undo: Some(|c| c.truncate(c.len() - 10)),
    },
    CordMutator {
        name: "append cord",
        mutate: |c| c.append(common::make_fragmented_cord(["12345", "67890"])),
        undo: Some(|c| c.truncate(c.len() - 10)),
    },
    CordMutator {
        name: "append self",
        mutate: |c| {
            let copy = c.clone();
            c.append(copy);
        },
        undo: Some(|c| c.truncate(c.len() / 2)),
    },
    CordMutator { name: "append empty string", mutate: |c| c.append(""), undo: Some(|_| {}) },
    CordMutator { name: "append empty cord", mutate: |c| c.append(Cord::new()), undo: Some(|_| {}) },
    CordMutator {
        name: "prepend string",
        mutate: |c| c.prepend("9876543210"),
        undo: Some(|c| c.advance(10)),
    },
    CordMutator {
        name: "prepend cord",
        mutate: |c| c.prepend(common::make_fragmented_cord(["98765", "43210"])),
        undo: Some(|c| c.advance(10)),
    },
    CordMutator { name: "prepend empty string", mutate: |c| c.prepend(""), undo: Some(|_| {}) },
    CordMutator { name: "prepend empty cord", mutate: |c| c.prepend(Cord::new()), undo: Some(|_| {}) },
    CordMutator {
        name: "prepend self",
        mutate: |c| {
            let copy = c.clone();
            c.prepend(copy);
        },
        undo: Some(|c| c.advance(c.len() / 2)),
    },
    CordMutator { name: "remove prefix", mutate: |c| c.advance(c.len() / 2), undo: None },
    CordMutator { name: "remove suffix", mutate: |c| c.truncate(c.len() / 2), undo: None },
    CordMutator { name: "remove 0-prefix", mutate: |c| c.advance(0), undo: None },
    CordMutator { name: "remove 0-suffix", mutate: |c| c.truncate(c.len()), undo: None },
    CordMutator { name: "subcord", mutate: |c| *c = c.slice(1..c.len() - 1), undo: None },
    CordMutator {
        name: "swap inline",
        mutate: |c| {
            let mut other = Cord::from("swap");
            core::mem::swap(c, &mut other);
        },
        undo: None,
    },
    CordMutator {
        name: "swap tree",
        mutate: |c| {
            let mut other = Cord::from(vec![b'x'; 10000]);
            core::mem::swap(c, &mut other);
        },
        undo: None,
    },
    CordMutator { name: "split off", mutate: |c| drop(c.split_off(c.len() / 2)), undo: None },
    CordMutator { name: "split to", mutate: |c| drop(c.split_to(c.len() / 2)), undo: None },
    CordMutator {
        name: "flatten",
        mutate: |c| {
            let _ = c.make_contiguous();
        },
        undo: Some(|_| {}),
    },
];

#[test]
fn every_mutation_leaves_a_valid_cord_in_every_representation() {
    for factory in CORD_FACTORIES {
        for shared in [false, true] {
            let shared_cord_source = (factory.generator)();
            let make_instance = || if shared { shared_cord_source.clone() } else { (factory.generator)() };

            let base_value = (factory.generator)();
            let base_value_as_bytes = (factory.generator)().to_vec();
            assert!(base_value.starts_with("abcde"), "{}", factory.name);
            common::assert_valid(&base_value);

            let c1 = make_instance();
            assert_eq!(c1, base_value);

            for mutator in CORD_MUTATORS {
                let mut c2 = make_instance();
                (mutator.mutate)(&mut c2);
                common::assert_valid(&c2);
                if let Some(undo) = mutator.undo {
                    undo(&mut c2);
                    common::assert_valid(&c2);
                    assert_eq!(c2, base_value, "{} / {} / shared {shared}", factory.name, mutator.name);
                }
            }

            // All reading operations function on any representation.
            let cc3 = make_instance();
            assert_eq!(cc3.len(), base_value_as_bytes.len());
            assert!(!cc3.is_empty());
            assert_eq!(cc3.compare(&base_value), std::cmp::Ordering::Equal);
            assert_eq!(cc3.compare(&base_value_as_bytes[..]), std::cmp::Ordering::Equal);
            assert_eq!(cc3.compare("wxyz"), std::cmp::Ordering::Less);
            assert_eq!(cc3.compare(&Cord::from("wxyz")), std::cmp::Ordering::Less);
            assert_eq!(cc3.compare("aaaa"), std::cmp::Ordering::Greater);
            assert_eq!(cc3.compare(&Cord::from("aaaa")), std::cmp::Ordering::Greater);
            assert_eq!(Cord::from("wxyz").compare(&cc3), std::cmp::Ordering::Greater);
            assert_eq!(Cord::from("aaaa").compare(&cc3), std::cmp::Ordering::Less);
            assert!(cc3.starts_with("abcd"));
            assert_eq!(cc3.to_vec(), base_value_as_bytes);
            assert!(cc3.chunks().next().unwrap().starts_with(b"abcde"));
            assert_eq!(cc3.bytes().next(), Some(b'a'));

            let mut char_it = cc3.cursor();
            char_it.advance(2);
            assert_eq!(char_it.read_cord(2), "cd");
            assert_eq!(char_it.peek(), Some(b'e'));
            let mut char_it = cc3.cursor();
            char_it.advance(2);
            assert!(char_it.chunk().starts_with(b"cde"));

            assert_eq!(cc3[0], b'a');
            assert_eq!(cc3[4], b'e');
            assert_eq!(common::default_hash(&cc3), common::default_hash(&base_value));
        }
    }
}

#[test]
fn a_thousand_single_byte_appends_stay_valid() {
    // `DumpGrowth` from the C++ file: single byte appends.
    let n: usize = if cfg!(miri) { 200 } else { 1000 };
    let mut s = Cord::new();
    for i in 0..n {
        s.append(&[b'a' + (i % 26) as u8][..]);
    }
    assert_eq!(s.len(), n);
    common::assert_valid(&s);
    let mut rng = common::Rng::new(37);
    let skewed_appends: usize = if cfg!(miri) { 20 } else { 200 };
    for _ in 0..skewed_appends {
        s.append(rng.lowercase_skewed());
    }
    common::assert_valid(&s);
}

#[test]
fn send_sync_across_threads() {
    let len: u32 = if cfg!(miri) { 2_000 } else { 100_000 };
    let num_threads: u32 = if cfg!(miri) { 3 } else { 8 };
    let data: Vec<u8> = (0..len).map(|i| (i % 200) as u8).collect();
    let cord = Cord::from(data.clone());
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let mut c = cord.clone();
            let data = data.clone();
            std::thread::spawn(move || {
                c.append(vec![t as u8; 1000]);
                c.prepend("x");
                let sub = c.slice(1..=data.len());
                assert_eq!(sub, data);
                c.len()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), data.len() + 1001);
    }
    common::check(&cord, &data);
}

#[test]
fn btree_height_stays_shallow_for_five_hundred_leaves() {
    assert_eq!(internal::btree_height(&Cord::new()), None);
    assert_eq!(internal::btree_height(&Cord::from("inline")), None);

    let mut cord = Cord::new();
    for i in 0..500u32 {
        cord.append(vec![(i % 256) as u8; 20]);
    }
    assert!(internal::is_btree(&cord));
    let height = internal::btree_height(&cord).expect("a btree cord must report a height");
    assert!((1..=6).contains(&height), "unexpected height {height} for a modest 500-leaf tree");
}
