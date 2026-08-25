//! Real bare-metal `no_std` + `no_main` binary for `aarch64-unknown-none`.
//!
//! `cargo check --lib` (what CI's `no_std` job otherwise runs) neither
//! links nor monomorphizes generic code nothing instantiates, and never
//! involves an allocator. This crate is the real thing: rustc must produce
//! a linked executable, so every symbol `cord-rs` references has to
//! resolve without `std` or libc, against a `#[global_allocator]` this
//! crate supplies itself. See `../../CONTRIBUTING.md` for what this proves
//! and how to build it.

#![no_std]
#![no_main]

extern crate alloc;

mod exercise;

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

// A bump allocator over a static arena: the simplest allocator that is
// still a real one (unlike `cargo check`, which never links an allocator
// at all). `dealloc` is deliberately a no-op -- nothing here needs memory
// back, and a bump allocator that can't reclaim is still sound.
const ARENA_SIZE: usize = 8 * 1024 * 1024;

#[repr(align(16))]
struct Arena(
    #[expect(dead_code, reason = "only ever addressed via `&raw mut ARENA`, never read as a field")]
    [u8; ARENA_SIZE],
);

static mut ARENA: Arena = Arena([0; ARENA_SIZE]);
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Bump;

unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let base = &raw mut ARENA as *mut u8;
        let align = layout.align();
        let size = layout.size();
        let mut cur = NEXT.load(Ordering::Relaxed);
        loop {
            let start = base as usize + cur;
            let aligned = (start + align - 1) & !(align - 1);
            let offset = aligned - base as usize;
            let Some(end) = offset.checked_add(size) else {
                return core::ptr::null_mut();
            };
            if end > ARENA_SIZE {
                return core::ptr::null_mut();
            }
            match NEXT.compare_exchange_weak(cur, end, Ordering::Relaxed, Ordering::Relaxed) {
                // SAFETY: `offset + size <= ARENA_SIZE` was just checked
                // above, so this stays within `ARENA`.
                Ok(_) => return unsafe { base.add(offset) },
                Err(actual) => cur = actual,
            }
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOC: Bump = Bump;

/// Where `exercise::run`'s result lands, so the optimizer cannot delete the
/// work and so the values are inspectable in a linked binary (e.g. under a
/// debugger or a QEMU + semihosting harness) without any I/O.
/// `RESULT = [checksum, first_failure_id, failure_count]`; a successful run
/// has `RESULT[2] == 0`.
#[unsafe(no_mangle)]
pub static mut RESULT: [u64; 3] = [0; 3];

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let (sum, first_failure, failures) = exercise::run();
    // SAFETY: single-threaded, single write, before anything could read it.
    unsafe {
        let r = &raw mut RESULT;
        (*r)[0] = sum;
        (*r)[1] = u64::from(first_failure);
        (*r)[2] = u64::from(failures);
    }
    loop {
        core::hint::spin_loop();
    }
}
