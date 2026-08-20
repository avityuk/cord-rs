//! Input traits: [`CordSource`] for values that can be appended to a cord,
//! [`CordLike`] for values a cord can be compared with, and [`CordIndex`]
//! for the index types accepted by [`Cord::get`].

use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::sync::Arc;

use crate::buffer::CordBuffer;
use crate::cord::Cord;
use crate::iter::Chunks;

mod sealed {
    pub trait SourceSealed {}
    pub trait LikeSealed {}
    pub trait IndexSealed {}
}

/// Types that can be appended to or prepended to a [`Cord`] via
/// [`Cord::append`] / [`Cord::prepend`].
///
/// Implemented for:
///
/// * `&T` where `T: AsRef<[u8]>` — byte slices, `&str`, `&Vec<u8>`,
///   `&String`, `&[u8; N]`, ... (copied).
/// * `&Cord` (shared, O(log n)) and `Cord` (moved, O(log n)).
/// * `Vec<u8>`, `String`, `Box<[u8]>`, `Box<str>` — adopted without copying
///   when larger than 511 bytes (and, for `Vec`/`String`, at least half
///   full), copied otherwise.
/// * `Arc<[u8]>`, `Arc<str>`, `Arc<Vec<u8>>`, `Arc<String>` — shared without
///   copying when larger than 511 bytes.
/// * [`CordBuffer`] — moved without copying.
/// * `bytes::Bytes` — shared without copying when larger than 511 bytes
///   (with the `bytes` feature).
///
/// This trait is sealed and cannot be implemented outside this crate.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be appended to a `Cord`",
    note = "`Cord::append` accepts `&[u8]`, `&str`, `&Cord`, `Cord`, `Vec<u8>`, `String`, `Box<[u8]>`, `Arc<[u8]>`, `CordBuffer` and references to any `AsRef<[u8]>` type"
)]
pub trait CordSource: sealed::SourceSealed {
    /// Appends `self` to `cord`.
    #[doc(hidden)]
    fn append_to(self, cord: &mut Cord);
    /// Prepends `self` to `cord`.
    #[doc(hidden)]
    fn prepend_to(self, cord: &mut Cord);
}

impl<T: AsRef<[u8]> + ?Sized> sealed::SourceSealed for &T {}
impl<T: AsRef<[u8]> + ?Sized> CordSource for &T {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        cord.append_slice(self.as_ref());
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        cord.prepend_slice(self.as_ref());
    }
}

impl sealed::SourceSealed for &Cord {}
impl CordSource for &Cord {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        cord.append_cord(self);
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        cord.prepend_cord(self);
    }
}

impl sealed::SourceSealed for Cord {}
impl CordSource for Cord {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        cord.append_owned_cord(self);
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        cord.prepend_owned_cord(self);
    }
}

impl sealed::SourceSealed for CordBuffer {}
impl CordSource for CordBuffer {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        cord.append_buffer(self);
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        cord.prepend_buffer(self);
    }
}

macro_rules! impl_owned_source {
    ($($t:ty => |$v:ident| $cap:expr),* $(,)?) => {$(
        impl sealed::SourceSealed for $t {}
        impl CordSource for $t {
            #[inline]
            fn append_to(self, cord: &mut Cord) {
                let $v = &self;
                let capacity = $cap;
                cord.append_owned(self, capacity);
            }
            #[inline]
            fn prepend_to(self, cord: &mut Cord) {
                let $v = &self;
                let capacity = $cap;
                cord.prepend_owned(self, capacity);
            }
        }
    )*};
}

impl_owned_source! {
    Arc<[u8]> => |v| v.len(),
    Arc<str> => |v| v.len(),
    Arc<Vec<u8>> => |v| v.len(),
    Arc<String> => |v| v.len(),
}

macro_rules! impl_global_source {
    ($($t:ty),* $(,)?) => {$(
        impl sealed::SourceSealed for $t {}
        impl CordSource for $t {
            #[inline]
            fn append_to(self, cord: &mut Cord) {
                cord.append_global(self);
            }
            #[inline]
            fn prepend_to(self, cord: &mut Cord) {
                cord.prepend_global(self);
            }
        }
    )*};
}

impl_global_source!(Vec<u8>, String, Box<[u8]>);

impl sealed::SourceSealed for Box<str> {}
impl CordSource for Box<str> {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        self.into_boxed_bytes().append_to(cord);
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        self.into_boxed_bytes().prepend_to(cord);
    }
}

#[cfg(feature = "bytes")]
impl sealed::SourceSealed for bytes::Bytes {}
#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
impl CordSource for bytes::Bytes {
    #[inline]
    fn append_to(self, cord: &mut Cord) {
        let capacity = self.len();
        cord.append_owned(self, capacity);
    }
    #[inline]
    fn prepend_to(self, cord: &mut Cord) {
        let capacity = self.len();
        cord.prepend_owned(self, capacity);
    }
}

/// Types usable as an index in [`Cord::get`]: `usize` (yielding a byte) and
/// the range types (yielding a sub-cord). Sealed.
pub trait CordIndex: sealed::IndexSealed {
    /// `u8` for `usize`, [`Cord`] for ranges.
    type Output;
    /// Returns the byte or sub-cord at `self`, or `None` if out of bounds.
    fn get(self, cord: &Cord) -> Option<Self::Output>;
}

impl sealed::IndexSealed for usize {}
impl CordIndex for usize {
    type Output = u8;
    #[inline]
    fn get(self, cord: &Cord) -> Option<u8> {
        cord.get_byte(self)
    }
}

macro_rules! impl_range_index {
    ($($t:ty),* $(,)?) => {$(
        impl sealed::IndexSealed for $t {}
        impl CordIndex for $t {
            type Output = Cord;
            #[inline]
            fn get(self, cord: &Cord) -> Option<Cord> {
                let (pos, new_size) = crate::cord::try_resolve_range(self, cord.len())?;
                Some(cord.subcord(pos, new_size))
            }
        }
    )*};
}

impl_range_index!(
    Range<usize>,
    RangeInclusive<usize>,
    RangeFrom<usize>,
    RangeTo<usize>,
    RangeToInclusive<usize>,
    RangeFull,
    (Bound<usize>, Bound<usize>),
);

/// Byte sequences a [`Cord`] can be compared with and searched for:
/// [`Cord`], `[u8]`, `str`, `Vec<u8>`, `String`, `[u8; N]`, [`CordBuffer`]
/// and references to those.
///
/// Used by [`Cord::compare`], [`Cord::starts_with`], [`Cord::ends_with`],
/// [`Cord::contains`], [`Cord::find`] and the `PartialEq` / `PartialOrd`
/// impls.
///
/// This trait is sealed and cannot be implemented outside this crate.
#[diagnostic::on_unimplemented(
    message = "a `Cord` cannot be compared with `{Self}`",
    note = "cords compare with `Cord`, `[u8]`, `str`, `Vec<u8>`, `String`, `[u8; N]`, `CordBuffer` and references to those"
)]
pub trait CordLike: sealed::LikeSealed {
    /// Number of bytes.
    #[doc(hidden)]
    fn len(&self) -> usize;
    /// Returns `true` if empty.
    #[doc(hidden)]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The first contiguous chunk (empty if empty).
    #[doc(hidden)]
    fn first_chunk(&self) -> &[u8];
    /// Iterator over the contiguous chunks.
    #[doc(hidden)]
    fn chunks(&self) -> Chunks<'_>;
    /// `Some` if `self` is a `Cord`, enabling representation based fast
    /// paths.
    #[doc(hidden)]
    #[inline]
    fn as_cord(&self) -> Option<&Cord> {
        None
    }
}

impl sealed::LikeSealed for Cord {}
impl CordLike for Cord {
    #[inline]
    fn len(&self) -> usize {
        Cord::len(self)
    }
    #[inline]
    fn first_chunk(&self) -> &[u8] {
        Cord::first_chunk(self)
    }
    #[inline]
    fn chunks(&self) -> Chunks<'_> {
        Cord::chunks(self)
    }
    #[inline]
    fn as_cord(&self) -> Option<&Cord> {
        Some(self)
    }
}

macro_rules! impl_slice_like {
    ($($(#[$meta:meta])* $t:ty => |$v:ident| $slice:expr),* $(,)?) => {$(
        $(#[$meta])*
        impl sealed::LikeSealed for $t {}
        $(#[$meta])*
        impl CordLike for $t {
            #[inline]
            fn len(&self) -> usize {
                let $v = self;
                let s: &[u8] = $slice;
                s.len()
            }
            #[inline]
            fn first_chunk(&self) -> &[u8] {
                let $v = self;
                $slice
            }
            #[inline]
            fn chunks(&self) -> Chunks<'_> {
                let $v = self;
                Chunks::single($slice)
            }
        }
    )*};
}

impl_slice_like! {
    [u8] => |v| v,
    str => |v| v.as_bytes(),
    Vec<u8> => |v| v.as_slice(),
    String => |v| v.as_bytes(),
    CordBuffer => |v| v.as_slice(),
}

impl<const N: usize> sealed::LikeSealed for [u8; N] {}
impl<const N: usize> CordLike for [u8; N] {
    #[inline]
    fn len(&self) -> usize {
        N
    }
    #[inline]
    fn first_chunk(&self) -> &[u8] {
        self
    }
    #[inline]
    fn chunks(&self) -> Chunks<'_> {
        Chunks::single(self)
    }
}

impl<T: CordLike + ?Sized> sealed::LikeSealed for &T {}
impl<T: CordLike + ?Sized> CordLike for &T {
    #[inline]
    fn len(&self) -> usize {
        T::len(self)
    }
    #[inline]
    fn first_chunk(&self) -> &[u8] {
        T::first_chunk(self)
    }
    #[inline]
    fn chunks(&self) -> Chunks<'_> {
        T::chunks(self)
    }
    #[inline]
    fn as_cord(&self) -> Option<&Cord> {
        T::as_cord(self)
    }
}

#[cfg(feature = "bytes")]
impl sealed::LikeSealed for bytes::Bytes {}
#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
impl CordLike for bytes::Bytes {
    #[inline]
    fn len(&self) -> usize {
        bytes::Bytes::len(self)
    }
    #[inline]
    fn first_chunk(&self) -> &[u8] {
        self
    }
    #[inline]
    fn chunks(&self) -> Chunks<'_> {
        Chunks::single(self)
    }
}
