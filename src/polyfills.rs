//! This module contains functions that act as a polyfill for unstable standard library functions
//! that we use in this crate. As these features become stabilized, we should remove these functions
//! and use the standard library implementations.

use std::{
  cell::UnsafeCell,
  sync::atomic::{AtomicU64, AtomicU8},
};

pub trait PolyfillAtomicFromMutSlice: Sized {
  type Int;
  fn polyfill_from_mut_slice(v: &mut [Self::Int]) -> &mut [Self];
}

pub trait PolyfillAtomicAsMutPtr {
  type Int;
  fn polyfill_as_mut_ptr(&self) -> *mut Self::Int;
}

impl PolyfillAtomicAsMutPtr for AtomicU8 {
  type Int = u8;

  fn polyfill_as_mut_ptr(&self) -> *mut Self::Int {
    unsafe { (*(self as *const Self as *const UnsafeCell<Self::Int>)).get() }
  }
}

impl PolyfillAtomicAsMutPtr for AtomicU64 {
  type Int = u64;

  fn polyfill_as_mut_ptr(&self) -> *mut Self::Int {
    unsafe { (*(self as *const Self as *const UnsafeCell<Self::Int>)).get() }
  }
}

#[cfg(
// Here we reference a custom cfg check to avoid the nightly `target_has_atomic_equal_alignment`
// cfg. The logic for supported platforms is implemented in `build.rs`
stable_target_has_atomic_equal_alignment
  )]
impl PolyfillAtomicFromMutSlice for AtomicU8 {
  type Int = u8;

  fn polyfill_from_mut_slice(v: &mut [u8]) -> &mut [Self] {
    use std::mem::align_of;
    let [] = [(); align_of::<Self>() - align_of::<u8>()];
    // SAFETY:
    //  - the mutable reference guarantees unique ownership.
    //  - the alignment of u8 and AtomicU8 is equal, according to our cfg check above.
    unsafe { &mut *(v as *mut [u8] as *mut [AtomicU8]) }
  }
}

#[cfg(
// Here we reference a custom cfg check to avoid the nightly `target_has_atomic_equal_alignment`
// cfg. The logic for supported platforms is implemented in `build.rs`
stable_target_has_atomic_equal_alignment
  )]
impl PolyfillAtomicFromMutSlice for AtomicU64 {
  type Int = u64;

  fn polyfill_from_mut_slice(v: &mut [u64]) -> &mut [Self] {
    use std::mem::align_of;
    let [] = [(); align_of::<Self>() - align_of::<u64>()];
    // SAFETY:
    //  - the mutable reference guarantees unique ownership.
    //  - the alignment of u8 and AtomicU8 is equal, according to our cfg check above.
    unsafe { &mut *(v as *mut [u64] as *mut [AtomicU64]) }
  }
}
