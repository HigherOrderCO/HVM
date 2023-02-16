// Redex Bag
// ---------
// Concurrent bag featuring insert, read and modify. No pop.

use crossbeam::utils::CachePadded;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const REDEX_BAG_SIZE: usize = 1 << 26;
pub const REDEX_CONT_RET: u64 = 0x3FFFFFF; // signals to return

// - 32 bits: host
// - 26 bits: cont
// -  6 bits: left
pub type Redex = u64;

pub struct RedexBag {
  tids: usize,
  next: Box<[CachePadded<AtomicUsize>]>,
  data: Box<[AtomicU64]>,
}

pub fn new_redex(host: u64, cont: u64, left: u64) -> Redex {
  (host << 32) | (cont << 6) | left
}

pub fn get_redex_host(redex: Redex) -> u64 {
  redex >> 32
}

pub fn get_redex_cont(redex: Redex) -> u64 {
  (redex >> 6) & 0x3FFFFFF
}

pub fn get_redex_left(redex: Redex) -> u64 {
  redex & 0x3F
}

impl RedexBag {
  pub fn new(tids: usize) -> Self {
    let mut next = vec![];
    for _ in 0..tids {
      next.push(CachePadded::new(0.into()));
    }
    let next = next.into_boxed_slice();
    let data = crate::runtime::new_atomic_u64_array(REDEX_BAG_SIZE);
    Self { tids, next, data }
  }

  //pub fn min_index(&self, tid: usize) -> usize {
  //return REDEX_BAG_SIZE / self.tids * (tid + 0);
  //}

  //pub fn max_index(&self, tid: usize) -> usize {
  //return std::cmp::min(REDEX_BAG_SIZE / self.tids * (tid + 1), REDEX_CONT_RET as usize - 1);
  //}

  #[inline(always)]
  pub fn insert(&self, tid: usize, redex: u64) -> u64 {
    loop {
      let index = unsafe { self.next.get_unchecked(tid) }.fetch_add(1, Ordering::Relaxed);
      if index + 2 >= REDEX_BAG_SIZE {
        unsafe { self.next.get_unchecked(tid) }.store(0, Ordering::Relaxed);
      }
      if unsafe { self.data.get_unchecked(index) }
        .compare_exchange_weak(0, redex, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
      {
        return index as u64;
      }
    }
  }

  #[inline(always)]
  pub fn complete(&self, index: u64) -> Option<(u64, u64)> {
    let redex = unsafe { self.data.get_unchecked(index as usize) }.fetch_sub(1, Ordering::Relaxed);
    if get_redex_left(redex) == 1 {
      unsafe { self.data.get_unchecked(index as usize) }.store(0, Ordering::Relaxed);
      Some((get_redex_cont(redex), get_redex_host(redex)))
    } else {
      None
    }
  }
}
