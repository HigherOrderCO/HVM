// Redex Bag
// ---------
// Concurrent bag featuring insert, read and modify. No pop.

use crossbeam::utils::{CachePadded};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const REDEX_BAG_SIZE : usize = 1 << 24;
pub const REDEX_CONT_RET : u64 = 0xFFFFFF; // signals to return

// - 32 bits: host
// - 24 bits: cont
// -  8 bits: left
pub type Redex = u64;

pub struct RedexBag {
  tids: usize,
  next: Box<[CachePadded<AtomicUsize>]>,
  data: Box<[AtomicU64]>,
}

pub fn new_redex(host: u64, cont: u64, left: u64) -> Redex {
  return (host << 32) | (cont << 8) | left;
}

pub fn get_redex_host(redex: Redex) -> u64 {
  return redex >> 32;
}

pub fn get_redex_cont(redex: Redex) -> u64 {
  return (redex >> 8) & 0xFFFFFF;
}

pub fn get_redex_left(redex: Redex) -> u64 {
  return redex & 0xFF;
}

impl RedexBag {
  pub fn new(tids: usize) -> RedexBag {
    let mut next = vec![];
    for _ in 0 .. tids {
      next.push(CachePadded::new(AtomicUsize::new(0)));
    }
    let next = next.into_boxed_slice();
    let data = crate::runtime::new_atomic_u64_array(REDEX_BAG_SIZE);
    return RedexBag { tids, next, data };
  }

  pub fn min_index(&self, tid: usize) -> usize {
    return REDEX_BAG_SIZE / self.tids * (tid + 0);
  }

  pub fn max_index(&self, tid: usize) -> usize {
    return std::cmp::min(REDEX_BAG_SIZE / self.tids * (tid + 1), REDEX_CONT_RET as usize - 1);
  }

  pub fn insert(&self, tid: usize, redex: u64) -> u64 {
    loop {
      let index = self.next[tid].fetch_add(1, Ordering::Relaxed);
      if index + 1 >= self.max_index(tid) { 
        self.next[tid].store(self.min_index(tid), Ordering::Relaxed);
      }
      if self.data[index].compare_exchange_weak(0, redex, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        return index as u64;
      }
    }
  }

  pub fn complete(&self, index: u64) -> Option<(u64,u64)> {
    let redex = self.data[index as usize].fetch_sub(1, Ordering::Relaxed);
    if get_redex_left(redex) == 1 {
      self.data[index as usize].store(0, Ordering::Relaxed);
      return Some((get_redex_host(redex), get_redex_cont(redex)));
    } else {
      return None;
    }
  }
}

