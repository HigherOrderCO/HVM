// Visit Queue
// -----------
// A concurrent task-stealing queue featuring push, pop and steal.

use crossbeam::utils::CachePadded;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const VISIT_QUEUE_SIZE: usize = 1 << 24;

// - 32 bits: host
// - 32 bits: cont
pub type Visit = u64;

pub struct VisitQueue {
  pub init: CachePadded<AtomicUsize>,
  pub last: CachePadded<AtomicUsize>,
  pub data: Box<[AtomicU64]>,
}

pub fn new_visit(host: u64, hold: bool, cont: u64) -> Visit {
  (host << 32) | (if hold { 0x80000000 } else { 0 }) | cont
}

pub fn get_visit_host(visit: Visit) -> u64 {
  visit >> 32
}

pub fn get_visit_hold(visit: Visit) -> bool {
  (visit >> 31) & 1 == 1
}

pub fn get_visit_cont(visit: Visit) -> u64 {
  visit & 0x3FFFFFF
}

impl VisitQueue {
  pub fn new() -> VisitQueue {
    VisitQueue {
      init: CachePadded::new(AtomicUsize::new(0)),
      last: CachePadded::new(AtomicUsize::new(0)),
      data: crate::runtime::new_atomic_u64_array(VISIT_QUEUE_SIZE),
    }
  }

  pub fn push(&self, value: u64) {
    let index = self.last.fetch_add(1, Ordering::Relaxed);
    unsafe { self.data.get_unchecked(index) }.store(value, Ordering::Relaxed);
  }

  #[inline(always)]
  pub fn pop(&self) -> Option<(u64, u64)> {
    loop {
      let last = self.last.load(Ordering::Relaxed);
      if last > 0 {
        self.last.fetch_sub(1, Ordering::Relaxed);
        self.init.fetch_min(last - 1, Ordering::Relaxed);
        let visit = unsafe { self.data.get_unchecked(last - 1) }.swap(0, Ordering::Relaxed);
        if visit == 0 {
          continue;
        } else {
          return Some((get_visit_cont(visit), get_visit_host(visit)));
        }
      } else {
        return None;
      }
    }
  }

  #[inline(always)]
  pub fn steal(&self) -> Option<(u64, u64)> {
    let index = self.init.load(Ordering::Relaxed);
    let visit = unsafe { self.data.get_unchecked(index) }.load(Ordering::Relaxed);
    if visit != 0 && !get_visit_hold(visit) {
      if let Ok(visit) = unsafe { self.data.get_unchecked(index) }.compare_exchange(
        visit,
        0,
        Ordering::Relaxed,
        Ordering::Relaxed,
      ) {
        self.init.fetch_add(1, Ordering::Relaxed);
        return Some((get_visit_cont(visit), get_visit_host(visit)));
      }
    }
    None
  }
}
