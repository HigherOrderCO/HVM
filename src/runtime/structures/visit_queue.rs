// Visit Queue
// -----------
// A concurrent task-stealing queue featuring push, pop and steal.

use crossbeam::utils::{CachePadded};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const VISIT_QUEUE_SIZE : usize = 1 << 24;

// - 32 bits: host
// - 32 bits: cont
pub type Visit = u64;

pub struct VisitQueue {
  pub init: CachePadded<AtomicUsize>,
  pub last: CachePadded<AtomicUsize>,
  pub data: Box<[AtomicU64]>,
}

pub fn new_visit(host: u64, cont: u64) -> Visit {
  return (host << 32) | cont;
}

pub fn get_visit_host(visit: Visit) -> u64 {
  return visit >> 32;
}

pub fn get_visit_cont(visit: Visit) -> u64 {
  return visit & 0xFFFFFFFF;
}

impl VisitQueue {

  pub fn new() -> VisitQueue {
    return VisitQueue {
      init: CachePadded::new(AtomicUsize::new(0)),
      last: CachePadded::new(AtomicUsize::new(0)),
      data: crate::runtime::new_atomic_u64_array(VISIT_QUEUE_SIZE),
    }
  }

  pub fn push(&self, value: u64) {
    let index = self.last.fetch_add(1, Ordering::Relaxed);
    self.data[index].store(value, Ordering::Relaxed);
  }

  pub fn pop(&self) -> Option<(u64, u64)> {
    loop {
      let last = self.last.load(Ordering::Relaxed);
      if last > 0 {
        self.last.fetch_sub(1, Ordering::Relaxed);
        self.init.fetch_min(last - 1, Ordering::Relaxed);
        let visit = self.data[last - 1].swap(0, Ordering::Relaxed);
        if visit == 0 {
          continue;
        } else {
          return Some((get_visit_host(visit), get_visit_cont(visit)));
        }
      } else {
        return None;
      }
    }
  }

  pub fn steal(&self) -> Option<(u64, u64)> {
    let index = self.init.load(Ordering::Relaxed);
    let visit = self.data[index].load(Ordering::Relaxed);
    if visit != 0 {
      if let Ok(visit) = self.data[index].compare_exchange(visit, 0, Ordering::Relaxed, Ordering::Relaxed) {
        self.init.fetch_add(1, Ordering::Relaxed);
        return Some((get_visit_host(visit), get_visit_cont(visit)));
      }
    }
    return None;
  }

}
