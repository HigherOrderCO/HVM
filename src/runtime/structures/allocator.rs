use crossbeam::utils::{CachePadded};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// Allocator
// ---------

pub struct AllocatorNext {
  pub cell: AtomicU64,
  pub area: AtomicU64,
}

pub struct Allocator {
  pub tids: usize,
  pub data: Box<[AtomicU64]>,
  pub used: Box<[AtomicU64]>,
  pub next: Box<[CachePadded<AllocatorNext>]>,
}

pub const PAGE_SIZE : usize = 4096;

impl Allocator {

  pub fn new(tids: usize) -> Allocator {
    let mut next = vec![];
    for i in 0 .. tids {
      let cell = AtomicU64::new(u64::MAX);
      let area = AtomicU64::new((crate::runtime::HEAP_SIZE / PAGE_SIZE / tids * i) as u64);
      next.push(CachePadded::new(AllocatorNext { cell, area }));
    }
    let data = crate::runtime::new_atomic_u64_array(crate::runtime::HEAP_SIZE);
    let used = crate::runtime::new_atomic_u64_array(crate::runtime::HEAP_SIZE / PAGE_SIZE);
    let next = next.into_boxed_slice();
    Allocator { tids, data, used, next }
  }

  pub fn alloc(&self, tid: usize, arity: u64) -> u64 {
    let next = unsafe { self.next.get_unchecked(tid) };
    // Attempts to allocate on this thread's owned area
    let aloc = next.cell.fetch_add(arity, Ordering::Relaxed);
    let area = aloc / PAGE_SIZE as u64;
    if aloc != u64::MAX && (aloc + arity) / PAGE_SIZE as u64 == area {
      unsafe { self.used.get_unchecked(area as usize) }.fetch_add(arity, Ordering::Relaxed);
      //println!("[{}] old_alloc {} at {}, used={} ({} {})", tid, arity, aloc, self.used[area as usize].load(Ordering::Relaxed), area, (aloc + arity) / PAGE_SIZE as u64);
      return aloc;
    }
    // If we can't, attempt to allocate on a new area
    let mut area = next.area.load(Ordering::Relaxed)  % ((crate::runtime::HEAP_SIZE / PAGE_SIZE) as u64);
    loop {
      if unsafe { self.used.get_unchecked(area as usize) }.compare_exchange_weak(0, arity, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        let aloc = area * PAGE_SIZE as u64;
        next.cell.store(aloc + arity, Ordering::Relaxed);
        next.area.store((area + 1) % ((crate::runtime::HEAP_SIZE / PAGE_SIZE) as u64), Ordering::Relaxed);
        //println!("[{}] new_alloc {} at {}, used={}", tid, arity, aloc, self.used[area as usize].load(Ordering::Relaxed));
        return aloc;
      } else {
        area = (area + 1) % ((crate::runtime::HEAP_SIZE / PAGE_SIZE) as u64);
      }
    }
  }

  pub fn free(&self, tid: usize, loc: u64, arity: u64) {
    //for i in 0 .. arity { unsafe { self.data.get_unchecked((loc + i) as usize) }.store(0, Ordering::Relaxed); }
    let area = loc / PAGE_SIZE as u64;
    let used = unsafe { self.used.get_unchecked(area as usize) }.fetch_sub(arity, Ordering::Relaxed);
    //println!("[{}] free {} at {}, used={}", tid, arity, loc, self.used[area as usize].load(Ordering::Relaxed));
  }

}
