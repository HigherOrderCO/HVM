#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

pub mod base;
pub mod data;
pub mod rule;
use sysinfo::{System, SystemExt};

pub use base::{*};
pub use data::{*};
pub use rule::{*};

//pub struct Runtime {
  //heap: runtime::Heap,
  //prog: runtime::Program,
  //book: language::rulebook::RuleBook,
//}

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

// If unspecified, allocates `max(16 GB, 75% free_sys_mem)` memory
pub fn default_heap_size() -> usize {
  use sysinfo::SystemExt;
  let system = System::new_all();
  let available_memory = system.free_memory();
  let heap_size = (available_memory * 3 / 4) / 8;
  let heap_size = std::cmp::min(heap_size as usize, 16 * CELLS_PER_GB);
  return heap_size as usize;
}

// If unspecified, spawns 1 thread for each available core
pub fn default_heap_tids() -> usize {
  return std::thread::available_parallelism().unwrap().get();
}


