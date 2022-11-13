#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

pub mod base;
pub mod data;
pub mod rule;

pub use base::{*};
pub use data::{*};
pub use rule::{*};

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

pub fn default_heap_size() -> usize {
  return 16 * CELLS_PER_GB;
}

pub fn default_heap_tids() -> usize {
  return std::thread::available_parallelism().unwrap().get();
}
