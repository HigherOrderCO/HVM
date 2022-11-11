#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

pub mod builtins;
pub mod debug;
pub mod memory;
pub mod program;
pub mod reducer;
pub mod rewriters;
pub mod structures;

pub use crate::runtime::builtins::{*};
pub use crate::runtime::debug::{*};
pub use crate::runtime::memory::{*};
pub use crate::runtime::program::{*};
pub use crate::runtime::reducer::{*};
pub use crate::runtime::rewriters::{*};
pub use crate::runtime::structures::allocator::{*};
pub use crate::runtime::structures::redex_bag::{*};
pub use crate::runtime::structures::u64_map::{*};
pub use crate::runtime::structures::visit_queue::{*};

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

pub const HEAP_SIZE: usize = 24 * CELLS_PER_GB;

pub fn available_parallelism() -> usize {
  return std::thread::available_parallelism().unwrap().get();
}

pub fn new_tids(tids: usize) -> Box<[usize]> {
  return (0 .. tids).collect::<Vec<usize>>().into_boxed_slice();
}
