use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::hvm::*;

// Abstract Global Net
// Allows any global net to be read back
// -------------

pub trait NetReadback {
  fn enter(&self, var: Port) -> Port;
  fn node_load(&self, loc: usize) -> Pair;
}

impl<'a> NetReadback for GNet<'a> {
  fn enter(&self, var: Port) -> Port { self.enter(var) }
  fn node_load(&self, loc: usize) -> Pair { self.node_load(loc) }
}

// Global Net equivalent to the C implementation.
// NOTE: If the C struct `Net` changes, this has to change as well.
// TODO: use `bindgen` crate (https://github.com/rust-lang/rust-bindgen) to generate C structs
// -------------

#[repr(C)]
pub struct NetC {
  pub node: [APair; NetC::G_NODE_LEN], // global node buffer
  pub vars: [APort; NetC::G_VARS_LEN], // global vars buffer
  pub rbag: [APair; NetC::G_RBAG_LEN], // global rbag buffer
  pub itrs: AtomicU64, // interaction count
  pub idle: AtomicU32, // idle thread counter
}

impl NetC {
  // Constants relevant in the C implementation
  // NOTE: If any of these constants are changed in C, they have to be changed here as well.
  pub const TPC_L2: usize = 3;
  pub const TPC: usize = 1 << NetC::TPC_L2;
  pub const CACHE_PAD: usize = 64; // Cache padding

  pub const HLEN: usize = 1 << 16; // max 16k high-priority redexes
  pub const RLEN: usize = 1 << 24; // max 16m low-priority redexes
  pub const G_NODE_LEN: usize = 1 << 29; // max 536m nodes
  pub const G_VARS_LEN: usize = 1 << 29; // max 536m vars
  pub const G_RBAG_LEN: usize = NetC::TPC * NetC::RLEN;

  pub fn vars_exchange(&self, var: usize, val: Port) -> Port {
    Port(self.vars[var].0.swap(val.0, Ordering::Relaxed) as u32)
  }

  pub fn vars_take(&self, var: usize) -> Port {
    self.vars_exchange(var, Port(0))
  }

  fn node_load(&self, loc:usize) -> Pair {
    Pair(self.node[loc].0.load(Ordering::Relaxed))
  }

  fn enter(&self, mut var: Port) -> Port {
    // While `B` is VAR: extend it (as an optimization)
    while var.get_tag() == VAR {
      // Takes the current `B` substitution as `B'`
      let val = self.vars_exchange(var.get_val() as usize, NONE);
      // If there was no `B'`, stop, as there is no extension
      if val == NONE || val == Port(0) {
        break;
      }
      // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
      self.vars_take(var.get_val() as usize);
      var = val;
    }
    return var;
  }
}

impl NetReadback for NetC {
  fn node_load(&self, loc:usize) -> Pair {
    self.node_load(loc)
  }
  
  fn enter(&self, var: Port) -> Port {
    self.enter(var)
  } 
}

// Global Net equivalent to the CUDA implementation.
// NOTE: If the CUDA struct `Net` changes, this has to change as well.
// -------------

// TODO
// Problem: CUDA's `GNet` is allocated using `cudaMalloc`
// Solution: Write a CUDA kernel to compact GPU memory and then `memcpy` it to RAM
