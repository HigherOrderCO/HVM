use std::alloc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::hvm::*;

#[cfg(feature = "c")]
extern "C" {
  fn hvm_c(book_buffer: *const u32, net_buffer: *const NetC, run_io: bool);
}

#[cfg(feature = "cuda")]
extern "C" {
  fn hvm_cu(book_buffer: *const u32, run_io: bool);
}

// Abstract Global Net
// Allows any global net to be read back
// -------------

pub trait NetReadback {
  fn run<T>(book: &Book, before: impl FnOnce() -> T, after: impl FnOnce(&Self, &Book, T) -> ());
  fn enter(&self, var: Port) -> Port;
  fn node_load(&self, loc: usize) -> Pair;
  fn itrs(&self) -> u64;
}

impl<'a> NetReadback for GNet<'a> {
  fn run<T>(book: &Book, before: impl FnOnce() -> T, after: impl FnOnce(&Self, &Book, T) -> ()) {
    // Initializes the global net
    let net = GNet::new(1 << 29, 1 << 29);

    // Initializes threads
    let mut tm = TMem::new(0, 1);

    // Creates an initial redex that calls main
    let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
    tm.rbag.push_redex(Pair::new(Port::new(REF, main_id as u32), ROOT));
    net.vars_create(ROOT.get_val() as usize, NONE);

    let initial_state = before();

    // Evaluates
    tm.evaluator(&net, &book);

    after(&net, book, initial_state);
  }
  fn enter(&self, var: Port) -> Port { self.enter(var) }
  fn node_load(&self, loc: usize) -> Pair { self.node_load(loc) }
  fn itrs(&self) -> u64 { self.itrs.load(Ordering::Relaxed) }
}

// Global Net equivalent to the C implementation.
// NOTE: If the C struct `Net` changes, this has to change as well.
// TODO: use `bindgen` crate (https://github.com/rust-lang/rust-bindgen) to generate C structs
// -------------

#[cfg(feature = "c")]
#[repr(C)]
pub struct NetC {
  pub node: [APair; NetC::G_NODE_LEN], // global node buffer
  pub vars: [APort; NetC::G_VARS_LEN], // global vars buffer
  pub rbag: [APair; NetC::G_RBAG_LEN], // global rbag buffer
  pub itrs: AtomicU64, // interaction count
  pub idle: AtomicU32, // idle thread counter
}

#[cfg(feature = "c")]
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

#[cfg(feature = "c")]
impl NetReadback for NetC {
  fn run<T>(book: &Book, before: impl FnOnce() -> T, after: impl FnOnce(&Self, &Book, T) -> ()) {
    // Serialize book
    let mut data : Vec<u8> = Vec::new();
    book.to_buffer(&mut data);
    //println!("{:?}", data);
    let book_buffer = data.as_mut_ptr() as *mut u32;

    let layout = alloc::Layout::new::<NetC>();
    let net_ptr = unsafe { alloc::alloc(layout) as *mut NetC };

    let initial_state = before();

    unsafe {
      hvm_c(data.as_mut_ptr() as *mut u32, net_ptr, true);
    }

    // Converts the raw pointer to a reference
    let net_ref = unsafe { &mut *net_ptr };

    after(net_ref, book, initial_state);

    // Deallocate network's memory
    unsafe { alloc::dealloc(net_ptr as *mut u8, layout) };
  }
  fn node_load(&self, loc:usize) -> Pair { self.node_load(loc) }
  fn enter(&self, var: Port) -> Port { self.enter(var) } 
  fn itrs(&self) -> u64 { self.itrs.load(Ordering::Relaxed) }
}

// Global Net equivalent to the CUDA implementation.
// NOTE: If the CUDA struct `Net` changes, this has to change as well.
// -------------

// TODO
// Problem: CUDA's `GNet` is allocated using `cudaMalloc`
// Solution: Write a CUDA kernel to compact GPU memory and then `memcpy` it to RAM
