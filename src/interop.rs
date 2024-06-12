use std::alloc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::hvm::*;

#[cfg(feature = "c")]
extern "C" {
  pub fn hvm_c(book_buffer: *const u32, return_output: u8) -> *mut OutputNetC;
  pub fn free_output_net(net: *mut OutputNetC);
}

#[cfg(feature = "cuda")]
extern "C" {
  pub fn hvm_cu(book_buffer: *const u32, return_output: bool);
}

// Abstract Global Net
// Allows any global net to be read back
// -------------

pub trait NetReadback {
  fn run(book: &Book) -> Self;
  fn node_load(&self, loc: usize) -> Pair;
  fn vars_exchange(&mut self, var: usize, val: Port) -> Port;
  fn itrs(&self) -> u64;

  fn vars_take(&mut self, var: usize) -> Port {
    self.vars_exchange(var, Port(0))
  }

  fn enter(&mut self, mut var: Port) -> Port {
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

impl<'a> NetReadback for GNet<'a> {
  fn run(book: &Book) -> Self {
    // Initializes the global net
    let net = GNet::new(1 << 29, 1 << 29);

    // Initializes threads
    let mut tm = TMem::new(0, 1);

    // Creates an initial redex that calls main
    let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
    tm.rbag.push_redex(Pair::new(Port::new(REF, main_id as u32), ROOT));
    net.vars_create(ROOT.get_val() as usize, NONE);

    // Evaluates
    tm.evaluator(&net, &book);

    net
  }
  fn node_load(&self, loc: usize) -> Pair { self.node_load(loc) }
  fn vars_exchange(&mut self, var: usize, val: Port) -> Port { GNet::vars_exchange(self, var, val) }
  fn itrs(&self) -> u64 { self.itrs.load(Ordering::Relaxed) }
}

// Global Net equivalent to the C implementation.
// NOTE: If the C struct `Net` changes, this has to change as well.
// TODO: use `bindgen` crate (https://github.com/rust-lang/rust-bindgen) to generate C structs
// -------------

#[cfg(feature = "c")]
#[repr(C)]
pub struct NetC {
  raw: *mut OutputNetC
}

#[cfg(feature = "c")]
#[repr(C)]
pub struct OutputNetC {
  pub original: *mut std::ffi::c_void,
  pub node_buf: *mut APair, // global node buffer
  pub vars_buf: *mut APort, // global vars buffer
  pub itrs: AtomicU64, // interaction count
}

#[cfg(feature = "c")]
impl NetC {
  pub fn net<'a>(&'a self) -> &'a OutputNetC {
    unsafe { &*self.raw }
  }

  pub fn net_mut<'a>(&'a mut self) -> &'a mut OutputNetC {
    unsafe { &mut *self.raw }
  }
}

#[cfg(feature = "c")]
impl Drop for NetC {
  fn drop(&mut self) {
    // Deallocate network's memory
    unsafe { free_output_net(self.raw); }
  }
}

#[cfg(feature = "c")]
impl NetReadback for NetC {
  fn run(book: &Book) -> Self {
    // Serialize book
    let mut data : Vec<u8> = Vec::new();
    book.to_buffer(&mut data);
    //println!("{:?}", data);
    let book_buffer = data.as_mut_ptr() as *mut u32;

    // Run net
    let raw = unsafe { hvm_c(data.as_mut_ptr() as *mut u32, 1) };

    NetC { raw }
  }

  fn node_load(&self, loc:usize) -> Pair {
    unsafe {
      Pair((*self.net().node_buf.add(loc)).0.load(Ordering::Relaxed))
    }
  }

  fn vars_exchange(&mut self, var: usize, val: Port) -> Port {
    unsafe {
      Port((*self.net().vars_buf.add(var)).0.swap(val.0, Ordering::Relaxed) as u32)
    }
  }

  fn itrs(&self) -> u64 {
    self.net().itrs.load(Ordering::Relaxed)
  }
}

// Global Net equivalent to the CUDA implementation.
// NOTE: If the CUDA struct `Net` changes, this has to change as well.
// -------------

// TODO
// Problem: CUDA's `GNet` is allocated using `cudaMalloc`
// Solution: Write a CUDA kernel to compact GPU memory and then `memcpy` it to RAM

// #[cfg(feature = "cuda")]
#[repr(C)]
pub struct NetCuda {
  raw: *mut RawNetCuda
}

// #[cfg(feature = "cuda")]
#[repr(C)]
pub struct RawNetCuda {
  pub rbag_use_a: u32, // total rbag redex count (buffer A)
  pub rbag_use_b: u32, // total rbag redex count (buffer B)
  pub rbag_buf_a: [Pair; NetCuda::G_RBAG_LEN], // global redex bag (buffer A)
  pub rbag_buf_b: [Pair; NetCuda::G_RBAG_LEN], // global redex bag (buffer B)
  pub node_buf: [Pair; NetCuda::G_NODE_LEN], // global node buffer
  pub vars_buf: [Port; NetCuda::G_VARS_LEN], // global vars buffer
  pub node_put: [u32; NetCuda::TPB * NetCuda::BPG],
  pub vars_put: [u32; NetCuda::TPB * NetCuda::BPG],
  pub rbag_put: [u32; NetCuda::TPB * NetCuda::BPG],
  pub mode: u8, // evaluation mode (curr)
  pub itrs: u64, // interaction count
  pub iadd: u64, // interaction count adder
  pub leak: u64, // leak count
  pub turn: u32, // turn count
  pub down: u8, // are we recursing down?
  pub rdec: u8, // decrease rpos by 1?
}

// #[cfg(feature = "cuda")]
impl NetCuda {
  // Constants relevant in the CUDA implementation
  // NOTE: If any of these constants are changed in CUDA, they have to be changed here as well.
  
  // Threads per Block
  pub const TPB_L2: usize = 7;
  pub const TPB: usize    = 1 << NetCuda::TPB_L2;
  
  // Blocks per GPU
  pub const BPG_L2: usize = 7;
  pub const BPG: usize    = 1 << NetCuda::BPG_L2;

  // Thread Redex Bag Length
  pub const RLEN: usize   = 256;

  pub const G_NODE_LEN: usize = 1 << 29; // max 536m nodes
  pub const G_VARS_LEN: usize = 1 << 29; // max 536m vars
  pub const G_RBAG_LEN: usize = NetCuda::TPB * NetCuda::BPG * NetCuda::RLEN * 3;

  pub fn net(&self) -> &RawNetCuda {
    unsafe { &*self.raw }
  }

  pub fn net_mut(&mut self) -> &mut RawNetCuda {
    unsafe { &mut *self.raw }
  }

  fn vars_exchange(&mut self, var: usize, val: Port) -> Port {
    let net = self.net_mut();
    let old = net.vars_buf[var];
    net.vars_buf[var] = val;
    old
  }
}
