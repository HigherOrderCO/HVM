use std::alloc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::hvm::*;

#[cfg(feature = "c")]
extern "C" {
  pub fn hvm_c(book_buffer: *const u32, return_output: u8) -> *mut OutputNetC;
  pub fn free_output_net_c(net: *mut OutputNetC);
}

#[cfg(feature = "cuda")]
extern "C" {
  pub fn hvm_cu(book_buffer: *const u32, return_output: bool) -> *mut OutputNetCuda;
  pub fn free_output_net_cuda(net: *mut OutputNetCuda);
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
    unsafe { free_output_net_c(self.raw); }
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

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct NetCuda {
  raw: *mut OutputNetCuda
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct OutputNetCuda {
  pub original: *mut std::ffi::c_void,
  pub node_buf: *mut Pair, // global node buffer
  pub vars_buf: *mut Port, // global vars buffer
  pub itrs: u64, // interaction count
}

#[cfg(feature = "cuda")]
impl NetCuda {
  pub fn net(&self) -> &OutputNetCuda {
    unsafe { &*self.raw }
  }

  pub fn net_mut(&mut self) -> &mut OutputNetCuda {
    unsafe { &mut *self.raw }
  }
}

#[cfg(feature = "cuda")]
impl Drop for NetCuda {
  fn drop(&mut self) {
    // Deallocate network's memory
    unsafe { free_output_net_cuda(self.raw); }
  }
}

#[cfg(feature = "cuda")]
impl NetReadback for NetCuda {
  fn run(book: &Book) -> Self {
    // Serialize book
    let mut data : Vec<u8> = Vec::new();
    book.to_buffer(&mut data);
    //println!("{:?}", data);
    let book_buffer = data.as_mut_ptr() as *mut u32;

    // Run net
    let raw = unsafe { hvm_cu(data.as_mut_ptr() as *mut u32, true) };

    NetCuda { raw }
  }

  fn node_load(&self, loc:usize) -> Pair {
    unsafe {
      Pair((*self.net().node_buf.add(loc)).0)
    }
  }

  fn vars_exchange(&mut self, var: usize, val: Port) -> Port {
    unsafe {
      let net = self.net_mut();
      let old = *net.vars_buf.add(var);
      *net.vars_buf.add(var) = val;
      old
    }
  }

  fn itrs(&self) -> u64 {
    self.net().itrs
  }
}
