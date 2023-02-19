#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

pub mod base;
pub mod data;
pub mod rule;

use sysinfo::{RefreshKind, System, SystemExt};

pub use base::*;
pub use data::*;
pub use rule::*;

use crate::language;
use crate::language::syntax::Oper;

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

// If unspecified, allocates `max(16 GB, 75% free_sys_mem)` memory
pub fn default_heap_size() -> usize {
  use sysinfo::SystemExt;
  let available_memory = System::new_with_specifics(RefreshKind::new().with_memory()).free_memory();
  let heap_size = (available_memory * 3 / 4) / 8;
  std::cmp::min(heap_size as usize, 16 * CELLS_PER_GB)
}

// If unspecified, spawns 1 thread for each available core
pub fn default_heap_tids() -> usize {
  std::thread::available_parallelism().unwrap().get()
}

use language::rulebook::RuleBook;

pub struct Runtime {
  pub heap: Heap,
  pub prog: Program,
  pub book: language::rulebook::RuleBook,
  pub tids: Box<[usize]>,
  pub dbug: bool,
}

impl Runtime {
  /// Creates a new, empty runtime
  pub fn new(size: usize, tids: usize, dbug: bool) -> Self {
    Self {
      heap: Heap::new(size, tids),
      prog: Program::new(),
      book: RuleBook::new(),
      tids: new_tids(tids),
      dbug,
    }
  }

  /// Creates a runtime from source code, given a max number of nodes
  pub fn from_code_with(code: &str, size: usize, tids: usize, dbug: bool) -> Result<Self, String> {
    let file = language::syntax::read_file(code)?;
    let heap = crate::runtime::Heap::new(size, tids);
    let prog = Program::new();
    let book = (&file).into();
    let tids = new_tids(tids);
    Ok(Self { heap, prog, book, tids, dbug })
  }

  ////fn get_area(&mut self) -> runtime::Area {
  ////return runtime::get_area(&mut self.heap, 0)
  ////}

  /// Creates a runtime from a source code
  //#[cfg(not(target_arch = "wasm32"))]
  pub fn from_code(code: &str) -> Result<Self, String> {
    Self::from_code_with(code, default_heap_size(), default_heap_tids(), false)
  }

  ///// Extends a runtime with new definitions
  //pub fn define(&mut self, _code: &str) {
  //todo!()
  //}

  /// Allocates a new term, returns its location
  pub fn alloc_code(&mut self, code: &str) -> Result<u64, String> {
    Ok(self.alloc_term(&*language::syntax::read_term(code)?))
  }

  /// Given a location, returns the pointer stored on it
  pub fn load_ptr(&self, host: u64) -> Ptr {
    self.heap.load_ptr(host)
  }

  /// Given a location, evaluates a term to head normal form
  pub fn reduce(&mut self, host: u64) {
    self.heap.reduce(&self.prog, &self.tids, host, false, self.dbug);
  }

  /// Given a location, evaluates a term to full normal form
  pub fn normalize(&mut self, host: u64) {
    self.heap.reduce(&self.prog, &self.tids, host, true, self.dbug);
  }

  /// Evaluates a code, allocs and evaluates to full normal form. Returns its location.
  pub fn normalize_code(&mut self, code: &str) -> u64 {
    let host = self.alloc_code(code).ok().unwrap();
    self.normalize(host);
    host
  }

  /// Evaluates a code to normal form. Returns its location.
  pub fn eval_to_loc(&mut self, code: &str) -> u64 {
    self.normalize_code(code)
  }

  /// Evaluates a code to normal form.
  pub fn eval(&mut self, code: &str) -> String {
    let host = self.normalize_code(code);
    self.show(host)
  }

  //// /// Given a location, runs side-effective actions
  ////#[cfg(not(target_arch = "wasm32"))]
  ////pub fn run_io(&mut self, host: u64) {
  ////runtime::run_io(&mut self.heap, &self.prog, &[0], host)
  ////}

  /// Given a location, recovers the lambda Term stored on it, as code
  pub fn show(&self, host: u64) -> String {
    language::readback::as_code(&self.heap, &self.prog, host)
  }

  /// Given a location, recovers the linear Term stored on it, as code
  pub fn show_linear(&self, host: u64) -> String {
    self.heap.as_linear_code(&self.prog, host)
  }

  /// Return the total number of graph rewrites computed
  pub fn get_rewrites(&self) -> u64 {
    self.heap.get_cost()
  }

  /// Returns the name of a given id
  pub fn get_name(&self, id: u64) -> String {
    self.prog.nams.get(&id).unwrap_or(&"?".to_string()).clone()
  }

  /// Returns the arity of a given id
  pub fn get_arity(&self, id: u64) -> u64 {
    *self.prog.aris.get(&id).unwrap_or(&u64::MAX)
  }

  /// Returns the name of a given id
  pub fn get_id(&self, name: &str) -> u64 {
    *self.book.name_to_id.get(name).unwrap_or(&u64::MAX)
  }

  //// WASM re-exports

  pub fn DP0() -> u64 {
    Tag::DP0.as_u64()
  }

  pub fn DP1() -> u64 {
    Tag::DP1.as_u64()
  }

  pub fn VAR() -> u64 {
    Tag::VAR.as_u64()
  }

  pub fn ARG() -> u64 {
    Tag::ARG.as_u64()
  }

  pub fn ERA() -> u64 {
    Tag::ERA.as_u64()
  }

  pub fn LAM() -> u64 {
    Tag::LAM.as_u64()
  }

  pub fn APP() -> u64 {
    Tag::APP.as_u64()
  }

  pub fn SUP() -> u64 {
    Tag::SUP.as_u64()
  }

  pub fn CTR() -> u64 {
    Tag::CTR.as_u64()
  }

  pub fn FUN() -> u64 {
    Tag::FUN.as_u64()
  }

  pub fn OP2() -> u64 {
    Tag::OP2.as_u64()
  }

  pub fn U60() -> u64 {
    Tag::U60.as_u64()
  }

  pub fn F60() -> u64 {
    Tag::F60.as_u64()
  }

  pub fn ADD() -> u64 {
    Oper::Add.as_u64()
  }

  pub fn SUB() -> u64 {
    Oper::Sub.as_u64()
  }

  pub fn MUL() -> u64 {
    Oper::Mul.as_u64()
  }

  pub fn DIV() -> u64 {
    Oper::Div.as_u64()
  }

  pub fn MOD() -> u64 {
    Oper::Mod.as_u64()
  }

  pub fn AND() -> u64 {
    Oper::And.as_u64()
  }

  pub fn OR() -> u64 {
    Oper::Or.as_u64()
  }

  pub fn XOR() -> u64 {
    Oper::Xor.as_u64()
  }

  pub fn SHL() -> u64 {
    Oper::Shl.as_u64()
  }

  pub fn SHR() -> u64 {
    Oper::Shr.as_u64()
  }

  pub fn LTN() -> u64 {
    Oper::Ltn.as_u64()
  }

  pub fn LTE() -> u64 {
    Oper::Lte.as_u64()
  }

  pub fn EQL() -> u64 {
    Oper::Eql.as_u64()
  }

  pub fn GTE() -> u64 {
    Oper::Gte.as_u64()
  }

  pub fn GTN() -> u64 {
    Oper::Gtn.as_u64()
  }

  pub fn NEQ() -> u64 {
    Oper::Neq.as_u64()
  }

  pub fn CELLS_PER_KB() -> usize {
    CELLS_PER_KB
  }

  pub fn CELLS_PER_MB() -> usize {
    CELLS_PER_MB
  }

  pub fn CELLS_PER_GB() -> usize {
    CELLS_PER_GB
  }

  pub fn get_tag(lnk: Ptr) -> Tag {
    get_tag(lnk)
  }

  pub fn get_ext(lnk: Ptr) -> u64 {
    get_ext(lnk)
  }

  pub fn get_val(lnk: Ptr) -> u64 {
    get_val(lnk)
  }

  pub fn get_num(lnk: Ptr) -> u64 {
    get_num(lnk)
  }

  pub fn get_loc(lnk: Ptr, arg: u64) -> u64 {
    get_loc(lnk, arg)
  }

  pub fn Var(pos: u64) -> Ptr {
    Var(pos)
  }

  pub fn Dp0(col: u64, pos: u64) -> Ptr {
    Dp0(col, pos)
  }

  pub fn Dp1(col: u64, pos: u64) -> Ptr {
    Dp1(col, pos)
  }

  pub fn Arg(pos: u64) -> Ptr {
    Arg(pos)
  }

  pub fn Era() -> Ptr {
    Era()
  }

  pub fn Lam(pos: u64) -> Ptr {
    Lam(pos)
  }

  pub fn App(pos: u64) -> Ptr {
    App(pos)
  }

  pub fn Sup(col: u64, pos: u64) -> Ptr {
    Sup(col, pos)
  }

  pub fn Op2(ope: u64, pos: u64) -> Ptr {
    Op2(ope, pos)
  }

  pub fn U6O(val: u64) -> Ptr {
    U6O(val)
  }

  pub fn F6O(val: u64) -> Ptr {
    F6O(val)
  }

  pub fn Ctr(fun: u64, pos: u64) -> Ptr {
    Ctr(fun, pos)
  }

  pub fn Fun(fun: u64, pos: u64) -> Ptr {
    Fun(fun, pos)
  }

  pub fn link(&mut self, loc: u64, lnk: Ptr) -> Ptr {
    self.heap.link(loc, lnk)
  }

  pub fn alloc(&mut self, size: u64) -> u64 {
    self.heap.alloc(0, size) // FIXME tid?
  }

  pub fn free(&mut self, loc: u64, size: u64) {
    self.heap.free(0, loc, size) // FIXME tid?
  }

  pub fn collect(&mut self, term: Ptr) {
    self.heap.collect(&self.prog.aris, 0, term) // FIXME tid?
  }
}

// Methods that aren't compiled to JS
impl Runtime {
  /// Allocates a new term, returns its location
  pub fn alloc_term(&mut self, term: &language::syntax::Term) -> u64 {
    alloc_term(&self.heap, &self.prog, 0, &self.book, term) // FIXME tid?
  }

  /// Given a location, recovers the Core stored on it
  pub fn readback(&self, host: u64) -> Box<language::syntax::Term> {
    language::readback::as_term(&self.heap, &self.prog, host)
  }

  /// Given a location, recovers the Term stored on it
  pub fn linear_readback(&self, host: u64) -> Box<language::syntax::Term> {
    self.heap.as_linear_term(&self.prog, host)
  }
}
