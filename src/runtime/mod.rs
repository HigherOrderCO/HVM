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

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

// If unspecified, allocates `max(16 GB, 75% free_sys_mem)` memory
pub fn default_heap_size() -> usize {
  use sysinfo::SystemExt;
  let available_memory = System::new_with_specifics(RefreshKind::new().with_memory()).free_memory();
  let heap_size = (available_memory * 3 / 4) / 8;
  let heap_size = std::cmp::min(heap_size as usize, 16 * CELLS_PER_GB);
  return heap_size as usize;
}

// If unspecified, spawns 1 thread for each available core
pub fn default_heap_tids() -> usize {
  return std::thread::available_parallelism().unwrap().get();
}

pub struct Runtime {
  pub heap: Heap,
  pub prog: Program,
  pub book: language::rulebook::RuleBook,
  pub tids: Box<[usize]>,
  pub dbug: bool,
}

impl Runtime {
  /// Creates a new, empty runtime
  pub fn new(size: usize, tids: usize, dbug: bool) -> Runtime {
    Runtime {
      heap: new_heap(size, tids),
      prog: Program::new(),
      book: language::rulebook::new_rulebook(),
      tids: new_tids(tids),
      dbug: dbug,
    }
  }

  /// Creates a runtime from source code, given a max number of nodes
  pub fn from_code_with(
    code: &str,
    size: usize,
    tids: usize,
    dbug: bool,
  ) -> Result<Runtime, String> {
    let file = language::syntax::read_file(code)?;
    let heap = new_heap(size, tids);
    let prog = Program::new();
    let book = language::rulebook::gen_rulebook(&file);
    let tids = new_tids(tids);
    return Ok(Runtime { heap, prog, book, tids, dbug });
  }

  ////fn get_area(&mut self) -> runtime::Area {
  ////return runtime::get_area(&mut self.heap, 0)
  ////}

  /// Creates a runtime from a source code
  //#[cfg(not(target_arch = "wasm32"))]
  pub fn from_code(code: &str) -> Result<Runtime, String> {
    Runtime::from_code_with(code, default_heap_size(), default_heap_tids(), false)
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
    load_ptr(&self.heap, host)
  }

  /// Given a location, evaluates a term to head normal form
  pub fn reduce(&mut self, host: u64) {
    reduce(&self.heap, &self.prog, &self.tids, host, false, self.dbug);
  }

  /// Given a location, evaluates a term to full normal form
  pub fn normalize(&mut self, host: u64) {
    reduce(&self.heap, &self.prog, &self.tids, host, true, self.dbug);
  }

  /// Evaluates a code, allocs and evaluates to full normal form. Returns its location.
  pub fn normalize_code(&mut self, code: &str) -> u64 {
    let host = self.alloc_code(code).ok().unwrap();
    self.normalize(host);
    return host;
  }

  /// Evaluates a code to normal form. Returns its location.
  pub fn eval_to_loc(&mut self, code: &str) -> u64 {
    return self.normalize_code(code);
  }

  /// Evaluates a code to normal form.
  pub fn eval(&mut self, code: &str) -> String {
    let host = self.normalize_code(code);
    return self.show(host);
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
    language::readback::as_linear_code(&self.heap, &self.prog, host)
  }

  /// Return the total number of graph rewrites computed
  pub fn get_rewrites(&self) -> u64 {
    get_cost(&self.heap)
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
    return DP0;
  }

  pub fn DP1() -> u64 {
    return DP1;
  }

  pub fn VAR() -> u64 {
    return VAR;
  }

  pub fn ARG() -> u64 {
    return ARG;
  }

  pub fn ERA() -> u64 {
    return ERA;
  }

  pub fn LAM() -> u64 {
    return LAM;
  }

  pub fn APP() -> u64 {
    return APP;
  }

  pub fn SUP() -> u64 {
    return SUP;
  }

  pub fn CTR() -> u64 {
    return CTR;
  }

  pub fn FUN() -> u64 {
    return FUN;
  }

  pub fn OP2() -> u64 {
    return OP2;
  }

  pub fn U60() -> u64 {
    return U60;
  }

  pub fn F60() -> u64 {
    return F60;
  }

  pub fn ADD() -> u64 {
    return ADD;
  }

  pub fn SUB() -> u64 {
    return SUB;
  }

  pub fn MUL() -> u64 {
    return MUL;
  }

  pub fn DIV() -> u64 {
    return DIV;
  }

  pub fn MOD() -> u64 {
    return MOD;
  }

  pub fn AND() -> u64 {
    return AND;
  }

  pub fn OR() -> u64 {
    return OR;
  }

  pub fn XOR() -> u64 {
    return XOR;
  }

  pub fn SHL() -> u64 {
    return SHL;
  }

  pub fn SHR() -> u64 {
    return SHR;
  }

  pub fn LTN() -> u64 {
    return LTN;
  }

  pub fn LTE() -> u64 {
    return LTE;
  }

  pub fn EQL() -> u64 {
    return EQL;
  }

  pub fn GTE() -> u64 {
    return GTE;
  }

  pub fn GTN() -> u64 {
    return GTN;
  }

  pub fn NEQ() -> u64 {
    return NEQ;
  }

  pub fn CELLS_PER_KB() -> usize {
    return CELLS_PER_KB;
  }

  pub fn CELLS_PER_MB() -> usize {
    return CELLS_PER_MB;
  }

  pub fn CELLS_PER_GB() -> usize {
    return CELLS_PER_GB;
  }

  pub fn get_tag(lnk: Ptr) -> u64 {
    return get_tag(lnk);
  }

  pub fn get_ext(lnk: Ptr) -> u64 {
    return get_ext(lnk);
  }

  pub fn get_val(lnk: Ptr) -> u64 {
    return get_val(lnk);
  }

  pub fn get_num(lnk: Ptr) -> u64 {
    return get_num(lnk);
  }

  pub fn get_loc(lnk: Ptr, arg: u64) -> u64 {
    return get_loc(lnk, arg);
  }

  pub fn Var(pos: u64) -> Ptr {
    return Var(pos);
  }

  pub fn Dp0(col: u64, pos: u64) -> Ptr {
    return Dp0(col, pos);
  }

  pub fn Dp1(col: u64, pos: u64) -> Ptr {
    return Dp1(col, pos);
  }

  pub fn Arg(pos: u64) -> Ptr {
    return Arg(pos);
  }

  pub fn Era() -> Ptr {
    return Era();
  }

  pub fn Lam(pos: u64) -> Ptr {
    return Lam(pos);
  }

  pub fn App(pos: u64) -> Ptr {
    return App(pos);
  }

  pub fn Sup(col: u64, pos: u64) -> Ptr {
    return Sup(col, pos);
  }

  pub fn Op2(ope: u64, pos: u64) -> Ptr {
    return Op2(ope, pos);
  }

  pub fn U6O(val: u64) -> Ptr {
    return U6O(val);
  }

  pub fn F6O(val: u64) -> Ptr {
    return F6O(val);
  }

  pub fn Ctr(fun: u64, pos: u64) -> Ptr {
    return Ctr(fun, pos);
  }

  pub fn Fun(fun: u64, pos: u64) -> Ptr {
    return Fun(fun, pos);
  }

  pub fn link(&mut self, loc: u64, lnk: Ptr) -> Ptr {
    return link(&self.heap, loc, lnk);
  }

  pub fn alloc(&mut self, size: u64) -> u64 {
    return alloc(&self.heap, 0, size); // FIXME tid?
  }

  pub fn free(&mut self, loc: u64, size: u64) {
    return free(&self.heap, 0, loc, size); // FIXME tid?
  }

  pub fn collect(&mut self, term: Ptr) {
    return collect(&self.heap, &self.prog.aris, 0, term); // FIXME tid?
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
    language::readback::as_linear_term(&self.heap, &self.prog, host)
  }
}
