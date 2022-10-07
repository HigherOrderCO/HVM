#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]

use crate::language as language;
use crate::rulebook as rulebook;
use crate::runtime as runtime;
use crate::builder as builder;
use crate::readback as readback;

use wasm_bindgen::prelude::*;

pub use builder::eval_code;

pub use runtime::{Ptr,
  DP0, DP1, VAR, ARG,
  ERA, LAM, APP, SUP,
  CTR, FUN, OP2, NUM,
  ADD, SUB, MUL, DIV,
  MOD, AND, OR , XOR,
  SHL, SHR, LTN, LTE,
  EQL, GTE, GTN, NEQ,
  get_tag,
  get_ext,
  get_val,
  get_num,
  get_loc,
  CELLS_PER_KB,
  CELLS_PER_MB,
  CELLS_PER_GB,
};

pub use language::{
  Term,
  Term::Var, // TODO: add `global: bool`
  Term::Dup,
  Term::Let,
  Term::Lam,
  Term::App,
  Term::Ctr,
  Term::Num,
  Term::Op2,
};

// Helps with WASM debugging
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

#[wasm_bindgen]
pub struct Runtime {
  heap: runtime::Worker,
  funs: runtime::Funs,
  book: rulebook::RuleBook,
}

pub fn make_call(func: &str, args: &[&str]) -> Result<language::Term, String> {
  // TODO: redundant with `make_main_call`
  let args = args.iter().map(|par| language::read_term(par).unwrap()).collect();
  let name = func.to_string();
  Ok(language::Term::Ctr { name, args })
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Reduced {
  norm: String,
  cost: u64,
  size: u64,
  time: u64,
}

#[wasm_bindgen]
impl Reduced {
  pub fn get_norm(&self) -> String {
    return self.norm.clone();
  }

  pub fn get_cost(&self) -> u64 {
    return self.cost;
  }

  pub fn get_size(&self) -> u64 {
    return self.size;
  }

  pub fn get_time(&self) -> u64 {
    return self.time;
  }
}

#[wasm_bindgen]
impl Runtime {

  /// Creates a new, empty runtime
  pub fn new(size: usize) -> Runtime {
    Runtime {
      heap: runtime::new_worker(size),
      funs: vec![],
      book: rulebook::new_rulebook(),
    }
  }

  /// Creates a runtime from source code, given a max number of nodes
  pub fn from_code_with_size(code: &str, size: usize) -> Result<Runtime, String> {
    let file = language::read_file(code)?;
    let book = rulebook::gen_rulebook(&file);
    let funs = builder::build_runtime_functions(&book);
    let mut heap = runtime::new_worker(size);
    heap.aris = builder::build_runtime_arities(&book);
    return Ok(Runtime { heap, funs, book });
  }

  /// Creates a runtime from a source code
  // Please, do *not* change the external API as it may break other libraries using HVM. Asking for
  // a memory amount here isn't future-stable, since HVM will allocate memory dynamically. It is
  // not responsibility of the API users to decide how much memory to alloc, thus it shouldn't be
  // an argument of the from_code method. Leave this function with a hardcoded memory amount and
  // create a separate function to explicitly allocate memory. When the HVM is updated to handle it
  // automatically, both functions will just do the same (i.e., the first argument of
  // `from_code_with_memory` will be ignored).
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_code(code: &str) -> Result<Runtime, String> {
    Runtime::from_code_with_size(code, 4 * CELLS_PER_GB)
  }

  #[cfg(target_arch = "wasm32")]
  pub fn from_code(code: &str) -> Result<Runtime, String> {
    Runtime::from_code_with_size(code, 256 * CELLS_PER_MB)
  }

  /// Extends a runtime with new definitions
  pub fn define(&mut self, _code: &str) {
    todo!()
  }

  /// Allocates a new term, returns its location
  pub fn alloc_code(&mut self, code: &str) -> Result<u64, String> {
    Ok(self.alloc_term(&*language::read_term(code)?))
  }

  /// Given a location, returns the pointer stored on it
  pub fn at(&self, host: u64) -> Ptr {
    runtime::ask_lnk(&self.heap, host)
  }

  /// Given a location, evaluates a term to head normal form
  pub fn reduce(&mut self, host: u64) {
    runtime::reduce(&mut self.heap, &self.funs, host, Some(&self.book.id_to_name), false);
  }

  /// Given a location, evaluates a term to full normal form
  pub fn normalize(&mut self, host: u64) {
    runtime::normal(&mut self.heap, &self.funs, host, Some(&self.book.id_to_name), false);
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

  /// Given a location, runs side-efefctive actions
  #[cfg(not(target_arch = "wasm32"))]
  pub fn run_io(&mut self, host: u64) {
    runtime::run_io(&mut self.heap, &self.funs, host, Some(&self.book.id_to_name), false)
  }

  /// Given a location, recovers the lambda Term stored on it, as code
  pub fn show(&self, host: u64) -> String {
    readback::as_code(&self.heap, Some(&self.book.id_to_name), host)
  }

  /// Given a location, recovers the linear Term stored on it, as code
  pub fn show_linear(&self, host: u64) -> String {
    readback::as_linear_code(&self.heap, Some(&self.book.id_to_name), host)
  }

  /// Return the total number of graph rewrites computed
  pub fn get_rewrites(&self) -> u64 {
    self.heap.cost
  }

  /// Returns the name of a given id
  pub fn get_name(&self, id: u64) -> String {
    self.book.id_to_name.get(&id).unwrap_or(&"?".to_string()).clone()
  }

  /// Returns the arity of a given id
  pub fn get_arity(&self, id: u64) -> u64 {
    *self.book.id_to_arit.get(&id).unwrap_or(&u64::MAX)
  }

  /// Returns the name of a given id
  pub fn get_id(&self, name: &str) -> u64 {
    *self.book.name_to_id.get(name).unwrap_or(&u64::MAX)
  }

  // WASM re-exports
  
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

  pub fn NUM() -> u64 {
    return NUM;
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
    return runtime::Var(pos);
  }

  pub fn Dp0(col: u64, pos: u64) -> Ptr {
    return runtime::Dp0(col, pos);
  }

  pub fn Dp1(col: u64, pos: u64) -> Ptr {
    return runtime::Dp1(col, pos);
  }

  pub fn Arg(pos: u64) -> Ptr {
    return runtime::Arg(pos);
  }

  pub fn Era() -> Ptr {
    return runtime::Era();
  }

  pub fn Lam(pos: u64) -> Ptr {
    return runtime::Lam(pos);
  }

  pub fn App(pos: u64) -> Ptr {
    return runtime::App(pos);
  }

  pub fn Par(col: u64, pos: u64) -> Ptr {
    return runtime::Par(col, pos);
  }

  pub fn Op2(ope: u64, pos: u64) -> Ptr {
    return runtime::Op2(ope, pos);
  }

  pub fn Num(val: u64) -> Ptr {
    return runtime::Num(val);
  }

  pub fn Ctr(fun: u64, pos: u64) -> Ptr {
    return runtime::Ctr(0, fun, pos);
  }

  pub fn Fun(fun: u64, pos: u64) -> Ptr {
    return runtime::Cal(0, fun, pos);
  }

  pub fn link(&mut self, loc: u64, lnk: Ptr) -> Ptr {
    return runtime::link(&mut self.heap, loc, lnk);
  }

  pub fn alloc(&mut self, size: u64) -> u64 {
    return runtime::alloc(&mut self.heap, size);
  }

  pub fn clear(&mut self, loc: u64, size: u64) {
    return runtime::clear(&mut self.heap, loc, size);
  }

  pub fn collect(&mut self, term: Ptr) {
    return runtime::collect(&mut self.heap, term);
  }

}

// Methods that aren't compiled to JS
impl Runtime {
  /// Allocates a new term, returns its location
  pub fn alloc_term(&mut self, term: &language::Term) -> u64 {
    builder::alloc_term(&mut self.heap, &self.book, term)
  }

  /// Given a location, recovers the Term stored on it
  pub fn readback(&self, host: u64) -> Box<Term> {
    readback::as_term(&self.heap, Some(&self.book.id_to_name), host)
  }

  /// Given a location, recovers the Term stored on it
  pub fn linear_readback(&self, host: u64) -> Box<Term> {
    readback::as_linear_term(&self.heap, Some(&self.book.id_to_name), host)
  }
}

pub fn example() -> Result<(), String> {
  let mut rt = crate::api::Runtime::from_code_with_size("
    (Double Zero)     = Zero
    (Double (Succ x)) = (Succ (Succ (Double x)))
  ", 10000).unwrap();
  let loc = rt.normalize_code("(Double (Succ (Succ Zero)))");
  println!("{}", rt.show(loc));
  return Ok(());
}

//#[wasm_bindgen]
//pub fn eval_code_main(code: &str) -> Result<Reduced, String> {
  //return Runtime::eval_code_main(code);
//}

//#[wasm_bindgen]
//pub fn eval_code_main_with_memory(code: &str, memory: usize) -> Result<Reduced, String> {
  //return Runtime::eval_code_main_with_memory(code, memory);
//}
