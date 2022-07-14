#![allow(unused_variables)]
#![allow(dead_code)]

use crate::language as language;
use crate::rulebook as rulebook;
use crate::runtime as runtime;
use crate::builder as builder;
use crate::readback as readback;

pub use builder::eval_code;

pub use runtime::{Ptr,
  DP0, DP1, VAR, ARG,
  ERA, LAM, APP, PAR,
  CTR, CAL, OP2, NUM,
  ADD, SUB, MUL, DIV,
  MOD, AND, OR , XOR,
  SHL, SHR, LTN, LTE,
  EQL, GTE, GTN, NEQ,
  get_tag,
  get_ext,
  get_val,
  get_num,
  get_loc,
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

pub struct Runtime {
  heap: runtime::Worker,
  book: rulebook::RuleBook,
}

pub fn make_call(func: &str, args: &[&str]) -> Result<language::Term, String> {
  let args = args.iter().map(|par| language::read_term(par).unwrap()).collect();
  let name = func.to_string();
  Ok(language::Term::Ctr { name, args })
}

#[cfg(test)]
mod tests {
  use crate::eval_code;
  use crate::make_call;

  #[test]
  fn test() {
    let code = "
    (Fn 0) = 0
    (Fn 1) = 1
    (Fn n) = (+ (Fn (- n 1)) (Fn (- n 2)))
    (Main) = (Fn 20)
    ";

    let (norm, _cost, _size, _time) =
      eval_code(&make_call("Main", &[]).unwrap(), code, false).unwrap();
    assert_eq!(norm, "6765");
  }
}

impl Runtime {

  /// Creates a new, empty runtime
  pub fn new() -> Self {
    Runtime {
      heap: runtime::new_worker(),
      book: rulebook::new_rulebook(),
    }
  }

  /// Creates a runtime from a source code
  pub fn from_code(code: &str) -> Result<Self, String> {
    let file = language::read_file(code)?;
    let book = rulebook::gen_rulebook(&file);
    let mut heap = runtime::new_worker();
    heap.funs = builder::build_runtime_functions(&book);
    heap.aris = builder::build_runtime_arities(&book);
    return Ok(Runtime { heap, book });
  }

  /// Extends a runtime with new definitions
  pub fn extend(&mut self, _code: &str) {
    todo!()
  }

  /// Allocates a new term, returns its location
  pub fn alloc_term(&mut self, term: &language::Term) -> u64 {
    builder::alloc_term(&mut self.heap, &self.book, term)
  }

  /// Allocates a new term, returns its location
  pub fn alloc_code(&mut self, code: &str) -> Result<u64, String> {
    Ok(self.alloc_term(&*language::read_term(code)?))
  }

  /// Evaluates a code, returns the result location
  pub fn normalize_code(&mut self, code: &str) -> Result<Box<Term>, String> {
    let host = self.alloc_code(code)?;
    self.normalize(host);
    return Ok(self.readback(host));
  }

  /// Given a location, returns the pointer stored on it
  pub fn ptr(&self, host: u64) -> Ptr {
    runtime::ask_lnk(&self.heap, host)
  }

  /// Given a location, evaluates a term to head normal form
  pub fn reduce(&mut self, host: u64) -> Ptr {
    runtime::reduce(&mut self.heap, host, Some(&self.book.id_to_name), false)
  }

  /// Given a location, evaluates a term to full normal form
  pub fn normalize(&mut self, host: u64) -> Ptr {
    runtime::normal(&mut self.heap, host, Some(&self.book.id_to_name), false)
  }

  /// Given a location, recovers the Term stored on it
  pub fn readback(&self, host: u64) -> Box<Term> {
    readback::as_term(&self.heap, Some(&self.book), host).unwrap()
  }

  /// Given a location, recovers the Term stored on it, as code
  pub fn show(&self, host: u64) -> String {
    readback::as_code(&self.heap, Some(&self.book), host)
  }

  /// Return the total number of graph rewrites computed
  pub fn get_rewrites(&self) -> u64 {
    self.heap.cost
  }

  /// Returns the name of a given id
  pub fn get_name(&self, id: u64) -> String {
    self.book.id_to_name.get(&id).unwrap_or(&"?".to_string()).clone()
  }

  /// Returns the name of a given id
  pub fn get_id(&self, name: &str) -> u64 {
    *self.book.name_to_id.get(name).unwrap_or(&u64::MAX)
  }

}

pub fn example() -> Result<(), String> {

  let mut rt = crate::api::Runtime::from_code("
    (Double Zero)     = Zero
    (Double (Succ x)) = (Succ (Succ (Double x)))
  ").unwrap();

  println!("{}", rt.normalize_code("(Double (Succ (Succ Zero)))")?);

  return Ok(());

}
