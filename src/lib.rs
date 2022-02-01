//! # High-order Virtual Machine (HVM) library
//! 
//! Note: this API is **unstable**.

pub mod builder;
pub mod compiler;
pub mod language;
pub mod parser;
pub mod readback;
pub mod rulebook;
pub mod runtime;

pub use builder::eval_code;

pub fn make_call(func: &str, args: &[&str]) -> language::Term {
  let args = args.iter().map(|par| language::read_term(par)).collect();
  let name = func.to_string();
  language::Term::Ctr { name, args }
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

    let (norm, _cost, _size, _time) = eval_code(&make_call("Main", &[]), code);
    let norm = norm.to_string();
    assert_eq!(norm, "6765");
  }
}
