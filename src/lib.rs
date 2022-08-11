//! # High-order Virtual Machine (HVM) library
//!
//! Note: this API is **unstable**.

// FIXME: what is the right way to export the definitions on api.rs as a lib?

pub mod builder;
pub mod compiler;
pub mod language;
pub mod parser;
pub mod readback;
pub mod rulebook;
pub mod runtime;
pub mod api;

pub use builder::eval_code;

pub use api::*;

// FIXME: is this needed?? it's redundant with lib.rs:make_call
// pub fn make_call(func: &str, args: &[&str]) -> Result<language::Term, String> {
//   let args = args.iter().map(|par| language::read_term(par).unwrap()).collect();
//   let name = func.to_string();
//   Ok(language::Term::Ctr { name, args })
// }

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
      eval_code(&make_call("Main", &[]).unwrap(), code, false, 6).unwrap();
    assert_eq!(norm, "6765");
  }
}
