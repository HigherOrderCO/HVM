#![feature(atomic_from_mut)]
#![feature(atomic_mut_ptr)]

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]

// FIXME: what is the right way to export the definitions on api.rs as a lib?

pub mod compiler;
pub mod language;
pub mod parser;
pub mod readback;
pub mod rulebook;
pub mod runtime;
pub mod api;

pub use api::*;

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

    let (norm, _cost, _size, _time) = rulebook::eval_code(&make_call("Main", &[]).unwrap(), code, false, 32 << 20).unwrap();
    assert_eq!(norm, "6765");
  }
}
