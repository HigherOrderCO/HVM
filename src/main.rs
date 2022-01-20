#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod compilable;
mod compiler;
mod convert;
mod lambolt;
mod parser;
mod runtime;

fn main() {
  let (norm, cost) = eval("Main", "
    (Double (Succ x)) = (Succ (Succ (Double x)))
    (Double (Zero))   = (Zero)
    (Main)            = (Double (Succ (Succ (Zero))))
  ");

  println!("{}", norm);
  println!("- rwts: {}", cost);
}

// Evaluates a Lambolt term to normal form
fn eval(main: &str, code: &str) -> (String, u64) {
  // Creates a new Runtime worker
  let mut worker = runtime::new_worker();

  // Parses and reads the input file
  let file = lambolt::read_file(code);

  // Converts the Lambolt file to a compilable file
  let comp = compilable::gen_compilable(&file);

  // Builds dynamic functions
  let funs = convert::build_dynamic_functions(&comp);

  // Builds a runtime "(Main)" term
  let term = lambolt::Term::Ctr {
    name: String::from("Main"),
    args: Vec::new(),
  };
  let term = convert::to_runtime_term(&comp, &term, 0);

  // Allocs it on the Runtime's memory
  let host = runtime::alloc_term(&mut worker, &term);

  // Normalizes it
  runtime::normal(&mut worker, host, &funs);

  // Reads it back to a Lambolt string
  let norm = convert::readback_as_code(&worker, &comp, host);

  // Returns the normal form and the gas cost
  (norm, worker.cost)
}
