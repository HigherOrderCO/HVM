#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod rulebook;
mod compiler;
mod dynfun;
mod lambolt;
mod parser;
mod readback;
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

  // Converts the Lambolt file to a rulebook file
  let book = rulebook::gen_rulebook(&file);

  // Builds dynamic functions
  let funs = dynfun::build_runtime_functions(&book);

  // Builds a runtime "(Main)" term
  let main = lambolt::read_term("(Main)");
  let host = dynfun::alloc_term(&mut worker, &book, &main);

  // Normalizes it
  runtime::normal(&mut worker, host, &funs);

  // Reads it back to a Lambolt string
  let norm = readback::as_code(&worker, &book, host);

  // Returns the normal form and the gas cost
  (norm, worker.cost)
}
