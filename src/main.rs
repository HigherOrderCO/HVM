#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod compilable;
mod convert;
mod lambolt;
mod parser;
mod runtime;

fn main() {
  let (norm, cost) = eval("Main", "
    (Main) = ((λf λx (f (f x))) (λf λx (f (f x))))
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

  // Finds the main rule on the compilable file and gets its body
  let body = &comp.func_rules.get(main).expect("Main not found.")[0].rhs;

  // Converts it to a Runtime Term and stores it on the worker's memory
  let term = convert::lambolt_to_runtime(&body, &comp);
  let host = runtime::alloc_term(&mut worker, &term);

  // Normalizes it
  runtime::normal(&mut worker, host);

  // Reads it back to a Lambolt term, as a String
  let norm = convert::runtime_to_lambolt(&worker, &comp, host);

  // Returns the normal form and the gas cost
  return (norm, worker.cost);
}
