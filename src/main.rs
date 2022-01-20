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

use std::time::Instant;

fn main() {
  let (norm, cost, time) = eval("Main", "
    (Slow (E))      = 1
    (Slow (O pred)) = (+ (Slow pred) (Slow pred))
    (Slow (I pred)) = (+ (Slow pred) (Slow pred))

    (Main) = (Slow 
      (O(O(O(O (O(O(O(O
      (O(O(O(O (O(O(O(O
      (O(O(O(O (O(O(O(O
      (E)
      )))) ))))
      )))) ))))
      )))) ))))
    )
  ");

  println!("{}", norm);
  println!("- rwts: {} ({:.2} rwt/s)", cost, (cost as f64) / (time as f64));
}

// Evaluates a Lambolt term to normal form
fn eval(main: &str, code: &str) -> (String, u64, u64) {
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
  let init = Instant::now();
  runtime::normal(&mut worker, host, &funs, Some(&book.id_to_name));
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a Lambolt string
  let norm = readback::as_code(&worker, &Some(book), host);

  // Returns the normal form and the gas cost
  (norm, worker.cost, time)
}
