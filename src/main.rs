#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_mut)]

mod compiled;
mod dynamic;
mod lambolt;
mod parser;
mod readback;
mod rulebook;
mod runtime;

use std::io::Write;

fn main() -> std::io::Result<()> {
  // Source code
  let code = "
    //(Main) = (λf λx (f (f x)) λf λx (f (f x)))
    
    (Slow (Z))      = 1
    (Slow (S pred)) = (+ (Slow pred) (Slow pred))

    (Main) = (Tuple
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
      (Slow (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )
    )
  ";

  // Compiles to C and saves as 'main.c'
  let as_clang = compiled::compile_code(code);
  let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open("./main.c")?;
  file.write_all(&as_clang.as_bytes())?;
  println!("Compiled to 'main.c'.");

  // Evaluates with interpreter
  println!("Reducing with interpreter.");
  let (norm, cost, time) = dynamic::eval_code("Main", code);
  println!("Rewrites: {} ({:.2} rw/s)", cost, (cost as f64) / (time as f64));
  println!("{}", norm);

  return Ok(());
}
