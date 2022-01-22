#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_mut)]

mod rulebook;
mod compiler;
mod dynfun;
mod lambolt;
mod parser;
mod readback;
mod runtime;

fn main() {
  let (norm, cost, time) = dynfun::eval_code("Main", "
    //(Main) = (λf λx (f (f x)) λf λx (f (f x)))

    (Slow (Z))      = 1
    (Slow (S pred)) = (+ (Slow pred) (Slow pred))
    
    (Main) = (Slow (S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )) )
  ");

  println!("{}", norm);
  println!("- rwts: {} ({:.2} rwt/s)", cost, (cost as f64) / (time as f64));
}
