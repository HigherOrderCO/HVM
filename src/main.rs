#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]

mod runtime;
mod parser;

use runtime as rt;

fn main() {
  let mut worker = rt::new_worker();

  worker.size = 1;
  worker.node[0] = rt::Cal(0, 0, 0);
  rt::normal(&mut worker, 0);

  println!("* rwt: {}", worker.cost);
}
