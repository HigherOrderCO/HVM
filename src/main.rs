#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod convert;
mod lambolt;
mod parser;
mod runtime;

fn main() {
  let mut worker = runtime::new_worker();

  let file = lambolt::read_file("(Main) = (Foo (Bar) (Bar))");
  let meta = lambolt::gen_meta(&file);

  let term = convert::lambolt_to_runtime(&file.rules[0].rhs, &meta);
  let host = runtime::alloc_term(&mut worker, &term);

  let term = convert::runtime_to_lambolt(&worker, &meta, host);

  println!("{}", term);
}
