#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

//mod common;
mod lambolt;
mod parser;
mod readback;
mod runtime;

//use std::collections::{HashMap, HashSet};

fn main() {
  let mut worker = runtime::new_worker();

  let term_in = runtime::Term::Ctr{func: 42, args: vec![]};
  runtime::make_term(&mut worker, &term_in);

  // TODO: not tested yet / stack overflows, I'll continue tomorrow.
  let term_out = readback::runtime_to_lambolt(
    &worker,
    Some(runtime::ask_lnk(&worker, 0)),
    &std::collections::HashMap::new()
  );

  //println!("Recovered term: {}", &term);

  //runtime::build_term(&mut worker, &term);

  //let file : lambolt::File = parser::read(Box::new(|x| lambolt::parse_file(x)), "
    //// Doubles a natural number
    //(Double (Zero)) = (Zero)
    //(Double (Succ a)) = (Succ (Succ (Double a)))
    //// Main function
    //(Main) = (Double (Succ (Succ (Zero))))
  //");

  //for rule in file.rules {
    //println!("{} = {}", rule.lhs, rule.rhs);
  //}

  // Testing the error highlighter
  //println!(
  //"{}",
  //&parser::highlight(
  //3,
  //7,
  //"oi tudo bem? como vai você hoje?\neu pessoalmente estou ok.\nespero que vc tbm"
  //)
  //);

  // Testing the parser
  //let tt: parser::Testree = *parser::read(parser::testree_parser(), "(oi ((tudo bem) (com voce)))");
  //println!("{}", parser::testree_show(&tt));
}
