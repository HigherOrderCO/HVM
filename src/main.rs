#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod lambolt;
mod parser;
mod readback;
mod runtime;

//use std::collections::{HashMap, HashSet};

fn main() {
  // TODO: not working yet, stack overflows, I'll continue tomorrow.
  let mut worker = runtime::new_worker();
  let term_in = runtime::Term::Ctr{func: 42, args: vec![]};
  
  let root = runtime::make_term(&mut worker, &term_in);
  runtime::link(&mut worker, 0, root);

  //println!("{} {} {}", worker.node[0], worker.node[1], worker.node[2]);

  let term_out = readback::runtime_to_lambolt(
    &worker,
    Some(runtime::ask_lnk(&worker, 0)),
    &std::collections::HashMap::new()
  );

  println!("Recovered term: {}", &term_out);

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

  //let file: lambolt::File = parser::read(
    //Box::new(|x| lambolt::parse_file(x)),
    //"
    //// Doubles a natural number
    //(Double (Zero))   = (Zero)
    //(Double (Succ a)) = (Succ (Succ (Double a)))
    //(Fn (Cons head tail)) = (Cons (Pair head head) Fn (tail))

    //// Main function
    //(Main) = (Double (Succ (Succ (Zero))))
  //",
  //);

  //for rule in file.rules {
    //let san_rule = sanitize(&rule);
    //match san_rule {
      //Ok(san_rule) => {
        //println!("===========");
        //println!("BEFORE {}", &rule);
        //println!("AFTER {}", san_rule.rule);
        //println!("USES {:?}", san_rule.uses);
        //println!();
      //}
      //Err(err) => {
        //println!("{}", err);
      //}
    //}
  //}
}
