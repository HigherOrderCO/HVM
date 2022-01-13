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

fn main() {
  //let term = lambolt::read_term("(Foo)");
  //println!("{}", term);

  let mut worker = runtime::new_worker();
  worker.node[0] = runtime::Ctr(0, 42, 0);

  let term = readback::runtime_to_lambolt(&worker, Some(runtime::ask_lnk(&worker, 0)), ());

  println!("Recovered term: {}", &term);







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
