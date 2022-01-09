#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod common;
mod lambolt;
mod parser;
mod readback;
mod runtime;

fn main() {
  // Testing the error highlighter
  println!(
    "{}",
    &parser::highlight(
      3,
      7,
      "oi tudo bem? como vai você hoje?\neu pessoalmente estou ok.\nespero que vc tbm"
    )
  );

  // Testing the parser
  let tt: parser::Testree = *parser::read(parser::testree_parser(), "(oi ((tudo bem) (com voce)))");
  println!("{}", parser::testree_show(&tt));
}
