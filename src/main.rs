#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod ast;
mod hvm;
mod cmp;

const CODE : &str = "
@F    = (?<(@FC0 @FC1) a> a)
@FC0  = a & @L ~ (#65536 a)
@FC1  = ({a b} c) & @F ~ (a <d c>) & @F ~ (b d)
@L    = (?<(#0 @LC0) a> a)
@LC0  = (a b) & @L ~ (a b)
@main = a & @F ~ (#10 a)
";

fn main() {
  println!("Hello, world!");

  match ast::CoreParser::new(CODE).parse_book() {
    Ok(book) => {
      println!("BOOK:\n{}", book.show());
    }
    Err(er) => println!("{}", er),
  }

  println!("----------");

  println!("COMP:\n{}", cmp::compile_book(&hvm::Book::new_demo(10,65536)));

  hvm::test_demo();

}
