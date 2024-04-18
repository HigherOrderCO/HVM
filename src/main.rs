#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod ast;
mod cmp;
mod hvm;

const CODE : &str = "
@fun  = (?<(@fun0 @fun1) a> a)
@fun0 = a & @lop ~ (#65536 a)
@fun1 = ({a b} c) & @fun ~ (a <d c>) & @fun ~ (b d)
@lop  = (?<(#0 @lop0) a> a)
@lop0 = (a b) & @lop ~ (a b)
@main = a & @fun ~ (#10 a)
";

fn main() {
    let ast_book = match ast::CoreParser::new(CODE).parse_book() {
      Ok(got) => got,
      Err(er) => panic!("{}", er),
    };
  
    let book = ast_book.build();
    println!("{}", cmp::compile_book(cmp::Target::C, &book));
}
