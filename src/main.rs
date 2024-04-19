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
@main = a & @fun ~ (#18 a)
";

//const CODE : &str = "
//@Tup8 = (a (b (c (d (e (f (g (h ((a (b (c (d (e (f (g (h i)))))))) i)))))))))
//@app  = (?<((* (a a)) @app0) b> b)
//@app0 = (a ({b (c d)} (c e))) & @app ~ (a (b (d e)))
//@rot  = ((@rot0 a) a)
//@rot0 = (a (b (c (d (e (f (g (h i)))))))) & @Tup8 ~ (b (c (d (e (f (g (h (a i))))))))
//@main = a
  //& @app ~ (#10000 (@rot (b a)))
  //& @Tup8 ~ (#1 (#2 (#3 (#4 (#5 (#6 (#7 (#8 b))))))))
//";

fn main() {
  cmp();
}

fn run() {
  let book = match ast::CoreParser::new(CODE).parse_book() {
    Ok(got) => got.build(),
    Err(er) => panic!("{}", er),
  };
  hvm::test(&book);
}

fn cmp() {
  let book = match ast::CoreParser::new(CODE).parse_book() {
    Ok(got) => got.build(),
    Err(er) => panic!("{}", er),
  };
  println!("{}", cmp::compile_book(cmp::Target::CUDA, &book));
  let mut buff = Vec::new();
  book.to_buffer(&mut buff);
  println!("{:?}", buff);
}
