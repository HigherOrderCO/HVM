use std::env;

enum Tree {
  Leaf{val: u32},
  Node{left: Box<Tree>, right: Box<Tree>}
}

fn gen(n: u32) -> Tree {
  if n == 0{
    Tree::Leaf{val: 1}
  } else {
    Tree::Node{left: Box::new(gen(n-1)), right: Box::new(gen(n-1))}
  }
}

fn sum(tree: &Tree) -> u32 {
  match tree {
    Tree::Leaf{val} => *val,
    Tree::Node{left, right} => sum(left) + sum(right)
  }
}

fn main() {
  let args: Vec<String> = env::args().collect();
  let n = args[1].parse::<u32>().unwrap();

  println!("{}", sum(&gen(n)));
}
