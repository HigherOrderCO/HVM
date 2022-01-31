use std::env;

#[derive(Debug)]
enum List {
  Nil,
  Cons{x: u32, xs: Box<List>}
}

#[derive(Debug)]
enum Tree {
  Empty,
  Single{val: u32},
  Concat{left: Box<Tree>, right: Box<Tree>}
}

fn randoms(s: u32, l: u32) -> List {
  if l == 0 {
    List::Nil
  } else {
    List::Cons{x: s, xs: Box::new(randoms(s * 1664525 + 1013904223, l-1))}
  }
}

fn sum(tree: &Tree) -> u32 {
  match tree {
    Tree::Empty => 0,
    Tree::Single{val} => *val,
    Tree::Concat{left, right} => sum(left) + sum(right),
  }
}

const PIVOT: u32 = 2147483648;

fn qsort(p: u32, s: u32, list: &List) -> Tree {
  match *list {
    List::Nil => Tree::Empty,
    List::Cons{x, ref xs} => {
      match **xs {
        List::Nil => Tree::Single{val: x},
        List::Cons{..} => split(p, s, list, Box::new(List::Nil), Box::new(List::Nil))
      }
    }
  }
}

fn split(p: u32, s: u32, list: &List, min: Box<List>, max: Box<List>) -> Tree {
  match *list {
    List::Nil => {
      let s = s >> 1;
      // println!("{:#?}", min);
      // println!("{:#?}", max);
      let min = qsort(p-s, s, &min);
      let max = qsort(p+s, s, &max);
      Tree::Concat{left: Box::new(min), right: Box::new(max)}
    },
    List::Cons{x, ref xs} => {
      place(p, s, p < x, x, xs, min, max)
    }
  }
}

fn place(p: u32, s: u32, smaller: bool, x: u32, xs: &List, min: Box<List>, max: Box<List>) -> Tree {
  if smaller {
    split(p, s, xs, min, Box::new(List::Cons{x, xs: max}))
  } else {
    split(p, s, xs, Box::new(List::Cons{x, xs: min}), max)
  }
}

fn main() {
  let args: Vec<String> = env::args().collect();
  let n = args[1].parse::<u32>().unwrap();
  let n = 100000 * n;
  println!("{:#?}", sum(&qsort(PIVOT, PIVOT, &randoms(1, n))));
}