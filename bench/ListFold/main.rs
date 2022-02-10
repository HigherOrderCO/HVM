use std::env;

enum List<A> {
  Nil,
  Cons { head: A, tail: Box<List<A>> },
}

fn fold<A, B>(list: &List<A>, c: &dyn Fn(&A, B) -> B, n: B) -> B {
  match list {
    List::Nil => n,
    List::Cons { head, tail } => c(head, fold(tail, c, n)),
  }
}

fn range(n: u32, list: List<u32>) -> List<u32> {
  if n == 0 {
    list
  } else {
    let m = n - 1;
    range(m, List::Cons { head: m, tail: Box::new(list) })
  }
}

fn main() {
  let args: Vec<String> = env::args().collect();
  let n = args[1].parse::<u32>().unwrap();

  let size = n * 100000;
  let list = range(size, List::Nil);
  println!("{}", fold(&list, &|x, y| x + y, 0));
}
