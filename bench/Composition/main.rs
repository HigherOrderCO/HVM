use std::env;

fn comp<A>(n: u32, f: &dyn Fn(A) -> A, x: A) -> A {
  if n == 0 {
    f(x)
  } else {
    comp(n-1, &|x| f(f(x)), x)
  }
} 

fn main() {
  let args: Vec<String> = env::args().collect();
  let n = args[1].parse::<u32>().unwrap();
  println!("{}", comp(n, &|x| {x}, 10));
}