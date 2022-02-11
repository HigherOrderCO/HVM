use std::env;

fn fib(n: u32) -> u64 {
  if n == 0 {
    return 0;
  }
  if n == 1 {
    return 1;
  }
  return fib(n - 1) + fib(n - 2);
}

fn main() {
  let args: Vec<String> = env::args().collect();
  let n = args[1].parse::<u32>().unwrap();
  println!("{}", fib(n));
}
