#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_mut)]

mod builder;
mod compiler;
mod language;
mod parser;
mod readback;
mod rulebook;
mod runtime;

fn main() -> std::io::Result<()> {
  cli_main()?;
  //run_example()?;
  return Ok(());
}

fn cli_main() -> std::io::Result<()> {
  let mut args : Vec<String> = std::env::args().collect();

  if args.len() <= 1 {
    show_help();
    return Ok(());
  }

  let cmd = &args[1];

  if cmd == "run" && args.len() == 3 {
    let file = &args[2];
    return run_code(&load_file_code(&file));
  }

  if cmd == "compile" && args.len() == 3 {
    let file = &args[2];
    return compile_code(&load_file_code(&file), &file);
  }

  println!("Invalid arguments: {:?}.", args);
  return Ok(());
}

fn show_help() {
  println!("High-Order Virtual Machine ({})", env!("CARGO_PKG_VERSION"));
  println!("==========================");
  println!("");
  println!("To run a file, interpreted:");
  println!("");
  println!("  hovm run file.hovm");
  println!("");
  println!("To compile a file to C:");
  println!("");
  println!("  hovm compile file.hovm");
  println!("");
  println!("More info: https://github.com/kindelia/hovm");
  println!("");
}

fn run_code(code: &str) -> std::io::Result<()> {
  println!("Reducing.");
  let (norm, cost, time) = builder::eval_code("Main", code);
  println!("Rewrites: {} ({:.2} rw/s)", cost, (cost as f64) / (time as f64));
  println!("{}", norm);
  return Ok(());
}

fn compile_code(code: &str, name: &str) -> std::io::Result<()> {
  let name = format!("{}.out.c", name);
  compiler::compile_code_and_save(code, &name)?;
  println!("Compiled to '{}'.", name);
  return Ok(());
}

fn load_file_code(file_name: &str) -> String {
  return std::fs::read_to_string(file_name).expect(&format!("Error reading file: '{}'.", file_name));
}

fn run_example() -> std::io::Result<()> {
  // Source code
  let code = "(Main) = (λf λx (f (f x)) λf λx (f (f x)))";

  // Compiles to C and saves as 'main.c'
  compiler::compile_code_and_save(code, "main.c")?;
  println!("Compiled to 'main.c'.");

  // Evaluates with interpreter
  println!("Reducing with interpreter.");
  let (norm, cost, time) = builder::eval_code("Main", code);
  println!("Rewrites: {} ({:.2} rw/s)", cost, (cost as f64) / (time as f64));
  println!("{}", norm);

  return Ok(());
}
