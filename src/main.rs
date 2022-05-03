mod builder;
mod compiler;
mod language;
mod parser;
mod readback;
mod rulebook;
mod runtime;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
  #[clap(short = 'M', long, help = "Set quantity of allocated memory", default_value = "6GB")]
  pub memory: String,

  #[clap(subcommand)]
  pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
  #[clap(about = "Run a file interpreted", name = "run", aliases = &["r"])]
  Run { file: String, params: Vec<String> },

  #[clap(about = "Compile file to C",  name = "compile", aliases = &["c"])]
  Compile {
    file: String,
    #[clap(long)]
    single_thread: bool,
  },

  #[clap(about = "Run in debug mode", name = "debug", aliases = &["d"])]
  Debug { file: String, params: Vec<String> },
}

fn main() {
  //match run_example() {
  match run_cli() {
    Ok(..) => {}
    Err(err) => {
      eprintln!("{}", err);
    }
  };
}

fn run_cli() -> Result<(), String> {
  let cli_matches = Cli::parse();

  fn hvm(file: &str) -> String {
    if file.ends_with(".hvm") {
      file.to_string()
    } else {
      format!("{}.hvm", file)
    }
  }

  match cli_matches.command {
    Command::Compile { file, single_thread } => {
      let file = &hvm(&file);
      let code = load_file_code(file)?;

      compile_code(&code, file, !single_thread)?;
      Ok(())
    }
    Command::Run { file, params } => {
      let code = load_file_code(&hvm(&file))?;

      run_code(&code, false, params, parse_gb(&cli_matches.memory))?;
      Ok(())
    }

    Command::Debug { file, params } => {
      let code = load_file_code(&hvm(&file))?;

      run_code(&code, true, params, parse_gb(&cli_matches.memory))?;
      Ok(())
    }
  }
}

fn parse_gb(raw: &str) -> usize {
  raw.to_ascii_lowercase().strip_suffix("gb").and_then(|s| s.parse().ok()).expect("Invalid format!")
}

fn make_call(params: &Vec<String>) -> Result<language::Term, String> {
  let name = "Main".to_string();
  let mut args = Vec::new();
  for param in params {
    let term = language::read_term(param)?;
    args.push(term);
  }
  Ok(language::Term::Ctr { name, args })
}

fn run_code(code: &str, debug: bool, params: Vec<String>, memory: usize) -> Result<(), String> {
  let call = make_call(&params)?;
  let (norm, cost, size, time) = builder::eval_code(&call, code, debug, memory)?;
  println!("Rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  println!("Mem.Size: {}", size);
  println!();
  println!("{}", norm);
  Ok(())
}

fn compile_code(code: &str, name: &str, parallel: bool) -> Result<(), String> {
  if !name.ends_with(".hvm") {
    return Err("Input file must end with .hvm.".to_string());
  }
  let name = format!("{}.c", &name[0..name.len() - 4]);
  compiler::compile_code_and_save(code, &name, parallel)?;
  println!("Compiled to '{}'.", name);
  Ok(())
}

fn load_file_code(file_name: &str) -> Result<String, String> {
  std::fs::read_to_string(file_name).map_err(|err| err.to_string())
}

#[allow(dead_code)]
fn run_example() -> Result<(), String> {
  // Source code
  let _code = "(Main) = (λf λx (f (f x)) λf λx (f (f x)))";

  let _code = "
    (Fn 0) = 1
    (Fn n) = (+ (Fn (- n 1)) (Fn (- n 1)))
    (Main) = (Fn 20)
  ";

  let _code = "
    // Applies a function to all elements in a list
    (Map fn (Nil))            = (Nil)
    (Map fn (Cons head tail)) = (Cons (fn head) (Map fn tail))

    // Increments all numbers on [1,2,3]
    (Main) = (Map λx(+ x 1) (Cons 1 (Cons 2 (Cons 3 (Nil)))))
  ";

  let _code = "
    (Filter fn (Cons x xs)) = (Filter_Cons (fn x) fn x xs)
      (Filter_Cons 1 fn x xs) = (Cons x (Filter fn xs))
      (Filter_Cons 0 fn x xs) = (Filter fn xs)
    (Filter fn (Nil)) = (Nil)

    (Concat (Nil) b)        = b
    (Concat (Cons ah at) b) = (Cons ah (Concat at b))

    (Quicksort (Nil)) = (Nil)
    (Quicksort (Cons h t)) =
      let min = (Filter λx(< x h) t)
      let max = (Filter λx(> x h) t)
      (Concat (Quicksort min) (Cons h (Quicksort max)))

    (Main) = (Quicksort (Cons 3 (Cons 1 (Cons 2 (Cons 4 (Nil))))))
  ";

  let code = "
    (Sort (Nil))                         = 0
    (Sort (Cons x Nil))                  = (Foo x)
    (Sort (Cons x (Cons y Nil)))         = (Foo x y)
    (Sort (Cons x (Cons y (Cons z zs)))) = (Foo x y z zs)
    (Foo (Bar 1 x) (Baz z k))            = (+ x z)
    (Foo (Bar 7 8) (Baz z k))            = 7
  ";

  // Compiles to C and saves as 'main.c'
  compiler::compile_code_and_save(code, "main.c", true)?;
  println!("Compiled to 'main.c'.");

  // Evaluates with interpreter

  println!("Reducing with interpreter.");
  let call = language::Term::Ctr { name: "Main".to_string(), args: Vec::new() };
  let (norm, cost, size, time) = builder::eval_code(&call, code, false, 6)?;
  println!("Rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  println!("Mem.Size: {}", size);
  println!();
  println!("{}", norm);
  Ok(())
}
