#![feature(atomic_from_mut)]
#![feature(atomic_mut_ptr)]

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_parens)]
#![allow(unused_labels)]

mod language;
mod runtime;
mod compiler;

pub use clap::{Parser, Subcommand};
use regex::RegexBuilder;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
  /// Set quantity of allocated memory
  #[clap(short = 'M', long, default_value = "", parse(try_from_str=parse_mem_size))]
  pub memory_size: usize,

  #[clap(subcommand)]
  pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
  /// Run a file interpreted
  #[clap(aliases = &["r"])]
  Run { file: String, params: Vec<String> },

  /// Run in debug mode
  #[clap(aliases = &["d"])]
  Debug { file: String, params: Vec<String> },

  /// Compile a file to C
  #[clap(aliases = &["c"])]
  Compile {
    file: String,
    #[clap(long)]
    /// Disable multi-threading
    single_thread: bool,
  },
}

fn main() {
  if let Err(err) = run_cli() {
    eprintln!("{}", err);
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
      compile_code(&code, file)?;
      Ok(())
    }
    Command::Run { file, params } => {
      let code = load_file_code(&hvm(&file))?;

      run_code(&code, false, params, cli_matches.memory_size / std::mem::size_of::<u64>())?;
      Ok(())
    }

    Command::Debug { file, params } => {
      let code = load_file_code(&hvm(&file))?;

      run_code(&code, true, params, cli_matches.memory_size / std::mem::size_of::<u64>())?;
      Ok(())
    }
  }
}


fn get_unit_ratio(unit: &str) -> Option<usize> {
  match unit.to_lowercase().as_str() {
    "k" => Some(1 << 10),
    "m" => Some(1 << 20),
    "g" => Some(1 << 30),
    _ => None,
  }
}

fn parse_mem_size(raw: &str) -> Result<usize, String> {
  if raw.is_empty() {
    return Ok(runtime::HEAP_SIZE * 64 / 8);
  } else {
    let re = RegexBuilder::new(r"^(\d+)([KMG])i?B?$").case_insensitive(true).build().unwrap();
    if let Some(caps) = re.captures(raw) {
      let size = caps.get(1).unwrap().as_str().parse::<usize>();
      let unit = caps.get(2).unwrap().as_str();
      if let Ok(size) = size {
        if let Some(unit) = get_unit_ratio(unit) {
          return Ok(size * unit);
        }
      }
    }
    Err(format!("'{}' is not a valid memory size", raw))
  }
}

fn make_main_call(params: &Vec<String>) -> Result<language::syntax::Term, String> {
  let name = "Main".to_string();
  let mut args = Vec::new();
  for param in params {
    let term = language::syntax::read_term(param)?;
    args.push(term);
  }
  Ok(language::syntax::Term::Ctr { name, args })
}

fn run_code(code: &str, debug: bool, params: Vec<String>, memory: usize) -> Result<(), String> {
  let call = make_main_call(&params)?;
  //FIXME: remove below (parallel debug)
  //let call = language::syntax::Term::Ctr {
    //name: "Pair".to_string(),
    //args: vec![
      //Box::new(language::syntax::Term::Ctr { name: "Main0".to_string(), args: vec![] }),
      //Box::new(language::syntax::Term::Ctr { name: "Main1".to_string(), args: vec![] }),
    //]
  //};
  let (norm, cost, used, time) = runtime::eval_code(&call, code, debug, memory)?;
  println!("{}", norm);
  eprintln!();
  eprintln!("rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  eprintln!("used_mem: {}", used);
  Ok(())
}

fn compile_code(code: &str, name: &str) -> Result<(), String> {
  if !name.ends_with(".hvm") {
    return Err("Input file must end with .hvm.".to_string());
  }
  let name = format!("{}.c", &name[0..name.len() - 4]);
  match compiler::compile(code, &name) {
    Err(er) => {
      println!("{}", er);
    }
    Ok(res) => {}
  }
  //compiler::compile_code_and_save(code, &name, heap_size, parallel)?;
  println!("Compiled to '{}'.", name);
  Ok(())
}

fn load_file_code(file_name: &str) -> Result<String, String> {
  std::fs::read_to_string(file_name).map_err(|err| err.to_string())
}
