#![feature(atomic_from_mut)]

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_parens)]
#![allow(unused_labels)]
#![allow(non_upper_case_globals)]

mod language;
mod runtime;
mod compiler;
mod api;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
struct Cli {
  #[clap(subcommand)]
  pub command: Command,
}

#[derive(Subcommand)]
enum Command {
  /// Load a file and run an expression
  #[clap(aliases = &["r"])]

  Run { 
    /// Set the heap size (in 64-bit nodes).
    #[clap(short = 's', long, default_value = "auto", parse(try_from_str=parse_size))]
    size: usize,

    /// Set the number of threads to use.
    #[clap(short = 't', long, default_value = "auto", parse(try_from_str=parse_tids))]
    tids: usize,

    /// Shows the number of graph rewrites performed.
    #[clap(short = 'c', long, default_value = "false", default_missing_value = "true", parse(try_from_str=parse_bool))]
    cost: bool,

    /// Toggles debug mode, showing each reduction step.
    #[clap(short = 'd', long, default_value = "false", default_missing_value = "true", parse(try_from_str=parse_bool))]
    debug: bool,

    /// A "file.hvm" to load.
    #[clap(short = 'f', long, default_value = "")]
    file: String,

    /// The expression to run.
    #[clap(default_value = "Main")]
    expr: String,
  },

  /// Compile a file to Rust
  #[clap(aliases = &["c"])]
  Compile {
    /// A "file.hvm" to load.
    file: String
  },
}

fn main() {
  if let Err(err) = run_cli() {
    eprintln!("{}", err);
    std::process::exit(1);
  };
}

fn run_cli() -> Result<(), String> {
  let cli = Cli::parse();

  match cli.command {
    Command::Run { size, tids, cost: show_cost, debug, file, expr } => {
      let tids = if debug { 1 } else { tids };
      let (norm, cost, time) = api::eval(&load_code(&file)?, &expr, Vec::new(), size, tids, debug)?;
      println!("{}", norm);
      if show_cost {
        eprintln!();
        eprintln!("\x1b[32m[TIME: {:.2}s | COST: {} | RPS: {:.2}m]\x1b[0m", ((time as f64)/1000.0), cost - 1, (cost as f64) / (time as f64) / 1000.0);
      }
      Ok(())
    }
    Command::Compile { file } => {
      let code = load_code(&file)?;
      let name = file.replace(".hvm", "");
      compiler::compile(&code, &name).map_err(|x| x.to_string())?;
      println!("Compiled definitions to '/{}'.", name);
      Ok(())
    }
  }
}

fn parse_size(text: &str) -> Result<usize, String> {
  if text == "auto" {
    return Ok(runtime::default_heap_size());
  } else {
    return text.parse::<usize>().map_err(|x| format!("{}", x));
  }
}

fn parse_tids(text: &str) -> Result<usize, String> {
  if text == "auto" {
    return Ok(runtime::default_heap_tids());
  } else {
    return text.parse::<usize>().map_err(|x| format!("{}", x));
  }
}

fn parse_bool(text: &str) -> Result<bool, String> {
  return text.parse::<bool>().map_err(|x| format!("{}", x));
}

fn load_code(file: &str) -> Result<String, String> {
  if file.is_empty() {
    return Ok(String::new());
  } else {
    return std::fs::read_to_string(file).map_err(|err| err.to_string());
  }
}
