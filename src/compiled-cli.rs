#![feature(atomic_from_mut)]
#![feature(atomic_mut_ptr)]

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_parens)]
#![allow(unused_labels)]
#![allow(non_upper_case_globals)]

mod cli;
mod language;
mod runtime;
mod api;

use clap::Parser;

#[derive(Parser)]
struct Cli {
  /// Set the heap size (in 64-bit nodes).
  #[arg(short = 's', long, default_value = "auto", value_parser = cli::parse_size)]
  size: usize,

  /// Set the number of threads to use.
  #[arg(short = 't', long, default_value = "auto", value_parser = cli::parse_tids)]
  tids: usize,

  /// Shows the number of graph rewrites performed.
  #[arg(short = 'c', long, default_value = "false", value_parser = cli::parse_bool)]
  cost: bool,

  /// Toggles debug mode, showing each reduction step.
  #[arg(short = 'd', long, default_value = "false", value_parser = cli::parse_bool)]
  debug: bool,

  /// The expression to run.
  #[arg(default_value = "Main")]
  expr: String,
}

fn main() {
  if let Err(err) = run_cli() {
    eprintln!("{}", err);
    std::process::exit(1);
  };
}

fn run_cli() -> Result<(), String> {
  let Cli { size, tids, cost: show_cost, debug, expr } = Cli::parse();
  let tids = if debug { 1 } else { tids };
  let (norm, cost, time) = api::eval(&cli::load_code(&"")?, &expr, Vec::new(), size, tids, debug)?;
  println!("{}", norm);
  if show_cost {
    eprintln!();
    eprintln!("\x1b[32m[TIME: {:.2}s | COST: {} | RPS: {:.2}m]\x1b[0m", ((time as f64)/1000.0), cost - 1, (cost as f64) / (time as f64) / 1000.0);
  }
  Ok(())
}