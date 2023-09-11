#![feature(atomic_from_mut)]

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_parens)]
#![allow(unused_labels)]
#![allow(non_upper_case_globals)]

use clap::{Parser, Subcommand};
use std::io::{self, prelude::*};
use hvm::{api,cli,compiler};

// mod cli;
// mod language;
// mod runtime;
// mod compiler;
// mod api;

#[derive(Subcommand)]
enum Commands {
  /// Load a file and run an expression
  #[clap(aliases = &["r"])]

  Run {
    /// A "file.hvm" to load.
    #[arg(short = 'f', long, default_value = "")]
    file: String,

    /// The expression to run.
    #[arg(default_value = "Main")]
    expr: String,
  },

  /// Compile a file to Rust
  #[command(aliases = &["c"])]
  Compile {
    /// A "file.hvm" to load.
    file: String
  },
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
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

  #[command(subcommand)]
  pub command: Option<Commands>,
}

fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
  let Cli { size, tids, cost: show_cost, debug, command } = Cli::parse();

  let run = |file: &str, expr: &str| {
    let tids = if debug { 1 } else { tids };
    let (norm, cost, time) = api::eval(&cli::load_code(file)?, expr, Vec::new(), size, tids, debug)?;
    println!("{}", norm);
    if show_cost {
      eprintln!();
      eprintln!("\x1b[32m[TIME: {:.2}s | COST: {} | RPS: {:.2}m]\x1b[0m", ((time as f64)/1000.0), cost - 1, (cost as f64) / (time as f64) / 1000.0);
    }
    Ok(())
  };

  fn prompt(prompt: &str) -> std::io::Result<()> {
    print!("{prompt}");
    io::stdout().flush()
  }

  let repl = || {
    let stdin = io::stdin();
    let readline = || {
      let output = stdin.lock().lines().next().unwrap().unwrap();
      output.trim().to_owned()
    };
    println!(
r#" __  __   __  __  __.   __
/\ \/\ \ /\ \/| |/\  `./  \
\ \  __ \\ \ \| |\ \ \`_/\ \
 \ \_\/\_\. `._/  \ \_\-\ \_\
  \/_/\/_/ `/_/    \/_/  \/_/ REPL"#);
    loop {
      prompt("> ")?;
      let line = readline();
      if line.is_empty() {
        continue;
      }
      run("", &line)?;
    }
  };

  match command {
    Some(Commands::Run { file, expr }) => {
      run(&file, &expr)
    }
    Some(Commands::Compile { file }) => {
      let code = cli::load_code(&file)?;
      let name = file.replace(".hvm", "");
      compiler::compile(&code, &name).map_err(|x| x.to_string())?;
      println!("Compiled definitions to '/{}'.", name);
      Ok(())
    }
    None => {
      repl()
    }
  }
}

fn main() {
  if let Err(err) = run_cli() {
    eprintln!("{}", err);
    std::process::exit(1);
  };
}