use clap::{Parser, Subcommand};

use crate::{compile_code, load_file_code, run_code};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
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


pub(crate) fn run_cli() -> Result<(), String> {
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

      run_code(&code, false, params)?;
      Ok(())
    }

    Command::Debug { file, params } => {
      let code = load_file_code(&hvm(&file))?;

      run_code(&code, true, params)?;
      Ok(())
    }
  }
}
