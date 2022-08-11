pub use clap::{Parser, Subcommand};
use regex::RegexBuilder;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
  /// Set quantity of allocated memory
  #[clap(short = 'M', long, default_value = "4G", parse(try_from_str=parse_mem_size))]
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

fn get_unit_ratio(unit: &str) -> Option<usize> {
  match unit.to_lowercase().as_str() {
    "k" => Some(1 << 10),
    "m" => Some(1 << 20),
    "g" => Some(1 << 30),
    _ => None,
  }
}

fn parse_mem_size(raw: &str) -> Result<usize, String> {
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
