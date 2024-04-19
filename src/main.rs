#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use clap::{Arg, ArgAction, Command};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command as SysCommand;

mod ast;
mod cmp;
mod hvm;


fn main() {
  let matches = Command::new("kind2")
    .about("HVM2 - Higher-order Virtual Machine 2")
    .subcommand_required(true)
    .arg_required_else_help(true)
    .subcommand(Command::new("run").about("Runs a file with Rust interpreter").arg(Arg::new("file").required(true)))
    .subcommand(Command::new("gen-bin").about("Compiles a file to binary").arg(Arg::new("file").required(true)))
    .subcommand(Command::new("gen-c").about("Compiles a file to C").arg(Arg::new("file").required(true)))
    .subcommand(Command::new("gen-cu").about("Compiles a file to CUDA").arg(Arg::new("file").required(true)))
    .get_matches();

  match matches.subcommand() {
    Some(("run", sub_matches)) => {
      // Loads file/code/book
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Runs on interpreted mode
      hvm::run(&book);
    }
    Some(("gen-bin", sub_matches)) => {
      // Loads file/code/book
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Saves Book to buffer
      let data = &mut Vec::new();
      book.to_buffer(data);

      // Outputs binary
      std::io::stdout().write_all(&data).expect("Unable to write data");
    }
    Some(("gen-c", sub_matches)) => {
      // Loads file/code/book
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Generates compiled functions
      let fns = cmp::compile_book(cmp::Target::C, &book);

      // Generates compiled C file
      let hvm_c = include_str!("hvm.c");
      let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &fns);
      let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");

      println!("{}", hvm_c);
    }
    Some(("gen-cu", sub_matches)) => {
      // Loads file/code/book
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();

      // Generates compiled functions
      let fns = cmp::compile_book(cmp::Target::CUDA, &book);

      // Generates compiled C file
      let hvm_c = include_str!("hvm.cu");
      let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &fns);
      let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");

      println!("{}", hvm_c);
    }
    _ => unreachable!(),
  }
}
