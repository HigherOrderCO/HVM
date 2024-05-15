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

extern "C" {
  fn hvm_c(book_buffer: *const u32, run_io: bool);
}

#[cfg(feature = "cuda")]
extern "C" {
  fn hvm_cu(book_buffer: *const u32, run_io: bool);
}

fn main() {
  let matches = Command::new("kind2")
    .about("HVM2: Higher-order Virtual Machine 2 (32-bit Version)")
    .subcommand_required(true)
    .arg_required_else_help(true)
    .subcommand(
      Command::new("run")
        .about("Interprets a file (using Rust)")
        .arg(Arg::new("file").required(true)))
    .subcommand(
      Command::new("run-c")
        .about("Interprets a file (using C)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Run with IO enabled"))
    )
    .subcommand(
      Command::new("run-cu")
        .about("Interprets a file (using CUDA)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Run with IO enabled")))
    .subcommand(
      Command::new("gen-c")
        .about("Compiles a file with IO (to standalone C)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Generate with IO enabled")))
    .subcommand(
      Command::new("gen-cu")
        .about("Compiles a file (to standalone CUDA)")
        .arg(Arg::new("file").required(true))
        .arg(Arg::new("io")
          .long("io")
          .action(ArgAction::SetTrue)
          .help("Generate with IO enabled")))
    .get_matches();

  match matches.subcommand() {
    Some(("run", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      run(&book);
    }
    Some(("run-c", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let mut data : Vec<u8> = Vec::new();
      book.to_buffer(&mut data);
      //println!("{:?}", data);
      let run_io = sub_matches.get_flag("io");
      unsafe {
        hvm_c(data.as_mut_ptr() as *mut u32, run_io);
      }
    }
    Some(("run-cu", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let mut data : Vec<u8> = Vec::new();
      book.to_buffer(&mut data);
      let run_io = sub_matches.get_flag("io");
      #[cfg(feature = "cuda")]
      unsafe {
        hvm_cu(data.as_mut_ptr() as *mut u32, run_io);
      }
      #[cfg(not(feature = "cuda"))]
      println!("CUDA not available!\n");
    }
    Some(("gen-c", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let fns = cmp::compile_book(cmp::Target::C, &book);
      let hvm_c = include_str!("hvm.c");
      let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &fns);
      let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");
      let run_io = sub_matches.get_flag("io");
      let hvm_c = if run_io {
        hvm_c.replace("#define DONT_RUN_IO", "#define RUN_IO")
      } else {
        hvm_c.replace("#define RUN_IO", "#define DONT_RUN_IO")
      };
      println!("{}", hvm_c);
    }
    Some(("gen-cu", sub_matches)) => {
      let file = sub_matches.get_one::<String>("file").expect("required");
      let code = fs::read_to_string(file).expect("Unable to read file");
      let book = ast::Book::parse(&code).unwrap_or_else(|er| panic!("{}",er)).build();
      let fns = cmp::compile_book(cmp::Target::CUDA, &book);
      let hvm_c = include_str!("hvm.cu");
      let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &fns);
      let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");
      let run_io = sub_matches.get_flag("io");
      let hvm_c = if run_io {
        hvm_c.replace("#define DONT_RUN_IO", "#define RUN_IO")
      } else {
        hvm_c.replace("#define RUN_IO", "#define DONT_RUN_IO")
      };
      println!("{}", hvm_c);
    }
    _ => unreachable!(),
  }
}

pub fn run(book: &hvm::Book) {
  // Initializes the global net
  let net = hvm::GNet::new(1 << 29, 1 << 29);

  // Initializes threads
  let mut tm = hvm::TMem::new(0, 1);

  // Creates an initial redex that calls main
  let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
  tm.rbag.push_redex(hvm::Pair::new(hvm::Port::new(hvm::REF, main_id as u32), hvm::ROOT));
  net.vars_create(hvm::ROOT.get_val() as usize, hvm::NONE);

  // Starts the timer
  let start = std::time::Instant::now();

  // Evaluates
  tm.evaluator(&net, &book);
  
  // Stops the timer
  let duration = start.elapsed();

  //println!("{}", net.show());

  // Prints the result
  if let Some(tree) = ast::Net::readback(&net, book) {
    println!("Result: {}", tree.show());
  } else {
    println!("Readback failed. Printing GNet memdump...\n");
    println!("{}", net.show());
  }

  // Prints interactions and time
  let itrs = net.itrs.load(std::sync::atomic::Ordering::Relaxed);
  println!("- ITRS: {}", itrs);
  println!("- TIME: {:.2}s", duration.as_secs_f64());
  println!("- MIPS: {:.2}", itrs as f64 / duration.as_secs_f64() / 1_000_000.0);
}
