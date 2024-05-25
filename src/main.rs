#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use ::hvm::ast::Numb;
use ::hvm::hvm::{Ctr, Port};
use ::hvm::{ast, cmp, hvm};
use clap::{Arg, ArgAction, Command};
use core::panic;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command as SysCommand;

#[cfg(feature = "c")]
extern "C" {
    fn hvm_c(book_buffer: *const u32, run_io: bool);
}

#[cfg(feature = "cuda")]
extern "C" {
    fn hvm_cu(book_buffer: *const u32, run_io: bool);
}

fn main() {
    let matches = Command::new("hvm")
        .about("HVM2: Higher-order Virtual Machine 2 (32-bit Version)")
        .version(env!("CARGO_PKG_VERSION"))
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("run")
                .about("Interprets a file (using Rust)")
                .arg(Arg::new("file").required(true)),
        )
        .subcommand(
            Command::new("run-c")
                .about("Interprets a file (using C)")
                .arg(Arg::new("file").required(true))
                .arg(
                    Arg::new("io")
                        .long("io")
                        .action(ArgAction::SetTrue)
                        .help("Run with IO enabled"),
                ),
        )
        .subcommand(
            Command::new("run-cu")
                .about("Interprets a file (using CUDA)")
                .arg(Arg::new("file").required(true))
                .arg(
                    Arg::new("io")
                        .long("io")
                        .action(ArgAction::SetTrue)
                        .help("Run with IO enabled"),
                ),
        )
        .subcommand(
            Command::new("gen-c")
                .about("Compiles a file with IO (to standalone C)")
                .arg(Arg::new("file").required(true))
                .arg(
                    Arg::new("io")
                        .long("io")
                        .action(ArgAction::SetTrue)
                        .help("Generate with IO enabled"),
                ),
        )
        .subcommand(
            Command::new("gen-cu")
                .about("Compiles a file (to standalone CUDA)")
                .arg(Arg::new("file").required(true))
                .arg(
                    Arg::new("io")
                        .long("io")
                        .action(ArgAction::SetTrue)
                        .help("Generate with IO enabled"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("run", sub_matches)) => {
            let file = sub_matches.get_one::<String>("file").expect("required");
            let code = fs::read_to_string(file).expect("Unable to read file");
            let book = ast::Book::parse(&code)
                .unwrap_or_else(|er| panic!("{}", er))
                .build();
            run(&book);
        }
        Some(("run-c", sub_matches)) => {
            let file = sub_matches.get_one::<String>("file").expect("required");
            let code = fs::read_to_string(file).expect("Unable to read file");
            let book = ast::Book::parse(&code)
                .unwrap_or_else(|er| panic!("{}", er))
                .build();
            let mut data: Vec<u8> = Vec::new();
            book.to_buffer(&mut data);
            //println!("{:?}", data);
            let run_io = sub_matches.get_flag("io");
            #[cfg(feature = "c")]
            unsafe {
                hvm_c(data.as_mut_ptr() as *mut u32, run_io);
            }
            #[cfg(not(feature = "c"))]
            println!("C runtime not available!\n");
        }
        Some(("run-cu", sub_matches)) => {
            let file = sub_matches.get_one::<String>("file").expect("required");
            let code = fs::read_to_string(file).expect("Unable to read file");
            let book = ast::Book::parse(&code)
                .unwrap_or_else(|er| panic!("{}", er))
                .build();
            let mut data: Vec<u8> = Vec::new();
            book.to_buffer(&mut data);
            let run_io = sub_matches.get_flag("io");
            #[cfg(feature = "cuda")]
            unsafe {
                hvm_cu(data.as_mut_ptr() as *mut u32, run_io);
            }
            #[cfg(not(feature = "cuda"))]
            println!("CUDA runtime not available!\n If you've installed CUDA and nvcc after HVM, please reinstall HVM.");
        }
        Some(("gen-c", sub_matches)) => {
            // Reads book from file
            let file = sub_matches.get_one::<String>("file").expect("required");
            let code = fs::read_to_string(file).expect("Unable to read file");
            let book = ast::Book::parse(&code)
                .unwrap_or_else(|er| panic!("{}", er))
                .build();

            // Gets optimal core count
            let cores = num_cpus::get();
            let tpcl2 = (cores as f64).log2().floor() as u32;

            // Generates the interpreted book
            let mut book_buf: Vec<u8> = Vec::new();
            book.to_buffer(&mut book_buf);
            let bookb = format!("{:?}", book_buf)
                .replace("[", "{")
                .replace("]", "}");
            let bookb = format!("static const u8 BOOK_BUF[] = {};", bookb);

            // Generates the C file
            let hvm_c = include_str!("hvm.c");
            let hvm_c = hvm_c.replace(
                "///COMPILED_INTERACT_CALL///",
                &cmp::compile_book(cmp::Target::C, &book),
            );
            let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");
            let hvm_c = hvm_c.replace("//COMPILED_BOOK_BUF//", &bookb);
            let hvm_c = hvm_c.replace("#define WITHOUT_MAIN", "#define WITH_MAIN");
            let hvm_c = hvm_c.replace(
                "#define TPC_L2 0",
                &format!("#define TPC_L2 {} // {} cores", tpcl2, cores),
            );
            let runio = sub_matches.get_flag("io");
            let hvm_c = if runio {
                hvm_c.replace("#define DONT_RUN_IO", "#define RUN_IO")
            } else {
                hvm_c.replace("#define RUN_IO", "#define DONT_RUN_IO")
            };
            println!("{}", hvm_c);
        }
        Some(("gen-cu", sub_matches)) => {
            // Reads book from file
            let file = sub_matches.get_one::<String>("file").expect("required");
            let code = fs::read_to_string(file).expect("Unable to read file");
            let book = ast::Book::parse(&code)
                .unwrap_or_else(|er| panic!("{}", er))
                .build();

            // Generates the interpreted book
            let mut book_buf: Vec<u8> = Vec::new();
            book.to_buffer(&mut book_buf);
            let bookb = format!("{:?}", book_buf)
                .replace("[", "{")
                .replace("]", "}");
            let bookb = format!("static const u8 BOOK_BUF[] = {};", bookb);

            //FIXME: currently, CUDA is faster on interpreted mode, so the compiler uses it.

            // Compile with compiled functions:
            //let hvm_c = include_str!("hvm.cu");
            //let hvm_c = hvm_c.replace("///COMPILED_INTERACT_CALL///", &cmp::compile_book(cmp::Target::CUDA, &book));
            //let hvm_c = hvm_c.replace("#define INTERPRETED", "#define COMPILED");

            // Compile with interpreted book:
            let hvm_c = include_str!("hvm.cu");
            let hvm_c = hvm_c.replace("//COMPILED_BOOK_BUF//", &bookb);
            let hvm_c = hvm_c.replace("#define WITHOUT_MAIN", "#define WITH_MAIN");
            let runio = sub_matches.get_flag("io");
            let hvm_c = if runio {
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
    // Starts the timer
    let start = std::time::Instant::now();

    // Initializes the global net
    let net = hvm::GNet::new(1 << 29, 1 << 29, 1 << 29);

    // Creates an initial redex that calls main
    let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
    let redex = hvm::Pair::new(hvm::Port::new(hvm::REF, main_id as u32), hvm::ROOT);
    net.rbag_create(hvm::FREE.get_val() as usize, redex);
    net.vars_create(hvm::ROOT.get_val() as usize, hvm::NONE);

    loop {
        // Normalizes the net
        net.normalize(&book);
        // Reads the λ-Encoded Ctr
        let ctr = hvm::Ctr::from(&net, &book, net.peek(hvm::ROOT));
        // Checks if IO Magic Number is a CON
        if ctr.args_buf[0].get_tag() != hvm::CON {
            break;
        }
        // Checks the IO Magic Number
        let io_magic = net.node_load(ctr.args_buf[0].get_val() as usize);
        if io_magic.get_fst().0 != hvm::Numb::new_u24(hvm::IO_MAGIC_0).0
            || io_magic.get_snd().0 != hvm::Numb::new_u24(hvm::IO_MAGIC_1).0
        {
            break;
        }

        match ctr.tag {
            hvm::IO_CALL => todo!("ffns are not implemented"),
            hvm::IO_DONE => break println!("done!"),
            _ => panic!("how did we get here?!"),
        };
    }

    // Stops the timer
    let duration = start.elapsed();

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
    println!(
        "- MIPS: {:.2}",
        itrs as f64 / duration.as_secs_f64() / 1_000_000.0
    );
}
