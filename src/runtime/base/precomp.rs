use crate::runtime::{*};
use std::sync::atomic::{AtomicBool, Ordering};

// Precomps
// --------

pub struct PrecompFns {
  pub visit: VisitFun,
  pub apply: ApplyFun,
}

pub struct Precomp {
  pub id    : u64,
  pub name  : &'static str,
  pub arity : usize,
  pub funcs : Option<PrecompFns>,
}

pub const STRING_NIL : u64 = 0;
pub const STRING_CONS : u64 = 1;
pub const KIND_TERM_CT0 : u64 = 2;
pub const KIND_TERM_CT1 : u64 = 3;
pub const KIND_TERM_CT2 : u64 = 4;
pub const KIND_TERM_CT3 : u64 = 5;
pub const KIND_TERM_CT4 : u64 = 6;
pub const KIND_TERM_CT5 : u64 = 7;
pub const KIND_TERM_CT6 : u64 = 8;
pub const KIND_TERM_CT7 : u64 = 9;
pub const KIND_TERM_CT8 : u64 = 10;
pub const KIND_TERM_CT9 : u64 = 11;
pub const KIND_TERM_CTA : u64 = 12;
pub const KIND_TERM_CTB : u64 = 13;
pub const KIND_TERM_CTC : u64 = 14;
pub const KIND_TERM_CTD : u64 = 15;
pub const KIND_TERM_CTE : u64 = 16;
pub const KIND_TERM_CTF : u64 = 17;
pub const KIND_TERM_CTG : u64 = 18;
pub const KIND_TERM_U60 : u64 = 19;
pub const KIND_TERM_F60 : u64 = 20;
pub const HVM_LOG : u64 = 21;
pub const HVM_QUERY : u64 = 22;
pub const HVM_PRINT : u64 = 23;
pub const HVM_SLEEP : u64 = 24;
pub const HVM_STORE : u64 = 25;
pub const HVM_LOAD : u64 = 26;
//[[CODEGEN:PRECOMP-IDS]]//

pub const PRECOMP : &[Precomp] = &[
  Precomp {
    id: STRING_NIL,
    name: "String.nil",
    arity: 0,
    funcs: None,
  },
  Precomp {
    id: STRING_CONS,
    name: "String.cons",
    arity: 2,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT0,
    name: "Kind.Term.ct0",
    arity: 2,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT1,
    name: "Kind.Term.ct1",
    arity: 3,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT2,
    name: "Kind.Term.ct2",
    arity: 4,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT3,
    name: "Kind.Term.ct3",
    arity: 5,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT4,
    name: "Kind.Term.ct4",
    arity: 6,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT5,
    name: "Kind.Term.ct5",
    arity: 7,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT6,
    name: "Kind.Term.ct6",
    arity: 8,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT7,
    name: "Kind.Term.ct7",
    arity: 9,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT8,
    name: "Kind.Term.ct8",
    arity: 10,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CT9,
    name: "Kind.Term.ct9",
    arity: 11,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTA,
    name: "Kind.Term.ctA",
    arity: 12,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTB,
    name: "Kind.Term.ctB",
    arity: 13,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTC,
    name: "Kind.Term.ctC",
    arity: 14,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTD,
    name: "Kind.Term.ctD",
    arity: 15,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTE,
    name: "Kind.Term.ctE",
    arity: 16,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTF,
    name: "Kind.Term.ctF",
    arity: 17,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_CTG,
    name: "Kind.Term.ctG",
    arity: 18,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_U60,
    name: "Kind.Term.u60",
    arity: 2,
    funcs: None,
  },
  Precomp {
    id: KIND_TERM_F60,
    name: "Kind.Term.f60",
    arity: 2,
    funcs: None,
  },
  Precomp {
    id: HVM_LOG,
    name: "HVM.log",
    arity: 2,
    funcs: Some(PrecompFns {
      visit: hvm_log_visit,
      apply: hvm_log_apply,
    }),
  },
  Precomp {
    id: HVM_QUERY,
    name: "HVM.query",
    arity: 1,
    funcs: Some(PrecompFns {
      visit: hvm_query_visit,
      apply: hvm_query_apply,
    }),
  },
  Precomp {
    id: HVM_PRINT,
    name: "HVM.print",
    arity: 2,
    funcs: Some(PrecompFns {
      visit: hvm_print_visit,
      apply: hvm_print_apply,
    }),
  },
  Precomp {
    id: HVM_SLEEP,
    name: "HVM.sleep",
    arity: 2,
    funcs: Some(PrecompFns {
      visit: hvm_sleep_visit,
      apply: hvm_sleep_apply,
    }),
  },
  Precomp {
    id: HVM_STORE,
    name: "HVM.store",
    arity: 3,
    funcs: Some(PrecompFns {
      visit: hvm_store_visit,
      apply: hvm_store_apply,
    }),
  },
  Precomp {
    id: HVM_LOAD,
    name: "HVM.load",
    arity: 2,
    funcs: Some(PrecompFns {
      visit: hvm_load_visit,
      apply: hvm_load_apply,
    }),
  },
//[[CODEGEN:PRECOMP-ELS]]//
];

pub const PRECOMP_COUNT : u64 = PRECOMP.len() as u64;

// HVM.log (term: Term)
// --------------------

fn hvm_log_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_log_apply(ctx: ReduceCtx) -> bool {
  normalize(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0), false);
  let code = crate::language::readback::as_code(ctx.heap, ctx.prog, get_loc(ctx.term, 0));
  println!("{}", code);
  link(ctx.heap, *ctx.host, load_arg(ctx.heap, ctx.term, 1));
  collect(ctx.heap, &ctx.prog.arit, ctx.tid, load_ptr(ctx.heap, get_loc(ctx.term, 0)));
  free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
  return true;
}

// HVM.query (cont: String -> Term)
// --------------------------------

fn hvm_query_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_query_apply(ctx: ReduceCtx) -> bool {
  fn read_input() -> String {
    use std::io::{stdin,stdout,Write};
    let mut input = String::new();
    stdin().read_line(&mut input).expect("string");
    if let Some('\n') = input.chars().next_back() { input.pop(); }
    if let Some('\r') = input.chars().next_back() { input.pop(); }
    return input;
  }
  let cont = load_arg(ctx.heap, ctx.term, 0);
  let text = make_string(ctx.heap, ctx.tid, &read_input());
  let app0 = alloc(ctx.heap, ctx.tid, 2);
  link(ctx.heap, app0 + 0, cont);
  link(ctx.heap, app0 + 1, text);
  free(ctx.heap, 0, get_loc(ctx.term, 0), 1);
  let done = App(app0);
  link(ctx.heap, *ctx.host, done);
  return true;
}

// HVM.print (text: String) (cont: Term)
// -----------------------------------------------

fn hvm_print_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_print_apply(ctx: ReduceCtx) -> bool {
  //normalize(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0), false);
  if let Some(text) = crate::language::readback::as_string(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0)) {
    println!("{}", text);
  }
  link(ctx.heap, *ctx.host, load_arg(ctx.heap, ctx.term, 1));
  collect(ctx.heap, &ctx.prog.arit, ctx.tid, load_ptr(ctx.heap, get_loc(ctx.term, 0)));
  free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
  return true;
}

// HVM.sleep (time: U60) (cont: Term)
// ----------------------------------

fn hvm_sleep_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_sleep_apply(ctx: ReduceCtx) -> bool {
  let time = reduce(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0), false);
  std::thread::sleep(std::time::Duration::from_nanos(get_num(time)));
  link(ctx.heap, *ctx.host, load_ptr(ctx.heap, get_loc(ctx.term, 1)));
  free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
  return true;
}

// HVM.store (key: String) (val: String) (cont: Term)
// --------------------------------------------------

fn hvm_store_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_store_apply(ctx: ReduceCtx) -> bool {
  if let Some(key) = crate::language::readback::as_string(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0)) {
    if let Some(val) = crate::language::readback::as_string(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 1)) {
      if std::fs::write(key, val).is_ok() {
        //let app0 = alloc(ctx.heap, ctx.tid, 2);
        //link(ctx.heap, app0 + 0, cont);
        //link(ctx.heap, app0 + 1, U6O(0));
        //free(ctx.heap, 0, get_loc(ctx.term, 0), 2);
        let done = load_arg(ctx.heap, ctx.term, 2);
        link(ctx.heap, *ctx.host, done);
        collect(ctx.heap, &ctx.prog.arit, ctx.tid, load_arg(ctx.heap, ctx.term, 0));
        collect(ctx.heap, &ctx.prog.arit, ctx.tid, load_arg(ctx.heap, ctx.term, 1));
        free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
        return true;
      }
    }
  }
  println!("Runtime failure on: {}", show_term(ctx.heap, ctx.prog, ctx.term, 0));
  std::process::exit(0);
}

// HVM.load (key: String) (cont: String -> Term)
// ---------------------------------------------

fn hvm_load_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_load_apply(ctx: ReduceCtx) -> bool {
  if let Some(key) = crate::language::readback::as_string(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0)) {
    if let Ok(file) = std::fs::read(key) {
      if let Ok(file) = std::str::from_utf8(&file) {
        let cont = load_arg(ctx.heap, ctx.term, 1); 
        let text = make_string(ctx.heap, ctx.tid, file);
        let app0 = alloc(ctx.heap, ctx.tid, 2);
        link(ctx.heap, app0 + 0, cont);
        link(ctx.heap, app0 + 1, text);
        free(ctx.heap, 0, get_loc(ctx.term, 0), 2);
        let done = App(app0);
        link(ctx.heap, *ctx.host, done);
        return true;
      }
    }
  }
  println!("Runtime failure on: {}", show_term(ctx.heap, ctx.prog, ctx.term, 0));
  std::process::exit(0);
}



//[[CODEGEN:PRECOMP-FNS]]//
