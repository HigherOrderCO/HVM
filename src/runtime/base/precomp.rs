use crate::runtime::{*};
use std::sync::atomic::{AtomicBool, Ordering};

// Precomps
// --------

pub struct PrecompFuns {
  pub visit: VisitFun,
  pub apply: ApplyFun,
}

pub struct Precomp {
  pub id: u64,
  pub name: &'static str,
  pub smap: &'static [bool],
  pub funs: Option<PrecompFuns>,
}

pub const STRING_NIL : u64 = 0;
pub const STRING_CONS : u64 = 1;
pub const BOTH : u64 = 2;
pub const KIND_TERM_CT0 : u64 = 3;
pub const KIND_TERM_CT1 : u64 = 4;
pub const KIND_TERM_CT2 : u64 = 5;
pub const KIND_TERM_CT3 : u64 = 6;
pub const KIND_TERM_CT4 : u64 = 7;
pub const KIND_TERM_CT5 : u64 = 8;
pub const KIND_TERM_CT6 : u64 = 9;
pub const KIND_TERM_CT7 : u64 = 10;
pub const KIND_TERM_CT8 : u64 = 11;
pub const KIND_TERM_CT9 : u64 = 12;
pub const KIND_TERM_CTA : u64 = 13;
pub const KIND_TERM_CTB : u64 = 14;
pub const KIND_TERM_CTC : u64 = 15;
pub const KIND_TERM_CTD : u64 = 16;
pub const KIND_TERM_CTE : u64 = 17;
pub const KIND_TERM_CTF : u64 = 18;
pub const KIND_TERM_CTG : u64 = 19;
pub const KIND_TERM_U60 : u64 = 20;
pub const KIND_TERM_F60 : u64 = 21;
pub const U60_IF : u64 = 22;
pub const U60_SWAP : u64 = 23;
pub const HVM_LOG : u64 = 24;
pub const HVM_QUERY : u64 = 25;
pub const HVM_PRINT : u64 = 26;
pub const HVM_SLEEP : u64 = 27;
pub const HVM_STORE : u64 = 28;
pub const HVM_LOAD : u64 = 29;
//[[CODEGEN:PRECOMP-IDS]]//

pub const PRECOMP : &[Precomp] = &[
  Precomp {
    id: STRING_NIL,
    name: "Data.String.nil",
    smap: &[false; 0],
    funs: None,
  },
  Precomp {
    id: STRING_CONS,
    name: "Data.String.cons",
    smap: &[false; 2],
    funs: None,
  },
  Precomp {
    id: BOTH,
    name: "Both",
    smap: &[false; 2],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT0,
    name: "Apps.Kind.Term.ct0",
    smap: &[false; 2],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT1,
    name: "Apps.Kind.Term.ct1",
    smap: &[false; 3],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT2,
    name: "Apps.Kind.Term.ct2",
    smap: &[false; 4],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT3,
    name: "Apps.Kind.Term.ct3",
    smap: &[false; 5],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT4,
    name: "Apps.Kind.Term.ct4",
    smap: &[false; 6],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT5,
    name: "Apps.Kind.Term.ct5",
    smap: &[false; 7],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT6,
    name: "Apps.Kind.Term.ct6",
    smap: &[false; 8],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT7,
    name: "Apps.Kind.Term.ct7",
    smap: &[false; 9],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT8,
    name: "Apps.Kind.Term.ct8",
    smap: &[false; 10],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CT9,
    name: "Apps.Kind.Term.ct9",
    smap: &[false; 11],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTA,
    name: "Apps.Kind.Term.ctA",
    smap: &[false; 12],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTB,
    name: "Apps.Kind.Term.ctB",
    smap: &[false; 13],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTC,
    name: "Apps.Kind.Term.ctC",
    smap: &[false; 14],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTD,
    name: "Apps.Kind.Term.ctD",
    smap: &[false; 15],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTE,
    name: "Apps.Kind.Term.ctE",
    smap: &[false; 16],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTF,
    name: "Apps.Kind.Term.ctF",
    smap: &[false; 17],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_CTG,
    name: "Apps.Kind.Term.ctG",
    smap: &[false; 18],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_U60,
    name: "Apps.Kind.Term.u60",
    smap: &[false; 2],
    funs: None,
  },
  Precomp {
    id: KIND_TERM_F60,
    name: "Apps.Kind.Term.f60",
    smap: &[false; 2],
    funs: None,
  },
  Precomp {
    id: U60_IF,
    name: "Data.U60.if",
    smap: &[true, false, false],
    funs: Some(PrecompFuns {
      visit: u60_if_visit,
      apply: u60_if_apply,
    }),
  },
  Precomp {
    id: U60_SWAP,
    name: "Data.U60.swap",
    smap: &[true, false, false],
    funs: Some(PrecompFuns {
      visit: u60_swap_visit,
      apply: u60_swap_apply,
    }),
  },
  Precomp {
    id: HVM_LOG,
    name: "Data.HVM.log",
    smap: &[false; 2],
    funs: Some(PrecompFuns {
      visit: hvm_log_visit,
      apply: hvm_log_apply,
    }),
  },
  Precomp {
    id: HVM_QUERY,
    name: "Data.HVM.query",
    smap: &[false; 1],
    funs: Some(PrecompFuns {
      visit: hvm_query_visit,
      apply: hvm_query_apply,
    }),
  },
  Precomp {
    id: HVM_PRINT,
    name: "Data.HVM.print",
    smap: &[false; 2],
    funs: Some(PrecompFuns {
      visit: hvm_print_visit,
      apply: hvm_print_apply,
    }),
  },
  Precomp {
    id: HVM_SLEEP,
    name: "Data.HVM.sleep",
    smap: &[false; 2],
    funs: Some(PrecompFuns {
      visit: hvm_sleep_visit,
      apply: hvm_sleep_apply,
    }),
  },
  Precomp {
    id: HVM_STORE,
    name: "Data.HVM.store",
    smap: &[false; 3],
    funs: Some(PrecompFuns {
      visit: hvm_store_visit,
      apply: hvm_store_apply,
    }),
  },
  Precomp {
    id: HVM_LOAD,
    name: "Data.HVM.load",
    smap: &[false; 2],
    funs: Some(PrecompFuns {
      visit: hvm_load_visit,
      apply: hvm_load_apply,
    }),
  },
//[[CODEGEN:PRECOMP-ELS]]//
];

pub const PRECOMP_COUNT : u64 = PRECOMP.len() as u64;

// Ul0.if (cond: Term) (if_t: Term) (if_f: Term)
// ---------------------------------------------

#[inline(always)]
pub fn u60_if_visit(ctx: ReduceCtx) -> bool {
  if is_whnf(load_arg(ctx.heap, ctx.term, 0)) {
    return false;
  } else {
    let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
    *ctx.cont = goup;
    *ctx.host = get_loc(ctx.term, 0);
    return true;
  }
}

#[inline(always)]
pub fn u60_if_apply(ctx: ReduceCtx) -> bool {
  let arg0 = load_arg(ctx.heap, ctx.term, 0);
  let arg1 = load_arg(ctx.heap, ctx.term, 1);
  let arg2 = load_arg(ctx.heap, ctx.term, 2);
  if get_tag(arg0) == SUP {
    fun::superpose(ctx.heap, &ctx.prog.aris, ctx.tid, *ctx.host, ctx.term, arg0, 0);
  }
  if (get_tag(arg0) == U60) {
    if (get_num(arg0) == 0) {
      inc_cost(ctx.heap, ctx.tid);
      let done = arg2;
      link(ctx.heap, *ctx.host, done);
      collect(ctx.heap, &ctx.prog.aris, ctx.tid, arg1);
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      return true;
    } else {
      inc_cost(ctx.heap, ctx.tid);
      let done = arg1;
      link(ctx.heap, *ctx.host, done);
      collect(ctx.heap, &ctx.prog.aris, ctx.tid, arg2);
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      return true;
    }
  }
  return false;
}

// U60.swap (cond: Term) (pair: Term)
// ----------------------------------

#[inline(always)]
pub fn u60_swap_visit(ctx: ReduceCtx) -> bool {
  if is_whnf(load_arg(ctx.heap, ctx.term, 0)) {
    return false;
  } else {
    let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
    *ctx.cont = goup;
    *ctx.host = get_loc(ctx.term, 0);
    return true;
  }
}

#[inline(always)]
pub fn u60_swap_apply(ctx: ReduceCtx) -> bool {
  let arg0 = load_arg(ctx.heap, ctx.term, 0);
  let arg1 = load_arg(ctx.heap, ctx.term, 1);
  let arg2 = load_arg(ctx.heap, ctx.term, 2);
  if get_tag(arg0) == SUP {
    fun::superpose(ctx.heap, &ctx.prog.aris, ctx.tid, *ctx.host, ctx.term, arg0, 0);
  }
  if (get_tag(arg0) == U60) {
    if (get_num(arg0) == 0) {
      inc_cost(ctx.heap, ctx.tid);
      let ctr_0 = alloc(ctx.heap, ctx.tid, 2);
      link(ctx.heap, ctr_0 + 0, arg1);
      link(ctx.heap, ctr_0 + 1, arg2);
      let done = Ctr(BOTH, ctr_0);
      link(ctx.heap, *ctx.host, done);
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      return true;
    } else {
      inc_cost(ctx.heap, ctx.tid);
      let ctr_0 = alloc(ctx.heap, ctx.tid, 2);
      link(ctx.heap, ctr_0 + 0, arg2);
      link(ctx.heap, ctr_0 + 1, arg1);
      let done = Ctr(BOTH, ctr_0);
      link(ctx.heap, *ctx.host, done);
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      return true;
    }
  }
  return false;
}

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
  collect(ctx.heap, &ctx.prog.aris, ctx.tid, load_ptr(ctx.heap, get_loc(ctx.term, 0)));
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
  collect(ctx.heap, &ctx.prog.aris, ctx.tid, load_ptr(ctx.heap, get_loc(ctx.term, 0)));
  free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
  return true;
}

// HVM.sleep (time: U60) (cont: Term)
// ----------------------------------

fn hvm_sleep_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_sleep_apply(ctx: ReduceCtx) -> bool {
  let time = reduce(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0), true, false);
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
        collect(ctx.heap, &ctx.prog.aris, ctx.tid, load_arg(ctx.heap, ctx.term, 0));
        collect(ctx.heap, &ctx.prog.aris, ctx.tid, load_arg(ctx.heap, ctx.term, 1));
        free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
        return true;
      }
    }
  }
  println!("Runtime failure on: {}", show_at(ctx.heap, ctx.prog, *ctx.host, &[]));
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
  println!("Runtime failure on: {}", show_at(ctx.heap, ctx.prog, *ctx.host, &[]));
  std::process::exit(0);
}

//[[CODEGEN:PRECOMP-FNS]]//
