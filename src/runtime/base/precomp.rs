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
pub const KIND_TERM_NUM : u64 = 19;
pub const HVM_LOG : u64 = 20;
pub const HVM_PUT : u64 = 21;
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
    id: KIND_TERM_NUM,
    name: "Kind.Term.num",
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
    id: HVM_PUT,
    name: "HVM.put",
    arity: 2,
    funcs: Some(PrecompFns {
      visit: hvm_put_visit,
      apply: hvm_put_apply,
    }),
  },
//[[CODEGEN:PRECOMP-ELS]]//
];

pub const PRECOMP_COUNT : u64 = PRECOMP.len() as u64;

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

fn hvm_put_visit(ctx: ReduceCtx) -> bool {
  return false;
}

fn hvm_put_apply(ctx: ReduceCtx) -> bool {
  normalize(ctx.heap, ctx.prog, &[ctx.tid], get_loc(ctx.term, 0), false);
  let code = crate::language::readback::as_code(ctx.heap, ctx.prog, get_loc(ctx.term, 0));
  if code.chars().nth(0) == Some('"') {
    println!("{}", &code[1 .. code.len() - 1]);
  } else {
    println!("{}", code);
  }
  link(ctx.heap, *ctx.host, load_arg(ctx.heap, ctx.term, 1));
  collect(ctx.heap, &ctx.prog.arit, ctx.tid, load_ptr(ctx.heap, get_loc(ctx.term, 0)));
  free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
  return true;
}

//[[CODEGEN:PRECOMP-FNS]]//
