#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]

// Constants
// ---------

const U64_PER_KB: u64 = 0x80;
const U64_PER_MB: u64 = 0x20000;
const U64_PER_GB: u64 = 0x8000000;

pub const MAX_ARITY: u64 = 16;
pub const MEM_SPACE: u64 = U64_PER_GB;

pub const SEEN_SIZE: usize = 4194304; // uses 32 MB, covers heaps up to 2 GB

pub const VAL: u64 = 1;
pub const EXT: u64 = 0x100000000;
pub const ARI: u64 = 0x100000000000000;
pub const TAG: u64 = 0x1000000000000000;

pub const DP0: u64 = 0x0;
pub const DP1: u64 = 0x1;
pub const VAR: u64 = 0x2;
pub const ARG: u64 = 0x3;
pub const ERA: u64 = 0x4;
pub const LAM: u64 = 0x5;
pub const APP: u64 = 0x6;
pub const PAR: u64 = 0x7;
pub const CTR: u64 = 0x8;
pub const FUN: u64 = 0x9;
pub const OP2: u64 = 0xA;
pub const U32: u64 = 0xB;
pub const F32: u64 = 0xC;
pub const OUT: u64 = 0xE;
pub const NIL: u64 = 0xF;

pub const ADD: u64 = 0x0;
pub const SUB: u64 = 0x1;
pub const MUL: u64 = 0x2;
pub const DIV: u64 = 0x3;
pub const MOD: u64 = 0x4;
pub const AND: u64 = 0x5;
pub const OR: u64 = 0x6;
pub const XOR: u64 = 0x7;
pub const SHL: u64 = 0x8;
pub const SHR: u64 = 0x9;
pub const LTN: u64 = 0xA;
pub const LTE: u64 = 0xB;
pub const EQL: u64 = 0xC;
pub const GTE: u64 = 0xD;
pub const GTN: u64 = 0xE;
pub const NEQ: u64 = 0xF;

const _MAIN: u64 = 0;
const _SLOW: u64 = 1;

// Types
// -----

pub type Lnk = u64;

pub enum Term {
  Var {
    bidx: u64,
  },
  Dup {
    expr: Box<Term>,
    body: Box<Term>,
  },
  Let {
    expr: Box<Term>,
    body: Box<Term>,
  },
  Lam {
    body: Box<Term>,
  },
  App {
    func: Box<Term>,
    argm: Box<Term>,
  },
  Ctr {
    func: u64,
    args: Vec<Box<Term>>,
  },
  // Fun { func: u64, args: Vec<Box<Term>> }, TODO: should we have this?
  U32 {
    numb: u32,
  },
  Op2 {
    oper: Oper,
    val0: Box<Term>,
    val1: Box<Term>,
  },
}

pub enum Oper {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  AND,
  OR,
  XOR,
  SHL,
  SHR,
  LTN,
  LTE,
  EQL,
  GTE,
  GTN,
  NEQ,
}

pub struct Worker {
  pub node: Vec<Lnk>,
  pub size: u64,
  pub free: Vec<Vec<u64>>,
  pub cost: u64,
}

pub fn new_worker() -> Worker {
  Worker {
    node: vec![0; 6 * 0x8000000],
    size: 0,
    free: vec![vec![]; 16],
    cost: 0,
  }
}

// Globals
// -------

static mut SEEN_DATA: [u64; SEEN_SIZE] = [0; SEEN_SIZE];

// Constructors
// ------------

pub fn Var(pos: u64) -> Lnk {
  return (VAR * TAG) | pos;
}

pub fn Dp0(col: u64, pos: u64) -> Lnk {
  return (DP0 * TAG) | (col * EXT) | pos;
}

pub fn Dp1(col: u64, pos: u64) -> Lnk {
  return (DP1 * TAG) | (col * EXT) | pos;
}

pub fn Arg(pos: u64) -> Lnk {
  return (ARG * TAG) | pos;
}

pub fn Era() -> Lnk {
  return (ERA * TAG);
}

pub fn Lam(pos: u64) -> Lnk {
  return (LAM * TAG) | pos;
}

pub fn App(pos: u64) -> Lnk {
  return (APP * TAG) | pos;
}

pub fn Par(col: u64, pos: u64) -> Lnk {
  return (PAR * TAG) | (col * EXT) | pos;
}

pub fn Op2(ope: u64, pos: u64) -> Lnk {
  return (OP2 * TAG) | (ope * EXT) | pos;
}

pub fn U_32(val: u64) -> Lnk {
  return (U32 * TAG) | val;
}

pub fn Nil() -> Lnk {
  return NIL * TAG;
}

pub fn Ctr(ari: u64, fun: u64, pos: u64) -> Lnk {
  return (CTR * TAG) | (ari * ARI) | (fun * EXT) | pos;
}

pub fn Cal(ari: u64, fun: u64, pos: u64) -> Lnk {
  return (FUN * TAG) | (ari * ARI) | (fun * EXT) | pos;
}

pub fn Out(arg: u64, fld: u64) -> Lnk {
  return (OUT * TAG) | (arg << 8) | fld;
}

// Getters
// -------

pub fn get_tag(lnk: Lnk) -> u64 {
  return lnk / TAG;
}

pub fn get_ext(lnk: Lnk) -> u64 {
  return (lnk / EXT) & 0xFFFFFF;
}

pub fn get_val(lnk: Lnk) -> u64 {
  return lnk & 0xFFFFFFFF;
}

pub fn get_col(lnk: Lnk) -> u64 {
  todo!()
}

pub fn get_fun(lnk: Lnk) -> u64 {
  todo!()
}

pub fn get_ari(lnk: Lnk) -> u64 {
  return (lnk / ARI) & 0xF;
}

pub fn get_loc(lnk: Lnk, arg: u64) -> u64 {
  return get_val(lnk) + arg;
}

// Memory
// ------

pub fn ask_lnk(mem: &Worker, loc: u64) -> Lnk {
  unsafe {
    return *mem.node.get_unchecked(loc as usize);
  }
  //return mem.node[loc as usize];
}

pub fn ask_arg(mem: &Worker, term: Lnk, arg: u64) -> Lnk {
  return ask_lnk(mem, get_loc(term, arg));
}

pub fn link(mem: &mut Worker, loc: u64, lnk: Lnk) -> Lnk {
  unsafe {
    // mem.node[loc as usize] = lnk;
    *mem.node.get_unchecked_mut(loc as usize) = lnk;
    if get_tag(lnk) <= VAR {
      // let pos = get_loc(lnk, if get_tag(lnk) == DP1 { 1 } else { 0 });
      let pos = get_loc(lnk, get_tag(lnk) & 0x01);
      // mem.node[pos as usize] = Arg(loc);
      *mem.node.get_unchecked_mut(pos as usize) = Arg(loc);
    }
  }
  lnk
}

pub fn alloc(mem: &mut Worker, size: u64) -> u64 {
  if size == 0 {
    return 0;
  } else {
    if size < 16 {
      if let Some(reuse) = mem.free[size as usize].pop() {
        return reuse;
      }
    }
    let loc = mem.size;
    mem.size += size;
    return loc;
  }
}

pub fn clear(mem: &mut Worker, loc: u64, size: u64) {
  mem.free[size as usize].push(loc);
}

pub fn collect(mem: &mut Worker, term: Lnk) {
  match get_tag(term) {
    DP0 => {
      link(mem, get_loc(term, 0), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,1),0));
      return;
    }
    DP1 => {
      link(mem, get_loc(term, 1), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,0),0));
      return;
    }
    VAR => {
      link(mem, get_loc(term, 0), Era());
      return;
    }
    LAM => {
      if get_tag(ask_arg(mem, term, 0)) != ERA {
        link(mem, get_loc(ask_arg(mem, term, 0), 0), Era());
      }
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
      return;
    }
    APP => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
      return;
    }
    PAR => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
      return;
    }
    OP2 => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
      return;
    }
    U32 => {
      return;
    }
    CTR | FUN => {
      let arity = get_ari(term);
      for i in 0..arity {
        collect(mem, ask_arg(mem, term, i));
      }
      clear(mem, get_loc(term, 0), arity);
      return;
    }
    _ => {
      return;
    }
  }
}

pub fn inc_cost(mem: &mut Worker) {
  mem.cost += 1;
}

// Reduction
// ---------

pub fn subst(mem: &mut Worker, lnk: Lnk, val: Lnk) {
  if get_tag(lnk) != ERA {
    link(mem, get_loc(lnk, 0), val);
  } else {
    collect(mem, val);
  }
}

pub fn cal_par(mem: &mut Worker, host: u64, term: Lnk, argn: Lnk, n: u64) -> Lnk {
  inc_cost(mem);
  let arit = get_ari(term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(mem, arit);
  let par0 = get_loc(argn, 0);
  for i in 0..arit {
    if i != n {
      let leti = alloc(mem, 3);
      let argi = ask_arg(mem, term, i);
      link(mem, fun0 + i, Dp0(get_ext(argn), leti));
      link(mem, fun1 + i, Dp1(get_ext(argn), leti));
      link(mem, leti + 2, argi);
    } else {
      link(mem, fun0 + i, ask_arg(mem, argn, 0));
      link(mem, fun1 + i, ask_arg(mem, argn, 1));
    }
  }
  link(mem, par0 + 0, Cal(arit, func, fun0));
  link(mem, par0 + 1, Cal(arit, func, fun1));
  let done = Par(get_ext(argn), par0);
  link(mem, host, done);
  return done;
}

pub fn reduce(mem: &mut Worker, root: u64, depth: u64) -> Lnk {
  let mut stack: Vec<u64> = Vec::new();

  let mut init = 1;
  let mut host = root;

  loop {
    let term = ask_lnk(mem, host);

    if init == 1 {
      match get_tag(term) {
        APP => {
          stack.push(host);
          init = 1;
          host = get_loc(term, 0);
          continue;
        }
        DP0 | DP1 => {
          stack.push(host);
          host = get_loc(term, 2);
          continue;
        }
        OP2 => {
          stack.push(host);
          stack.push(get_loc(term, 0) | 0x80000000);
          host = get_loc(term, 1);
          continue;
        }
        FUN => {
          let fun = get_ext(term);
          let ari = get_ari(term);
          match fun {
            _SLOW => {
              stack.push(host);
              host = get_loc(term, 0);
              continue;
            }
            _MAIN => {
              init = 0;
              continue;
            }
            _ => {}
          }
        }
        _ => {}
      }
    } else {
      match get_tag(term) {
        APP => {
          let arg0 = ask_arg(mem, term, 0);
          if get_tag(arg0) == LAM {
            inc_cost(mem);
            subst(mem, ask_arg(mem, arg0, 0), ask_arg(mem, term, 1));
            let done = link(mem, host, ask_arg(mem, arg0, 1));
            clear(mem, get_loc(term, 0), 2);
            clear(mem, get_loc(arg0, 0), 2);
            init = 1;
            continue;
          }
          if get_tag(arg0) == PAR {
            inc_cost(mem);
            let app0 = get_loc(term, 0);
            let app1 = get_loc(arg0, 0);
            let let0 = alloc(mem, 3);
            let par0 = alloc(mem, 2);
            link(mem, let0 + 2, ask_arg(mem, term, 1));
            link(mem, app0 + 1, Dp0(get_ext(arg0), let0));
            link(mem, app0 + 0, ask_arg(mem, arg0, 0));
            link(mem, app1 + 0, ask_arg(mem, arg0, 1));
            link(mem, app1 + 1, Dp1(get_ext(arg0), let0));
            link(mem, par0 + 0, App(app0));
            link(mem, par0 + 1, App(app1));
            let done = Par(get_ext(arg0), par0);
            link(mem, host, done);
          }
          continue;
        }
        DP0 | DP1 => {
          let arg0 = ask_arg(mem, term, 2);
          if get_tag(arg0) == LAM {
            inc_cost(mem);
            let let0 = get_loc(term, 0);
            let par0 = get_loc(arg0, 0);
            let lam0 = alloc(mem, 2);
            let lam1 = alloc(mem, 2);
            link(mem, let0 + 2, ask_arg(mem, arg0, 1));
            link(mem, par0 + 1, Var(lam1));
            let arg0_arg_0 = ask_arg(mem, arg0, 0);
            link(mem, par0 + 0, Var(lam0));
            subst(mem, arg0_arg_0, Par(get_ext(term), par0));
            let term_arg_0 = ask_arg(mem, term, 0);
            link(mem, lam0 + 1, Dp0(get_ext(term), let0));
            subst(mem, term_arg_0, Lam(lam0));
            let term_arg_1 = ask_arg(mem, term, 1);
            link(mem, lam1 + 1, Dp1(get_ext(term), let0));
            subst(mem, term_arg_1, Lam(lam1));
            let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
            link(mem, host, done);
            init = 1;
            continue;
          }
          if get_tag(arg0) == PAR {
            if get_ext(term) == get_ext(arg0) {
              inc_cost(mem);
              subst(mem, ask_arg(mem, term, 0), ask_arg(mem, arg0, 0));
              subst(mem, ask_arg(mem, term, 1), ask_arg(mem, arg0, 1));
              let done = link(
                mem,
                host,
                ask_arg(mem, arg0, if get_tag(term) == DP0 { 0 } else { 1 }),
              );
              clear(mem, get_loc(term, 0), 3);
              clear(mem, get_loc(arg0, 0), 2);
              init = 1;
              continue;
            } else {
              inc_cost(mem);
              let par0 = alloc(mem, 2);
              let let0 = get_loc(term, 0);
              let par1 = get_loc(arg0, 0);
              let let1 = alloc(mem, 3);
              link(mem, let0 + 2, ask_arg(mem, arg0, 0));
              link(mem, let1 + 2, ask_arg(mem, arg0, 1));
              let term_arg_0 = ask_arg(mem, term, 0);
              let term_arg_1 = ask_arg(mem, term, 1);
              link(mem, par1 + 0, Dp1(get_ext(term), let0));
              link(mem, par1 + 1, Dp1(get_ext(term), let1));
              link(mem, par0 + 0, Dp0(get_ext(term), let0));
              link(mem, par0 + 1, Dp0(get_ext(term), let1));
              subst(mem, term_arg_0, Par(get_ext(arg0), par0));
              subst(mem, term_arg_1, Par(get_ext(arg0), par1));
              let done = Par(
                get_ext(arg0),
                if get_tag(term) == DP0 { par0 } else { par1 },
              );
              link(mem, host, done);
            }
          }
          if get_tag(arg0) == U32 {
            inc_cost(mem);
            subst(mem, ask_arg(mem, term, 0), arg0);
            subst(mem, ask_arg(mem, term, 1), arg0);
            let done = arg0;
            link(mem, host, arg0);
          }
          if get_tag(arg0) == CTR {
            inc_cost(mem);
            let func = get_ext(arg0);
            let arit = get_ari(arg0);
            if arit == 0 {
              subst(mem, ask_arg(mem, term, 0), Ctr(0, func, 0));
              subst(mem, ask_arg(mem, term, 1), Ctr(0, func, 0));
              clear(mem, get_loc(term, 0), 3);
              let done = link(mem, host, Ctr(0, func, 0));
            } else {
              let ctr0 = get_loc(arg0, 0);
              let ctr1 = alloc(mem, arit);
              let term_arg_0 = ask_arg(mem, term, 0);
              let term_arg_1 = ask_arg(mem, term, 1);
              for i in 0..arit {
                let leti = if i == 0 {
                  get_loc(term, 0)
                } else {
                  alloc(mem, 3)
                };
                let arg0_arg_i = ask_arg(mem, arg0, i);
                link(mem, ctr0 + i, Dp0(get_ext(term), leti));
                link(mem, ctr1 + i, Dp1(get_ext(term), leti));
                link(mem, leti + 2, arg0_arg_i);
              }
              subst(mem, term_arg_0, Ctr(arit, func, ctr0));
              subst(mem, term_arg_1, Ctr(arit, func, ctr1));
              let done = Ctr(arit, func, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
              link(mem, host, done);
            }
          }
        }
        OP2 => {
          let arg0 = ask_arg(mem, term, 0);
          let arg1 = ask_arg(mem, term, 1);
          if get_tag(arg0) == U32 && get_tag(arg1) == U32 {
            inc_cost(mem);
            let a = get_val(arg0);
            let b = get_val(arg1);
            let c;
            match get_ext(term) {
              ADD => c = (a + b) & 0xFFFFFFFF,
              SUB => c = (a - b) & 0xFFFFFFFF,
              MUL => c = (a * b) & 0xFFFFFFFF,
              DIV => c = (a / b) & 0xFFFFFFFF,
              MOD => c = (a % b) & 0xFFFFFFFF,
              AND => c = (a & b) & 0xFFFFFFFF,
              OR => c = (a | b) & 0xFFFFFFFF,
              XOR => c = (a ^ b) & 0xFFFFFFFF,
              SHL => c = (a << b) & 0xFFFFFFFF,
              SHR => c = (a >> b) & 0xFFFFFFFF,
              LTN => c = if a < b { 1 } else { 0 },
              LTE => c = if a <= b { 1 } else { 0 },
              EQL => c = if a == b { 1 } else { 0 },
              GTE => c = if a >= b { 1 } else { 0 },
              GTN => c = if a > b { 1 } else { 0 },
              NEQ => c = if a != b { 1 } else { 0 },
              _ => c = 0,
            }
            let done = U_32(c);
            clear(mem, get_loc(term, 0), 2);
            link(mem, host, done);
          }
          if get_tag(arg0) == PAR {
            inc_cost(mem);
            let op20 = get_loc(term, 0);
            let op21 = get_loc(arg0, 0);
            let let0 = alloc(mem, 3);
            let par0 = alloc(mem, 2);
            link(mem, let0 + 2, arg1);
            link(mem, op20 + 1, Dp0(get_ext(arg0), let0));
            link(mem, op20 + 0, ask_arg(mem, arg0, 0));
            link(mem, op21 + 0, ask_arg(mem, arg0, 1));
            link(mem, op21 + 1, Dp1(get_ext(arg0), let0));
            link(mem, par0 + 0, Op2(get_ext(term), op20));
            link(mem, par0 + 1, Op2(get_ext(term), op21));
            let done = Par(get_ext(arg0), par0);
            link(mem, host, done);
          }
          if get_tag(arg1) == PAR {
            inc_cost(mem);
            let op20 = get_loc(term, 0);
            let op21 = get_loc(arg1, 0);
            let let0 = alloc(mem, 3);
            let par0 = alloc(mem, 2);
            link(mem, let0 + 2, arg0);
            link(mem, op20 + 0, Dp0(get_ext(arg1), let0));
            link(mem, op20 + 1, ask_arg(mem, arg1, 0));
            link(mem, op21 + 1, ask_arg(mem, arg1, 1));
            link(mem, op21 + 0, Dp1(get_ext(arg1), let0));
            link(mem, par0 + 0, Op2(get_ext(term), op20));
            link(mem, par0 + 1, Op2(get_ext(term), op21));
            let done = Par(get_ext(arg1), par0);
            link(mem, host, done);
          }
        }
        FUN => {
          let fun = get_ext(term);
          let ari = get_ari(term);
          match fun {
            _SLOW => {
              let LOC_0: u64 = get_loc(term, 0);
              let LNK_0: u64 = ask_arg(mem, term, 0);
              if (get_tag(LNK_0) == PAR) {
                cal_par(mem, host, term, LNK_0, 0);
              }
              if (get_tag(LNK_0) == U32 && get_val(LNK_0) == 0) {
                inc_cost(mem);
                link(mem, host, U_32(1));
                clear(mem, get_loc(term, 0), 1);
                host = host;
                init = 1;
                continue;
              }
              inc_cost(mem);
              let loc_0: u64 = LOC_0;
              let lnk_1: u64 = LNK_0;
              let dup_2: u64 = alloc(mem, 3);
              let col_3: u64 = 0;
              link(mem, dup_2 + 0, Era());
              link(mem, dup_2 + 1, Era());
              link(mem, dup_2 + 2, lnk_1);
              let ret_6: u64;
              let op2_7: u64 = alloc(mem, 2);
              link(mem, op2_7 + 0, Dp0(col_3, dup_2));
              link(mem, op2_7 + 1, U_32(1));
              ret_6 = Op2(SUB, op2_7);
              let ctr_8: u64 = alloc(mem, 1);
              link(mem, ctr_8 + 0, ret_6);
              let ret_9: u64;
              let op2_10: u64 = alloc(mem, 2);
              link(mem, op2_10 + 0, Dp1(col_3, dup_2));
              link(mem, op2_10 + 1, U_32(1));
              ret_9 = Op2(SUB, op2_10);
              let ctr_11: u64 = alloc(mem, 1);
              link(mem, ctr_11 + 0, ret_9);
              let ret_4: u64;
              let op2_5: u64 = alloc(mem, 2);
              link(mem, op2_5 + 0, Cal(1, _SLOW, ctr_8));
              link(mem, op2_5 + 1, Cal(1, _SLOW, ctr_11));
              ret_4 = Op2(ADD, op2_5);
              link(mem, host, ret_4);
              clear(mem, get_loc(term, 0), 1);
              host = host;
              init = 1;
              continue;
            }
            _MAIN => {
              inc_cost(mem);
              let dup_0: u64 = alloc(mem, 3);
              let col_1: u64 = 0;
              //OP2:0:5|ARG:0:4|U32:0:2|~      |DP1:0:0|FUN:1:3|FUN:1:4|~|~|~|~|~|~|~|~|~|~|~|~|~|~|~|~|~|
              link(mem, dup_0 + 0, Era());
              link(mem, dup_0 + 1, Era());
              link(mem, dup_0 + 2, U_32(25));
              let ctr_4: u64 = alloc(mem, 1);
              link(mem, ctr_4 + 0, Dp0(col_1, dup_0));
              let ctr_5: u64 = alloc(mem, 1);
              link(mem, ctr_5 + 0, Dp1(col_1, dup_0));
              let ret_2: u64;
              let op2_3: u64 = alloc(mem, 2);
              link(mem, op2_3 + 0, Cal(1, _SLOW, ctr_4));
              link(mem, op2_3 + 1, Cal(1, _SLOW, ctr_5));
              ret_2 = Op2(ADD, op2_3);
              link(mem, host, ret_2);
              clear(mem, get_loc(term, 0), 0);
              host = host;
              init = 1;
              continue;
            }
            _ => {
              //let_fun();
              break;
            }
          }
        }
        _ => {}
      }
    }

    if let Some(item) = stack.pop() {
      init = item >> 31;
      host = item & 0x7FFFFFFF;
      continue;
    }

    break;
  }

  return ask_lnk(mem, root);
}

pub fn set_bit(bits: &mut [u64], bit: u64) {
  bits[bit as usize >> 6] |= (1 << (bit & 0x3f));
}

pub fn get_bit(bits: &[u64], bit: u64) -> bool {
  return (((bits[bit as usize >> 6] >> (bit & 0x3f)) as u8) & 1) == 1;
}

pub fn normal_go(mem: &mut Worker, host: u64, seen: &mut [u64]) -> Lnk {
  let term = ask_lnk(mem, host);
  if get_bit(seen, host) {
    return term;
  } else {
    let term = reduce(mem, host, 0);
    set_bit(seen, host);
    let mut rec_locs = Vec::with_capacity(16);
    match get_tag(term) {
      LAM => {
        rec_locs.push(get_loc(term, 1));
      }
      APP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      PAR => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      DP0 => {
        rec_locs.push(get_loc(term, 2));
      }
      DP1 => {
        rec_locs.push(get_loc(term, 2));
      }
      CTR | FUN => {
        let arity = get_ari(term);
        for i in 0..arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    for loc in rec_locs {
      let lnk: Lnk = normal_go(mem, loc, seen);
      link(mem, loc, lnk);
    }
    return term;
  }
}

pub fn normal(mem: &mut Worker, host: u64) -> Lnk {
  let mut seen = vec![0; 4194304];
  return normal_go(mem, host, &mut seen);
}

// Debug
// -----

fn show_lnk(x: Lnk) -> String {
  if (x == 0) {
    return String::from("~");
  } else {
    let tag = get_tag(x);
    let ext = get_ext(x);
    let val = get_val(x);
    let tgs = match tag {
      DP0 => "DP0",
      DP1 => "DP1",
      VAR => "VAR",
      ARG => "ARG",
      ERA => "ERA",
      LAM => "LAM",
      APP => "APP",
      PAR => "PAR",
      CTR => "CTR",
      FUN => "FUN",
      OP2 => "OP2",
      U32 => "U32",
      F32 => "F32",
      OUT => "OUT",
      NIL => "NIL",
      _ => "???",
    };
    return format!("{}:{:x}:{:x}", tgs, ext, val);
  }
}

fn show_mem(worker: &Worker) -> String {
  let mut s: String = String::new();
  for i in 0..24 {
    // pushes to the string
    s.push_str(&show_lnk(worker.node[i]));
    s.push_str("|");
  }
  return s;
}

fn test_runtime() {
  //let mut worker = new_worker();
  //worker.size = 1;
  //worker.node[0] = Cal(0, 0, 0);
  //let start = Instant::now();
  //normal(&mut worker, 0);
  //let total = (start.elapsed().as_millis() as f64) / 1000.0;
  //println!("* rwts: {} ({:.2}m rwt/s)", worker.cost, (worker.cost as f64) / total / 1000000.0);
  //println!("* time: {:?}", total);
}

// Dynamic functions
// -----------------

// Interpreted, with recursive builder would be like:
//(Foo A@(Tic a b) B@(Tac c d)) = (Bar (Tac a b) (Tic c d))
//Ctr(Bar, Ctr(Tac, Var(0), Var(1)), ...
//vars = [
//get_arg(A, 0),
//get_arg(A, 1),
//get_arg(A, 2),
//get_arg(A, 3),
//]
//tac = alloc(2)
//tic = alloc(2)
//bar = alloc(2)
//link(bar[0], tac)
//link(bar[1], tic)
//link(tac[0], vars[0]);

//pub enum Term {
//Var { name: u64 },
//Dup { expr: BTerm, body: BTerm },
//Let { expr: BTerm, body: BTerm },
//Lam { body: BTerm },
//App { func: BTerm, argm: BTerm },
//Ctr { name: String, args: Vec<BTerm> },
//U32 { numb: u32 },
//Op2 { oper: Oper, val0: BTerm, val1: BTerm },
//}

// Recursivelly builds a term.
fn make_term_go(mem: &mut Worker, term: &Term, vars: &mut Vec<u64>) -> Lnk {
  match term {
    Term::Var { bidx } => {
      if *bidx < vars.len() as u64 {
        return vars[*bidx as usize];
      } else {
        panic!("Unbound variable.");
      }
    }
    Term::Dup { expr, body } => {
      panic!("TODO");
    }
    Term::Let { expr, body } => {
      panic!("TODO");
    }
    Term::Lam { body } => {
      let node = alloc(mem, 2);
      vars.push(Var(node));
      let body = make_term_go(mem, body, vars);
      vars.pop();
      return Lam(node);
    }
    Term::App { func, argm } => {
      panic!("TODO");
    }
    Term::Ctr { func, args } => {
      let size = args.len() as u64;
      let node = alloc(mem, size);
      for (i, arg) in args.iter().enumerate() {
        let arg_lnk = make_term_go(mem, arg, vars);
        link(mem, node + i as u64, arg_lnk);
      }
      return Ctr(size, *func, node);
    }
    Term::U32 { numb } => {
      panic!("TODO");
    }
    Term::Op2 { oper, val0, val1 } => {
      panic!("TODO");
    }
  }
}

pub fn make_term(mem: &mut Worker, term: &Term) -> Lnk {
  return make_term_go(mem, term, &mut Vec::new());
}
