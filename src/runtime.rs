// HVM's memory model
// ------------------
// 
// The runtime memory consists of just a vector of u64 pointers. That is:
//
//   Mem ::= Vec<Ptr>
// 
// A pointer has 3 parts:
//
//   Ptr ::= TT AAAAAAAAAAAAAAA BBBBBBBBBBBBBBB
//
// Where:
//
//   T : u8  is the pointer tag 
//   A : u30 is the 1st value
//   B : u30 is the 2nd value
//
// There are 12 possible tags:
//
//   Tag | Val | Meaning  
//   ----| --- | -------------------------------
//   DP0 |   0 | a variable, bound to the 1st argument of a duplication
//   DP1 |   1 | a variable, bound to the 2nd argument of a duplication
//   VAR |   2 | a variable, bound to the one argument of a lambda
//   ARG |   3 | an used argument of a lambda or duplication
//   ERA |   4 | an erased argument of a lambda or duplication
//   LAM |   5 | a lambda
//   APP |   6 | an application
//   SUP |   7 | a superposition
//   CTR |   8 | a constructor
//   FUN |   9 | a function
//   OP2 |  10 | a numeric operation
//   NUM |  11 | a 60-bit number
//
// The semantics of the 1st and 2nd values depend on the pointer tag. 
//
//   Tag | 1st ptr value                | 2nd ptr value
//   --- | ---------------------------- | ---------------------------------
//   DP0 | the duplication label        | points to the duplication node
//   DP1 | the duplication label        | points to the duplication node
//   VAR | not used                     | points to the lambda node
//   ARG | not used                     | points to the variable occurrence
//   ERA | not used                     | not used
//   LAM | not used                     | points to the lambda node
//   APP | not used                     | points to the application node
//   SUP | the duplication label        | points to the superposition node
//   CTR | the constructor name         | points to the constructor node
//   FUN | the function name            | points to the function node
//   OP2 | the operation name           | points to the operation node
//   NUM | the most significant 30 bits | the least significant 30 bits
//
// Notes:
//
//   1. The duplication label is an internal value used on the DUP-SUP rule.
//   2. The operation name only uses 4 of the 30 bits, as there are only 16 ops.
//   3. NUM pointers don't point anywhere, they just store the number directly.
//
// A node is a tuple of N pointers stored on sequential memory indices.
// The meaning of each index depends on the node. There are 7 types:
//
//   Duplication Node:
//   - [0] => either an ERA or an ARG pointing to the 1st variable location
//   - [1] => either an ERA or an ARG pointing to the 2nd variable location
//   - [2] => pointer to the duplicated expression
//
//   Lambda Node:
//   - [0] => either and ERA or an ERA pointing to the variable location
//   - [1] => pointer to the lambda's body
//   
//   Application Node:
//   - [0] => pointer to the lambda
//   - [1] => pointer to the argument
//
//   Superposition Node:
//   - [0] => pointer to the 1st superposed value
//   - [1] => pointer to the 2sd superposed value
//
//   Constructor Node:
//   - [0] => pointer to the 1st field
//   - [1] => pointer to the 2nd field
//   - ... => ...
//   - [N] => pointer to the Nth field
//
//   Function Node:
//   - [0] => pointer to the 1st argument
//   - [1] => pointer to the 2nd argument
//   - ... => ...
//   - [N] => pointer to the Nth argument
//
//   Operation Node:
//   - [0] => pointer to the 1st operand
//   - [1] => pointer to the 2nd operand
//
// Notes:
//
//   1. Duplication nodes DON'T have a body. They "float" on the global scope.
//   2. Lambdas and Duplications point to their variables, and vice-versa.
//   3. ARG pointers can only show up inside Lambdas and Duplications.
//   4. Nums and vars don't require a node type, because they're unboxed.
//   5. Function and Constructor arities depends on the user-provided definition.
//
// Example 0:
// 
//   Term:
//
//    {Tuple2 #7 #8}
//
//   Memory:
//
//     Root : Ptr(CTR, 0x00000001, 0x00000000)
//     0x00 | Ptr(NUM, 0x00000000, 0x00000007) // the tuple's 1st field
//     0x01 | Ptr(NUM, 0x00000000, 0x00000008) // the tuple's 2nd field
//
//   Notes:
//     
//     1. This is just a pair with two numbers.
//     2. The root pointer is not stored on memory.
//     3. The 'Tuple2' name was encoded as the ID 1.
//     4. Since nums are unboxed, a 2-tuple uses 2 memory slots, or 32 bytes.
//
// Example 1:
//
//   Term:
//
//     λ~ λb b
//
//   Memory:
//
//     Root : Ptr(LAM, 0x00000000, 0x00000000)
//     0x00 | Ptr(ERA, 0x00000000, 0x00000000) // 1st lambda's argument
//     0x01 | Ptr(LAM, 0x00000000, 0x00000002) // 1st lambda's body
//     0x02 | Ptr(ARG, 0x00000000, 0x00000003) // 2nd lambda's argument
//     0x03 | Ptr(VAR, 0x00000000, 0x00000002) // 2nd lambda's body
//
//   Notes:
//
//     1. This is a λ-term that discards the 1st argument and returns the 2nd.
//     2. The 1st lambda's argument not used, thus, an ERA pointer.
//     3. The 2nd lambda's argument points to its variable, and vice-versa.
//     4. Each lambda uses 2 memory slots. This term uses 64 bytes in total.
//     
// Example 2:
//
//   Term:
//     
//     λx dup x0 x1 = x; (* x0 x1)
//
//   Memory:
//
//     Root : Ptr(LAM, 0x00000000, 0x00000000)
//     0x00 | Ptr(ARG, 0x00000000, 0x00000004) // the lambda's argument
//     0x01 | Ptr(OP2, 0x00000002, 0x00000005) // the lambda's body
//     0x02 | Ptr(ARG, 0x00000000, 0x00000005) // the duplication's 1st argument
//     0x03 | Ptr(ARG, 0x00000000, 0x00000006) // the duplication's 2nd argument
//     0x04 | Ptr(VAR, 0x00000000, 0x00000000) // the duplicated expression
//     0x05 | Ptr(DP0, 0xba31fb21, 0x00000002) // the operator's 1st operand
//     0x06 | Ptr(DP1, 0xba31fb21, 0x00000002) // the operator's 2st operand
//
//   Notes:
//     
//     1. This is a lambda function that squares a number.
//     2. Notice how every ARGs point to a VAR/DP0/DP1, that points back its source node.
//     3. DP1 does not point to its ARG. It points to the duplication node, which is at 0x02.
//     4. The lambda's body does not point to the dup node, but to the operator. Dup nodes float.
//     5. 0xba31fb21 is a globally unique random label assigned to the duplication node.
//     6. That duplication label is stored on the DP0/DP1 that point to the node, not on the node.
//     7. A lambda uses 2 memory slots, a duplication uses 3, an operator uses 2. Total: 112 bytes.
//     8. In-memory size is different to, and larger than, serialization size.

#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]

use std::collections::{hash_map, HashMap};

// Constants
// ---------

const U64_PER_KB: u64 = 0x80;
const U64_PER_MB: u64 = 0x20000;
const U64_PER_GB: u64 = 0x8000000;

pub const MAX_ARITY: u64 = 16;
pub const MEM_SPACE: u64 = U64_PER_GB;
pub const MAX_DYNFUNS: u64 = 65536;

pub const SEEN_SIZE: usize = 4194304; // uses 32 MB, covers heaps up to 2 GB

pub const VAL: u64 = 1;
pub const EXT: u64 = 0x100000000;
pub const ARI: u64 = 0x100000000000000;
pub const TAG: u64 = 0x1000000000000000;

pub const NUM_MASK: u64 = 0xFFF_FFFF_FFFF_FFFF;

pub const DP0: u64 = 0x0;
pub const DP1: u64 = 0x1;
pub const VAR: u64 = 0x2;
pub const ARG: u64 = 0x3;
pub const ERA: u64 = 0x4;
pub const LAM: u64 = 0x5;
pub const APP: u64 = 0x6;
pub const PAR: u64 = 0x7;
pub const CTR: u64 = 0x8;
pub const CAL: u64 = 0x9;
pub const OP2: u64 = 0xA;
pub const NUM: u64 = 0xB;
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

// Types
// -----

pub type Lnk = u64;

pub type Rewriter = Box<dyn Fn(&mut Worker, &mut u64, u64, Lnk) -> bool>;

pub struct Function {
  pub arity: u64,
  pub stricts: Vec<u64>,
  pub rewriter: Rewriter,
}

pub struct Arity(pub u64);

type Funs = Vec<Option<Function>>;
type Aris = Vec<Arity>;

pub struct Worker {
  pub node: Vec<Lnk>,
  pub funs: Funs,
  pub aris: Aris,
  pub size: u64,
  pub free: Vec<Vec<u64>>,
  pub cost: u64,
}

pub fn new_worker() -> Worker {
  Worker {
    node: vec![0; 6 * 0x8000000],
    aris: vec![],
    funs: vec![],
    size: 0,
    free: vec![vec![]; 16],
    cost: 0,
  }
}

// Globals
// -------

static mut SEEN_DATA: [u64; SEEN_SIZE] = [0; SEEN_SIZE];
static mut CALL_COUNT: &mut [u64] = &mut [0; MAX_DYNFUNS as usize];

// Constructors
// ------------

pub fn Var(pos: u64) -> Lnk {
  (VAR * TAG) | pos
}

pub fn Dp0(col: u64, pos: u64) -> Lnk {
  (DP0 * TAG) | (col * EXT) | pos
}

pub fn Dp1(col: u64, pos: u64) -> Lnk {
  (DP1 * TAG) | (col * EXT) | pos
}

pub fn Arg(pos: u64) -> Lnk {
  (ARG * TAG) | pos
}

pub fn Era() -> Lnk {
  ERA * TAG
}

pub fn Lam(pos: u64) -> Lnk {
  (LAM * TAG) | pos
}

pub fn App(pos: u64) -> Lnk {
  (APP * TAG) | pos
}

pub fn Par(col: u64, pos: u64) -> Lnk {
  (PAR * TAG) | (col * EXT) | pos
}

pub fn Op2(ope: u64, pos: u64) -> Lnk {
  (OP2 * TAG) | (ope * EXT) | pos
}

pub fn Num(val: u64) -> Lnk {
  (NUM * TAG) | (val & NUM_MASK)
}

pub fn Nil() -> Lnk {
  NIL * TAG
}

pub fn Ctr(ari: u64, fun: u64, pos: u64) -> Lnk {
  (CTR * TAG) | (ari * ARI) | (fun * EXT) | pos
}

pub fn Cal(ari: u64, fun: u64, pos: u64) -> Lnk {
  (CAL * TAG) | (ari * ARI) | (fun * EXT) | pos
}

pub fn Out(arg: u64, fld: u64) -> Lnk {
  (OUT * TAG) | (arg << 8) | fld
}

// Getters
// -------

pub fn get_tag(lnk: Lnk) -> u64 {
  lnk / TAG
}

pub fn get_ext(lnk: Lnk) -> u64 {
  (lnk / EXT) & 0xFF_FFFF
}

pub fn get_val(lnk: Lnk) -> u64 {
  lnk & 0xFFFF_FFFF
}

pub fn get_num(lnk: Lnk) -> u64 {
  lnk & 0xFFF_FFFF_FFFF_FFFF
}

pub fn get_ari(lnk: Lnk) -> u64 {
  (lnk / ARI) & 0xF
}

pub fn get_loc(lnk: Lnk, arg: u64) -> u64 {
  get_val(lnk) + arg
}

// Memory
// ------

pub fn ask_ari(mem: &Worker, lnk: Lnk) -> u64 {
  let got = match mem.aris.get(get_ext(lnk) as usize) {
    Some(Arity(arit)) => *arit,
    None              => 0,
  };
  // TODO: remove this in a future update where ari will be removed from the lnk
  if get_ari(lnk) != got {
    println!("[WARNING] arity inconsistency");
  }
  return got;
}

pub fn ask_lnk(mem: &Worker, loc: u64) -> Lnk {
  unsafe { *mem.node.get_unchecked(loc as usize) }
  // mem.node[loc as usize]
}

pub fn ask_arg(mem: &Worker, term: Lnk, arg: u64) -> Lnk {
  ask_lnk(mem, get_loc(term, arg))
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
    0
  } else if let Some(reuse) = mem.free[size as usize].pop() {
    reuse
  } else {
    let loc = mem.size;
    mem.size += size;
    loc
  }
}

pub fn clear(mem: &mut Worker, loc: u64, size: u64) {
  mem.free[size as usize].push(loc);
}

pub fn collect(mem: &mut Worker, term: Lnk) {
  let mut stack: Vec<Lnk> = Vec::new();
  let mut next = term;
  //let mut dups : Vec<u64> = Vec::new();
  loop {
    let term = next;
    match get_tag(term) {
      DP0 => {
        link(mem, get_loc(term, 0), Era());
        //dups.push(term);
      }
      DP1 => {
        link(mem, get_loc(term, 1), Era());
        //dups.push(term);
      }
      VAR => {
        link(mem, get_loc(term, 0), Era());
      }
      LAM => {
        if get_tag(ask_arg(mem, term, 0)) != ERA {
          link(mem, get_loc(ask_arg(mem, term, 0), 0), Era());
        }
        next = ask_arg(mem, term, 1);
        clear(mem, get_loc(term, 0), 2);
        continue;
      }
      APP => {
        stack.push(ask_arg(mem, term, 0));
        next = ask_arg(mem, term, 1);
        clear(mem, get_loc(term, 0), 2);
        continue;
      }
      PAR => {
        stack.push(ask_arg(mem, term, 0));
        next = ask_arg(mem, term, 1);
        clear(mem, get_loc(term, 0), 2);
        continue;
      }
      OP2 => {
        stack.push(ask_arg(mem, term, 0));
        next = ask_arg(mem, term, 1);
        clear(mem, get_loc(term, 0), 2);
        continue;
      }
      NUM => {}
      CTR | CAL => {
        let arity = ask_ari(mem, term);
        for i in 0..arity {
          if i < arity - 1 {
            stack.push(ask_arg(mem, term, i));
          } else {
            next = ask_arg(mem, term, i);
          }
        }
        clear(mem, get_loc(term, 0), arity);
        if arity > 0 {
          continue;
        }
      }
      _ => {}
    }
    if let Some(got) = stack.pop() {
      next = got;
    } else {
      break;
    }
  }
  // TODO: add this to the C version
  //for dup in dups {
    //let fst = ask_arg(mem, dup, 0);
    //let snd = ask_arg(mem, dup, 1);
    //if get_tag(fst) == ERA && get_tag(snd) == ERA {
      //collect(mem, ask_arg(mem, dup, 2));
      //clear(mem, get_loc(dup, 0), 3);
    //}
  //}
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
  let arit = ask_ari(mem, term);
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
  done
}

pub fn reduce(
  mem: &mut Worker,
  dups: &mut u64,
  root: u64,
  _i2n: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let mut stack: Vec<u64> = Vec::new();

  let mut init = 1;
  let mut host = root;
  let funs = std::mem::take(&mut mem.funs); // necessary to satisfy the burrow checker

  loop {
    let term = ask_lnk(mem, host);

    if debug {
      println!("------------------------");
      println!("{}", show_term(mem, ask_lnk(mem, 0), _i2n, term));
    }

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
          stack.push(get_loc(term, 1) | 0x80000000);
          host = get_loc(term, 0);
          continue;
        }
        CAL => {
          let fid = get_ext(term);
          //let ari = ask_ari(mem, term);
          if let Some(Some(f)) = &funs.get(fid as usize) {
            let len = f.stricts.len() as u64;
            if len == 0 {
              init = 0;
            } else {
              stack.push(host);
              for (i, strict) in f.stricts.iter().enumerate() {
                if i < f.stricts.len() - 1 {
                  stack.push(get_loc(term, *strict) | 0x80000000);
                } else {
                  host = get_loc(term, *strict);
                }
              }
            }
            continue;
          }
        }
        _ => {}
      }
    } else {
      match get_tag(term) {
        APP => {
          let arg0 = ask_arg(mem, term, 0);
          if get_tag(arg0) == LAM {
            //println!("app-lam");
            inc_cost(mem);
            subst(mem, ask_arg(mem, arg0, 0), ask_arg(mem, term, 1));
            let _done = link(mem, host, ask_arg(mem, arg0, 1));
            clear(mem, get_loc(term, 0), 2);
            clear(mem, get_loc(arg0, 0), 2);
            init = 1;
            continue;
          }
          if get_tag(arg0) == PAR {
            //println!("app-sup");
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
        }
        DP0 | DP1 => {
          let arg0 = ask_arg(mem, term, 2);
          // let argK = ask_arg(mem, term, if get_tag(term) == DP0 { 1 } else { 0 });
          // if get_tag(argK) == ERA {
          //   let done = arg0;
          //   link(mem, host, done);
          //   init = 1;
          //   continue;
          // }
          if get_tag(arg0) == LAM {
            //println!("dup-lam");
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
          } else if get_tag(arg0) == PAR {
            //println!("dup-sup");
            if get_ext(term) == get_ext(arg0) {
              inc_cost(mem);
              subst(mem, ask_arg(mem, term, 0), ask_arg(mem, arg0, 0));
              subst(mem, ask_arg(mem, term, 1), ask_arg(mem, arg0, 1));
              let _done =
                link(mem, host, ask_arg(mem, arg0, if get_tag(term) == DP0 { 0 } else { 1 }));
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
              let done = Par(get_ext(arg0), if get_tag(term) == DP0 { par0 } else { par1 });
              link(mem, host, done);
            }
          } else if get_tag(arg0) == NUM {
            //println!("dup-u32");
            inc_cost(mem);
            subst(mem, ask_arg(mem, term, 0), arg0);
            subst(mem, ask_arg(mem, term, 1), arg0);
            clear(mem, get_loc(term, 0), 3);
            let _done = arg0;
            link(mem, host, arg0);
          } else if get_tag(arg0) == CTR {
            //println!("dup-ctr");
            inc_cost(mem);
            let fnid = get_ext(arg0);
            let arit = ask_ari(mem, arg0);
            if arit == 0 {
              subst(mem, ask_arg(mem, term, 0), Ctr(0, fnid, 0));
              subst(mem, ask_arg(mem, term, 1), Ctr(0, fnid, 0));
              clear(mem, get_loc(term, 0), 3);
              let _done = link(mem, host, Ctr(0, fnid, 0));
            } else {
              let ctr0 = get_loc(arg0, 0);
              let ctr1 = alloc(mem, arit);
              for i in 0..arit - 1 {
                let leti = alloc(mem, 3);
                link(mem, leti + 2, ask_arg(mem, arg0, i));
                link(mem, ctr0 + i, Dp0(get_ext(term), leti));
                link(mem, ctr1 + i, Dp1(get_ext(term), leti));
              }
              let leti = get_loc(term, 0);
              link(mem, leti + 2, ask_arg(mem, arg0, arit - 1));
              let term_arg_0 = ask_arg(mem, term, 0);
              link(mem, ctr0 + arit - 1, Dp0(get_ext(term), leti));
              subst(mem, term_arg_0, Ctr(arit, fnid, ctr0));
              let term_arg_1 = ask_arg(mem, term, 1);
              link(mem, ctr1 + arit - 1, Dp1(get_ext(term), leti));
              subst(mem, term_arg_1, Ctr(arit, fnid, ctr1));
              let done = Ctr(arit, fnid, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
              link(mem, host, done);
            }
          } else if get_tag(arg0) == ERA {
            inc_cost(mem);
            subst(mem, ask_arg(mem, term, 0), Era());
            subst(mem, ask_arg(mem, term, 1), Era());
            link(mem, host, Era());
            clear(mem, get_loc(term, 0), 3);
            init = 1;
            continue;
          }
        }
        OP2 => {
          let arg0 = ask_arg(mem, term, 0);
          let arg1 = ask_arg(mem, term, 1);
          if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
            //println!("op2-u32");
            inc_cost(mem);
            let a = get_num(arg0);
            let b = get_num(arg1);
            let c = match get_ext(term) {
              ADD => a.wrapping_add(b) & NUM_MASK,
              SUB => a.wrapping_sub(b) & NUM_MASK,
              MUL => a.wrapping_mul(b) & NUM_MASK,
              DIV => a.wrapping_div(b) & NUM_MASK,
              MOD => a.wrapping_rem(b) & NUM_MASK,
              AND => (a &  b) & NUM_MASK,
              OR  => (a |  b) & NUM_MASK,
              XOR => (a ^  b) & NUM_MASK,
              SHL => a.wrapping_shl(b as u32) & NUM_MASK,
              SHR => a.wrapping_shr(b as u32) & NUM_MASK,
              LTN => u64::from(a <  b),
              LTE => u64::from(a <= b),
              EQL => u64::from(a == b),
              GTE => u64::from(a >= b),
              GTN => u64::from(a >  b),
              NEQ => u64::from(a != b),
              _   => panic!("Invalid operation!"),
            };
            let done = Num(c);
            clear(mem, get_loc(term, 0), 2);
            link(mem, host, done);
          } else if get_tag(arg0) == PAR {
            //println!("op2-sup-0");
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
          } else if get_tag(arg1) == PAR {
            //println!("op2-sup-1");
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
        CAL => {
          let fid = get_ext(term);
          let _ari = ask_ari(mem, term);
          if let Some(Some(f)) = &funs.get(fid as usize) {
            if (f.rewriter)(mem, dups, host, term) {
              //unsafe { CALL_COUNT[fun as usize] += 1; } //TODO: uncomment
              init = 1;
              continue;
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
  mem.funs = funs;
  ask_lnk(mem, root)
}

pub fn set_bit(bits: &mut [u64], bit: u64) {
  bits[bit as usize >> 6] |= 1 << (bit & 0x3f);
}

pub fn get_bit(bits: &[u64], bit: u64) -> bool {
  (((bits[bit as usize >> 6] >> (bit & 0x3f)) as u8) & 1) == 1
}

pub fn normal_go(
  mem: &mut Worker,
  dups: &mut u64,
  host: u64,
  seen: &mut [u64],
  i2n: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let term = ask_lnk(mem, host);
  if get_bit(seen, host) {
    term
  } else {
    let term = reduce(mem, dups, host, i2n, debug);
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
      CTR | CAL => {
        let arity = ask_ari(mem, term);
        for i in 0..arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    for loc in rec_locs {
      let lnk: Lnk = normal_go(mem, dups, loc, seen, i2n, debug);
      link(mem, loc, lnk);
    }
    term
  }
}

pub fn normal(
  mem: &mut Worker,
  host: u64,
  i2n: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let mut done;
  let mut dups = 0;
  let mut cost = mem.cost;
  loop {
    let mut seen = vec![0; 4194304];
    done = normal_go(mem, &mut dups, host, &mut seen, i2n, debug);
    if mem.cost != cost {
      cost = mem.cost;
    } else {
      break;
    }
  }
  //print_call_counts(i2n); // TODO: uncomment
  done
}

// Debug: prints call counts
fn print_call_counts(i2n: Option<&HashMap<u64, String>>) {
  unsafe {
    let mut counts: Vec<(String, u64)> = Vec::new();
    for fun in 0..MAX_DYNFUNS {
      if let Some(id_to_name) = i2n {
        match id_to_name.get(&fun) {
          None => {
            break;
          }
          Some(fun_name) => {
            counts.push((fun_name.clone(), CALL_COUNT[fun as usize]));
          }
        }
      }
    }
    counts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (name, count) in counts {
      println!("{} - {}", name, count);
    }
    println!();
  }
}

// Debug
// -----

pub fn show_lnk(x: Lnk) -> String {
  if x == 0 {
    String::from("~")
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
      CAL => "CAL",
      OP2 => "OP2",
      NUM => "NUM",
      OUT => "OUT",
      NIL => "NIL",
      _ => "?",
    };
    format!("{}:{:x}:{:x}", tgs, ext, val)
  }
}

pub fn show_mem(worker: &Worker) -> String {
  let mut s: String = String::new();
  for i in 0..48 {
    // pushes to the string
    s.push_str(&format!("{:x} | ", i));
    s.push_str(&show_lnk(worker.node[i]));
    s.push('\n');
  }
  s
}

pub fn show_term(
  mem: &Worker,
  term: Lnk,
  i2n: Option<&HashMap<u64, String>>,
  focus: u64,
) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    mem: &Worker,
    term: Lnk,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      PAR => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(mem, ask_arg(mem, term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(mem, ask_arg(mem, term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      CTR | CAL => {
        let arity = ask_ari(mem, term);
        for i in 0..arity {
          find_lets(mem, ask_arg(mem, term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    mem: &Worker,
    term: Lnk,
    names: &HashMap<u64, String>,
    i2n: Option<&HashMap<u64, String>>,
    focus: u64,
  ) -> String {
    let done = match get_tag(term) {
      DP0 => {
        format!("a{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?a")))
      }
      DP1 => {
        format!("b{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?b")))
      }
      VAR => {
        format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?c")))
      }
      LAM => {
        let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?d")));
        format!("λ{} {}", name, go(mem, ask_arg(mem, term, 1), names, i2n, focus))
      }
      APP => {
        let func = go(mem, ask_arg(mem, term, 0), names, i2n, focus);
        let argm = go(mem, ask_arg(mem, term, 1), names, i2n, focus);
        format!("({} {})", func, argm)
      }
      PAR => {
        //let kind = get_ext(term);
        let func = go(mem, ask_arg(mem, term, 0), names, i2n, focus);
        let argm = go(mem, ask_arg(mem, term, 1), names, i2n, focus);
        format!("{{{} {}}}", func, argm)
      }
      OP2 => {
        let oper = get_ext(term);
        let val0 = go(mem, ask_arg(mem, term, 0), names, i2n, focus);
        let val1 = go(mem, ask_arg(mem, term, 1), names, i2n, focus);
        let symb = match oper {
          0x0 => "+",
          0x1 => "-",
          0x2 => "*",
          0x3 => "/",
          0x4 => "%",
          0x5 => "&",
          0x6 => "|",
          0x7 => "^",
          0x8 => "<<",
          0x9 => ">>",
          0xA => "<",
          0xB => "<=",
          0xC => "=",
          0xD => ">=",
          0xE => ">",
          0xF => "!=",
          _ => "?e",
        };
        format!("({} {} {})", symb, val0, val1)
      }
      NUM => {
        format!("{}", get_val(term))
      }
      CTR | CAL => {
        let func = get_ext(term);
        let arit = ask_ari(mem, term);
        let args: Vec<String> =
          (0..arit).map(|i| go(mem, ask_arg(mem, term, i), names, i2n, focus)).collect();
        let name = if let Some(id_to_name) = i2n {
          id_to_name.get(&func).unwrap_or(&String::from("?f")).clone()
        } else {
          format!(
            "{}{}",
            if get_tag(term) < CAL { String::from("C") } else { String::from("F") },
            func
          )
        };
        format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
      ERA => "*".to_string(),
      _ => format!("?g({})", get_tag(term)),
    };
    if term == focus {
      format!("${}", done)
    } else {
      done
    }
  }
  find_lets(mem, term, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(mem, term, &names, i2n, focus);
  for (_key, pos) in lets {
    // todo: reverse
    let what = String::from("?h");
    //let kind = kinds.get(&key).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 =
      if ask_lnk(mem, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 =
      if ask_lnk(mem, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!(
      "\ndup {} {} = {};",
      //kind,
      nam0,
      nam1,
      go(mem, ask_lnk(mem, pos + 2), &names, i2n, focus)
    ));
  }
  text
}
