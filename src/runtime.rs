// HVM's memory model
// ------------------
// 
// The runtime memory consists of just a vector of u64 pointers. That is:
//
//   Mem ::= Vec<Ptr>
// 
// A pointer has 3 parts:
//
//   Ptr ::= 0xTAAAAAAABBBBBBBB
//
// Where:
//
//   T : u4  is the pointer tag 
//   A : u28 is the 1st value
//   B : u32 is the 2nd value
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
//   NUM | the most significant 28 bits | the least significant 32 bits
//
// Notes:
//
//   1. The duplication label is an internal value used on the DUP-SUP rule.
//   2. The operation name only uses 4 of the 28 bits, as there are only 16 ops.
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
//     Root : Ptr(CTR, 0x0000001, 0x00000000)
//     0x00 | Ptr(NUM, 0x0000000, 0x00000007) // the tuple's 1st field
//     0x01 | Ptr(NUM, 0x0000000, 0x00000008) // the tuple's 2nd field
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
//     Root : Ptr(LAM, 0x0000000, 0x00000000)
//     0x00 | Ptr(ERA, 0x0000000, 0x00000000) // 1st lambda's argument
//     0x01 | Ptr(LAM, 0x0000000, 0x00000002) // 1st lambda's body
//     0x02 | Ptr(ARG, 0x0000000, 0x00000003) // 2nd lambda's argument
//     0x03 | Ptr(VAR, 0x0000000, 0x00000002) // 2nd lambda's body
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
//     Root : Ptr(LAM, 0x0000000, 0x00000000)
//     0x00 | Ptr(ARG, 0x0000000, 0x00000004) // the lambda's argument
//     0x01 | Ptr(OP2, 0x0000002, 0x00000005) // the lambda's body
//     0x02 | Ptr(ARG, 0x0000000, 0x00000005) // the duplication's 1st argument
//     0x03 | Ptr(ARG, 0x000000, 0x00000006) // the duplication's 2nd argument
//     0x04 | Ptr(VAR, 0x0000000, 0x00000000) // the duplicated expression
//     0x05 | Ptr(DP0, 0xa31fb21, 0x00000002) // the operator's 1st operand
//     0x06 | Ptr(DP1, 0xa31fb21, 0x00000002) // the operator's 2st operand
//
//   Notes:
//     
//     1. This is a lambda function that squares a number.
//     2. Notice how every ARGs point to a VAR/DP0/DP1, that points back its source node.
//     3. DP1 does not point to its ARG. It points to the duplication node, which is at 0x02.
//     4. The lambda's body does not point to the dup node, but to the operator. Dup nodes float.
//     5. 0xa31fb21 is a globally unique random label assigned to the duplication node.
//     6. That duplication label is stored on the DP0/DP1 that point to the node, not on the node.
//     7. A lambda uses 2 memory slots, a duplication uses 3, an operator uses 2. Total: 112 bytes.
//     8. In-memory size is different to, and larger than, serialization size.

#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

use std::borrow::Cow;
use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

// Constants
// ---------

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

pub const VAL: u64 = 1;
pub const EXT: u64 = 0x100000000;
pub const TAG: u64 = 0x1000000000000000;

pub const NUM_MASK: u64 = 0xFFF_FFFF_FFFF_FFFF;

pub const DP0: u64 = 0x0;
pub const DP1: u64 = 0x1;
pub const VAR: u64 = 0x2;
pub const ARG: u64 = 0x3;
pub const ERA: u64 = 0x4;
pub const LAM: u64 = 0x5;
pub const APP: u64 = 0x6;
pub const SUP: u64 = 0x7;
pub const CTR: u64 = 0x8;
pub const FUN: u64 = 0x9;
pub const OP2: u64 = 0xA;
pub const NUM: u64 = 0xB;
pub const NIL: u64 = 0xF;

pub const ADD: u64 = 0x0;
pub const SUB: u64 = 0x1;
pub const MUL: u64 = 0x2;
pub const DIV: u64 = 0x3;
pub const MOD: u64 = 0x4;
pub const AND: u64 = 0x5;
pub const OR : u64 = 0x6;
pub const XOR: u64 = 0x7;
pub const SHL: u64 = 0x8;
pub const SHR: u64 = 0x9;
pub const LTN: u64 = 0xA;
pub const LTE: u64 = 0xB;
pub const EQL: u64 = 0xC;
pub const GTE: u64 = 0xD;
pub const GTN: u64 = 0xE;
pub const NEQ: u64 = 0xF;

// Reserved ids
pub const HVM_LOG      : u64 = 0;
pub const HVM_PUT      : u64 = 1;
pub const STRING_NIL   : u64 = 2;
pub const STRING_CONS  : u64 = 3;
pub const IO_DONE      : u64 = 4;
pub const IO_DO_INPUT  : u64 = 5;
pub const IO_DO_OUTPUT : u64 = 6;
pub const IO_DO_FETCH  : u64 = 7;
pub const IO_DO_STORE  : u64 = 8;
pub const IO_DO_LOAD   : u64 = 9;

// This is a Kind2-specific optimization. Check 'HOAS_OPT'.
pub const HOAS_CT0 : u64 = 10;
pub const HOAS_CT1 : u64 = 11;
pub const HOAS_CT2 : u64 = 12;
pub const HOAS_CT3 : u64 = 13;
pub const HOAS_CT4 : u64 = 14;
pub const HOAS_CT5 : u64 = 15;
pub const HOAS_CT6 : u64 = 16;
pub const HOAS_CT7 : u64 = 17;
pub const HOAS_CT8 : u64 = 18;
pub const HOAS_CT9 : u64 = 19;
pub const HOAS_CTA : u64 = 20;
pub const HOAS_CTB : u64 = 21;
pub const HOAS_CTC : u64 = 22;
pub const HOAS_CTD : u64 = 23;
pub const HOAS_CTE : u64 = 24;
pub const HOAS_CTF : u64 = 25;
pub const HOAS_CTG : u64 = 26;
pub const HOAS_NUM : u64 = 27;

//GENERATED-FUN-IDS//

pub const HVM_MAX_RESERVED_ID : u64 = HOAS_NUM;

// Types
// -----

pub type Ptr = u64;
pub type AtomicPtr = AtomicU64;

// A runtime term
#[derive(Debug)]
pub enum Term {
  Var { bidx: u64 },
  Glo { glob: u64 },
  Dup { eras: (bool, bool), expr: Box<Term>, body: Box<Term> },
  Let { expr: Box<Term>, body: Box<Term> },
  Lam { eras: bool, glob: u64, body: Box<Term> },
  App { func: Box<Term>, argm: Box<Term> },
  Fun { func: u64, args: Vec<Term> },
  Ctr { func: u64, args: Vec<Term> },
  Num { numb: u64 },
  Op2 { oper: u64, val0: Box<Term>, val1: Box<Term> },
}

// A runtime rule
#[derive(Debug)]
pub struct Rule {
  pub hoas: bool,
  pub cond: Vec<Ptr>,
  pub vars: Vec<RuleVar>,
  pub term: Term,
  pub body: RuleBody,
  pub free: Vec<(u64, u64)>,
}

// A rule left-hand side variable
#[derive(Debug)]
pub struct RuleVar {
  pub param: u64,
  pub field: Option<u64>,
  pub erase: bool,
}

// The rule right-hand side body (TODO: can the RuleBodyNode Vec be unboxed?)
pub type RuleBody = (RuleBodyCell, Vec<RuleBodyNode>, u64);

// A body node
pub type RuleBodyNode = Vec<RuleBodyCell>;

// A body cell
#[derive(Copy, Clone, Debug)]
pub enum RuleBodyCell {
  Fix { value: u64 }, // Fixed value, doesn't require adjustment
  Loc { value: u64, targ: u64, slot: u64 }, // Local link, requires adjustment
  Ext { index: u64 }, // Link to an external variable
}

pub struct Function {
  pub arity: u64,
  pub is_strict: Vec<bool>,
  pub stricts: Vec<u64>,
  pub rules: Vec<Rule>,
}

pub struct Arity(pub u64);

pub type Funs = Vec<Option<Function>>;
pub type Aris = Vec<Arity>;
pub type Nams = HashMap<u64, String>;

// Static information
pub struct Info {
  pub funs: Funs,
  pub aris: Aris,
  pub nams: Nams,
}

// Global memory buffer
pub struct Heap {
  node: Box<[AtomicPtr]>,
  lock: Box<[AtomicU8]>,
}

// Thread local data and stats
#[derive(Clone, Debug)]
pub struct Stat {
  pub tid: usize,
  pub free: Vec<Vec<u64>>, // free cells to re-allocate
  pub abuf: Vec<u64>, // allocation buffer used by alloc_body
  pub used: u64, // number of used memory cells
  pub next: u64, // next allocation index
  pub dups: u64, // next dup label to be created
  pub cost: u64, // total number of rewrite rules
}

pub type Stats = [Stat];

fn available_parallelism() -> usize {
  return 8;
  //return std::thread::available_parallelism().unwrap().get();
}

// Initializers
// ------------

const HEAP_SIZE : usize = 4 * CELLS_PER_GB;

pub fn new_atomic_u8_array(size: usize) -> Box<[AtomicU8]> {
  return unsafe { Box::from_raw(AtomicU8::from_mut_slice(Box::leak(vec![0xFFu8; size].into_boxed_slice()))) }
}

pub fn new_atomic_u64_array(size: usize) -> Box<[AtomicU64]> {
  return unsafe { Box::from_raw(AtomicU64::from_mut_slice(Box::leak(vec![0u64; size].into_boxed_slice()))) }
}

pub fn new_heap() -> Heap {
  let size = HEAP_SIZE; // FIXME: accept size param
  let node = new_atomic_u64_array(size);
  let lock = new_atomic_u8_array(size);
  return Heap { node, lock };
}

pub fn new_stats() -> Box<Stats> {
  let thread_count = available_parallelism();
  let mut stats = vec![];
  for tid in 0 .. thread_count {
    stats.push(Stat {
      tid: tid,
      abuf: vec![0; 32 * 256 * 256],
      free: vec![vec![]; 32],
      used: 0,
      next: (HEAP_SIZE / thread_count * tid) as u64,
      dups: ((1 << 28) / thread_count * tid) as u64,
      cost: 0,
    })
  }
  return stats.into_boxed_slice();
}

pub fn new_info() -> Info {
  Info {
    funs: vec![],
    aris: vec![],
    nams: HashMap::new(),
  }
}

// Pointers
// --------

pub fn Var(pos: u64) -> Ptr {
  (VAR * TAG) | pos
}

pub fn Dp0(col: u64, pos: u64) -> Ptr {
  (DP0 * TAG) | (col * EXT) | pos
}

pub fn Dp1(col: u64, pos: u64) -> Ptr {
  (DP1 * TAG) | (col * EXT) | pos
}

pub fn Arg(pos: u64) -> Ptr {
  (ARG * TAG) | pos
}

pub fn Era() -> Ptr {
  ERA * TAG
}

pub fn Lam(pos: u64) -> Ptr {
  (LAM * TAG) | pos
}

pub fn App(pos: u64) -> Ptr {
  (APP * TAG) | pos
}

pub fn Par(col: u64, pos: u64) -> Ptr {
  (SUP * TAG) | (col * EXT) | pos
}

pub fn Op2(ope: u64, pos: u64) -> Ptr {
  (OP2 * TAG) | (ope * EXT) | pos
}

pub fn Num(val: u64) -> Ptr {
  (NUM * TAG) | (val & NUM_MASK)
}

pub fn Ctr(_ari: u64, fun: u64, pos: u64) -> Ptr {
  (CTR * TAG) | (fun * EXT) | pos
}

// FIXME: update name to Fun
pub fn Fun(_ari: u64, fun: u64, pos: u64) -> Ptr {
  (FUN * TAG) | (fun * EXT) | pos
}

// Pointer Getters
// ---------------

pub fn get_tag(lnk: Ptr) -> u64 {
  lnk / TAG
}

pub fn get_ext(lnk: Ptr) -> u64 {
  (lnk / EXT) & 0xFFF_FFFF
}

pub fn get_val(lnk: Ptr) -> u64 {
  lnk & 0xFFFF_FFFF
}

pub fn get_num(lnk: Ptr) -> u64 {
  lnk & 0xFFF_FFFF_FFFF_FFFF
}

pub fn get_loc(lnk: Ptr, arg: u64) -> u64 {
  get_val(lnk) + arg
}

// Stats
// -----

pub fn get_cost(stats: &Stats) -> u64 {
  stats.iter().map(|x| x.cost).sum()
}

pub fn get_used(stats: &Stats) -> u64 {
  stats.iter().map(|x| x.cost).sum()
}

// Memory
// ------

pub fn ask_ari(info: &Info, lnk: Ptr) -> u64 {
  let fid = get_ext(lnk);
  match fid {
    HVM_LOG => { return 2; }
    HVM_PUT => { return 2; }
    STRING_NIL => { return 0; }
    STRING_CONS => { return 2; }
    IO_DONE => { return 1; }
    IO_DO_INPUT => { return 1; }
    IO_DO_OUTPUT => { return 2; }
    IO_DO_FETCH => { return 3; }
    IO_DO_STORE => { return 3; }
    IO_DO_LOAD => { return 2; }
//GENERATED-FUN-ARI//
    _ => {
      // Dynamic functions
      if let Some(Arity(arit)) = info.aris.get(fid as usize) {
        return *arit;
      }
      return 0;
    }
  }
}

// DEPRECATED
pub fn ask_lnk(heap: &Heap, loc: u64) -> Ptr {
  unsafe { *(*heap.node.get_unchecked(loc as usize)).as_mut_ptr() }
}

// DEPRECATED
pub fn ask_arg(heap: &Heap, term: Ptr, arg: u64) -> Ptr {
  ask_lnk(heap, get_loc(term, arg))
}

// Given a location, loads the ptr stored on it
pub fn load_ptr(heap: &Heap, loc: u64) -> Ptr {
  unsafe { heap.node.get_unchecked(loc as usize).load(Ordering::Relaxed) }
}

// Given a pointer to a node, loads its nth field
pub fn load_field(heap: &Heap, term: Ptr, field: u64) -> Ptr {
  load_ptr(heap, get_loc(term, field))
}

// Given a location, takes the ptr stored on it
pub fn take_ptr(heap: &Heap, loc: u64) -> Ptr {
  unsafe { heap.node.get_unchecked(loc as usize).swap(0, Ordering::Relaxed) }
}

// Given a pointer to a node, takes its nth field
pub fn take_field(heap: &Heap, term: Ptr, field: u64) -> Ptr {
  take_ptr(heap, get_loc(term, field))
}

// Writes a ptr to memory. Updates binders.
pub fn link(heap: &Heap, loc: u64, ptr: Ptr) -> Ptr {
  unsafe {
    *(*heap.node.get_unchecked(loc as usize)).as_mut_ptr() = ptr; // FIXME: safe?
    //heap.node.get_unchecked(loc as usize).store(ptr, Ordering::Relaxed);
    if get_tag(ptr) <= VAR {
      let arg_loc = get_loc(ptr, get_tag(ptr) & 0x01);
      heap.node.get_unchecked(arg_loc as usize).store(Arg(loc), Ordering::Relaxed);
    }
  }
  ptr
}

pub fn alloc(stat: &mut Stat, arity: u64) -> u64 {
  if arity == 0 {
    0
  } else if let Some(reuse) = stat.free[arity as usize].pop() {
    stat.used += arity;
    reuse
  } else {
    let loc = stat.next;
    stat.next += arity;
    stat.used += arity;
    loc
  }
}

pub fn free(heap: &Heap, stat: &mut Stat, loc: u64, arity: u64) {
  stat.free[arity as usize].push(loc);
  stat.used -= arity;
  for i in 0 .. arity {
    heap.node[(loc + i) as usize].store(0, Ordering::Relaxed); 
  }
}

// Generates a new, globally unique dup label
pub fn gen_dup(stat: &mut Stat) -> u64 {
  let dup = stat.dups;
  stat.dups += 1;
  return dup & 0xFFFFFF;
}

pub fn inc_cost(stat: &mut Stat) {
  stat.cost += 1;
}

// Substitution
// ------------

pub fn subst(heap: &Heap, stat: &mut Stat, info: &Info, lnk: Ptr, val: Ptr) {
  let var = load_ptr(heap, get_loc(lnk, get_tag(lnk) & 0x01));
  if get_tag(var) != ERA {
    link(heap, get_loc(var, 0), val);
  } else {
    collect(heap, stat, info, val);
  }
}

// Atomically replaces a ptr by another. Updates binders.
pub fn atomic_relink(heap: &Heap, loc: u64, old: Ptr, neo: Ptr) -> Result<Ptr, Ptr> {
  unsafe {
    let got = heap.node.get_unchecked(loc as usize).compare_exchange_weak(old, neo, Ordering::Relaxed, Ordering::Relaxed)?;
    if get_tag(neo) <= VAR {
      let arg_loc = get_loc(neo, get_tag(neo) & 0x01);
      heap.node.get_unchecked(arg_loc as usize).store(Arg(loc), Ordering::Relaxed);
    }
    return Ok(got);
  }
}

pub fn atomic_subst(heap: &Heap, stat: &mut Stat, info: &Info, var: Ptr, val: Ptr) {
  loop {
    let arg_loc = get_loc(var, get_tag(var) & 0x01);
    let arg_ptr = load_ptr(heap, arg_loc);
    if get_tag(arg_ptr) != ERA {
      match atomic_relink(heap, get_loc(arg_ptr, 0), var, val) {
        Ok(_) => { return; }
        Err(_) => { continue; }
      }
    } else {
      collect(heap, stat, info, val); // safe, since `val` is owned by this thread
      return;
    }
  }
}

// Garbage Collection
// ------------------

pub fn collect(heap: &Heap, stat: &mut Stat, info: &Info, term: Ptr) {
  let mut stack: Vec<Ptr> = Vec::new();
  let mut next = term;
  //let mut dups : Vec<u64> = Vec::new();
  loop {
    let term = next;
    match get_tag(term) {
      DP0 => {
        link(heap, get_loc(term, 0), Era());
        //atomic_subst(heap, stat, info, term, Era());
        //link(heap, get_loc(term, 0), Era());
        //dups.push(term);
      }
      DP1 => {
        link(heap, get_loc(term, 1), Era());
        //atomic_subst(heap, stat, info, term, Era());
        //link(heap, get_loc(term, 1), Era());
        //dups.push(term);
      }
      VAR => {
        link(heap, get_loc(term, 0), Era());
      }
      LAM => {
        //atomic_subst(heap, stat, info, Var(get_loc(term,0)), Era());
        next = take_field(heap, term, 1);
        free(heap, stat, get_loc(term, 0), 2);
        continue;
      }
      APP => {
        stack.push(take_field(heap, term, 0));
        next = take_field(heap, term, 1);
        free(heap, stat, get_loc(term, 0), 2);
        continue;
      }
      SUP => {
        stack.push(take_field(heap, term, 0));
        next = take_field(heap, term, 1);
        free(heap, stat, get_loc(term, 0), 2);
        continue;
      }
      OP2 => {
        stack.push(take_field(heap, term, 0));
        next = take_field(heap, term, 1);
        free(heap, stat, get_loc(term, 0), 2);
        continue;
      }
      NUM => {}
      CTR | FUN => {
        let arity = ask_ari(info, term);
        for i in 0 .. arity {
          if i < arity - 1 {
            stack.push(take_field(heap, term, i));
          } else {
            next = take_field(heap, term, i);
          }
        }
        free(heap, stat, get_loc(term, 0), arity);
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
    //let fst = ask_arg(rt, dup, 0);
    //let snd = ask_arg(rt, dup, 1);
    //if get_tag(fst) == ERA && get_tag(snd) == ERA {
      //collect(rt, ask_arg(rt, dup, 2));
      //free(heap, rt, get_loc(dup, 0), 3);
    //}
  //}
}

pub fn alloc_body(heap: &Heap, stat: &mut Stat, term: Ptr, vars: &[RuleVar], body: &RuleBody) -> Ptr {
  let (elem, nodes, dupk) = body;
  fn elem_to_lnk(
    heap: &Heap,
    stat: &mut Stat,
    term: Ptr,
    vars: &[RuleVar],
    elem: &RuleBodyCell,
  ) -> Ptr {
    match elem {
      RuleBodyCell::Fix { value } => {
        *value
      },
      RuleBodyCell::Ext { index } => {
        get_var(heap, term, &vars[*index as usize])
      },
      RuleBodyCell::Loc { value, targ, slot } => {
        let mut val = value + stat.abuf[*targ as usize] + slot;
        // should be changed if the pointer format changes
        if get_tag(*value) == DP0 {
          val += (stat.dups & 0xFFFFFF) * EXT;
        }
        if get_tag(*value) == DP1 {
          val += (stat.dups & 0xFFFFFF) * EXT;
        }
        val
      }
    }
  }
  for i in 0 .. nodes.len() {
    stat.abuf[i] = alloc(stat, nodes[i].len() as u64);
  };
  for i in 0 .. nodes.len() {
    let host = stat.abuf[i] as usize;
    for j in 0 .. nodes[i].len() {
      let elem = &nodes[i][j];
      let lnk = elem_to_lnk(heap, stat, term, vars, elem);
      if let RuleBodyCell::Ext { .. } = elem {
        link(heap, (host + j) as u64, lnk);
      } else {
        heap.node[host + j].store(lnk, Ordering::Relaxed);
      }
    };
  };
  let done = elem_to_lnk(heap, stat, term, vars, elem);
  stat.dups += dupk;
  return done;
}

pub fn get_var(heap: &Heap, term: Ptr, var: &RuleVar) -> Ptr {
  let RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => take_field(heap, load_field(heap, term, *param), *i),
    None    => take_field(heap, term, *param),
  }
}

pub fn fun_sup(heap: &Heap, stat: &mut Stat, info: &Info, host: u64, term: Ptr, argn: Ptr, n: u64) -> Ptr {
  inc_cost(stat);
  let arit = ask_ari(info, term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(stat, arit);
  let par0 = get_loc(argn, 0);
  for i in 0..arit {
    if i != n {
      let leti = alloc(stat, 3);
      let argi = take_field(heap, term, i);
      link(heap, fun0 + i, Dp0(get_ext(argn), leti));
      link(heap, fun1 + i, Dp1(get_ext(argn), leti));
      link(heap, leti + 2, argi);
    } else {
      link(heap, fun0 + i, take_field(heap, argn, 0));
      link(heap, fun1 + i, take_field(heap, argn, 1));
    }
  }
  link(heap, par0 + 0, Fun(arit, func, fun0));
  link(heap, par0 + 1, Fun(arit, func, fun1));
  let done = Par(get_ext(argn), par0);
  link(heap, host, done);
  done
}

// Tasks
// -----

const TASK_NONE : u64 = 0; // none      : not a task
const TASK_INIT : u64 = 1; // init(loc) : initializer task, gets a node's children
const TASK_WORK : u64 = 2; // comp(loc) : computer task, applies rewrite rules
const TASK_WAIT : u64 = 3; // wait(bid) : stolen task. must wait for its completion
const TASK_BACK : u64 = 4; // back(bid) : gives a task back to its owner once complete

fn show_task(task: u64) -> String {
  let tag = get_task_tag(task);
  let val = get_task_val(task);
  match tag {
    TASK_NONE => format!("none"),
    TASK_INIT => format!("init({})", val),
    TASK_WORK => format!("comp({})", val),
    TASK_WAIT => format!("wait({})", val),
    TASK_BACK => format!("back({})", val),
    _         => panic!("task?({:x})", task),
  }
}

fn new_task(tag: u64, val: u64) -> u64 {
  return (tag << 32) | val;
}

fn get_task_tag(task: u64) -> u64 {
  return task >> 32;
}

fn get_task_val(task: u64) -> u64 {
  return task & 0xFFFF_FFFF;
}

// Buckets
// -------

const BUCKET_EMPTY         : u64 = 0; // empty bucket
const BUCKET_HAS_TASK      : u64 = 1; // bucket with a task to steal; holds task loc
const BUCKET_TASK_STOLEN   : u64 = 2; // bucket task has been stolen
const BUCKET_TASK_COMPLETE : u64 = 3; // bucket tas was completed

const BUCKET_COUNT : usize = 65536;

fn new_bucket(tag: u64, val: u64) -> u64 {
  return (tag << 60) | val;
}

fn get_bucket_tag(bucket: u64) -> u64 {
  return bucket >> 60;
}

fn get_bucket_val(bucket: u64) -> u64 {
  return bucket & 0xFFF_FFFF_FFFF_FFFF;
}

pub fn reducer<const PARALLEL: bool>(heap: &Heap, stats: &mut Stats, info: &Info, root: u64) -> Ptr {
  println!("reduce root={} threads={} parallel={}", root, stats.len(), PARALLEL);

  #[derive(PartialEq, Clone, Copy, Debug)]
  enum Mode {
    Working,
    Getting,
    Sharing,
    Robbing,
  }

  let threads = stats.len();
  let buckets = &new_atomic_u64_array(if PARALLEL { BUCKET_COUNT } else { 0 });
  let halters = &AtomicU64::new(0);
  let fst_tid = stats[0].tid;

  let is_local = &new_atomic_u8_array(if PARALLEL { 256 } else { 0 });
  if PARALLEL {
    for stat in stats.into_iter() {
      is_local[stat.tid].store(1, Ordering::Relaxed);
    }
  }

  std::thread::scope(|s| {

    for stat in stats.iter_mut() {

      s.spawn(move || {
        let tid = stat.tid;

        println!("[{}] spawn", tid);

        let mut stack = vec![];
        let mut locks = 0;

        let mut init = true;
        let mut host = if !PARALLEL || tid == fst_tid { root } else { 0 };
        let mut mode = if !PARALLEL || tid == fst_tid { Mode::Working } else { Mode::Robbing };
        let mut tick = 0;
        let mut sidx = 0;
        let mut halt = false;

        'main: loop {

          // FIXME: improve halting condition logic / ugly code
          if PARALLEL {
            if !halt && mode == Mode::Robbing && stack.len() == 0 {
              halt = true;
              halters.fetch_add(1, Ordering::Relaxed);
            }
            if halt && mode == Mode::Working {
              halt = false;
              halters.fetch_sub(1, Ordering::Relaxed);
            }
            if halt && halters.load(Ordering::Relaxed) == threads as u64 {
              break;
            }
          }

          match mode {
            Mode::Working => {
              if PARALLEL {
                tick += 1;
                if tick > 65536 {
                  mode = Mode::Sharing;
                  tick = 0;
                  continue;
                }
              }

              let term = load_ptr(heap, host);

              if term == 0 {
                println!("found 0");
                mode = Mode::Getting;
                continue;
              }

              //println!("[{}] reduce:\n{}\n", tid, show_term(heap, info, ask_lnk(heap, 0), term));
              //validate_heap(heap);

              if init {
                match get_tag(term) {
                  APP => {
                    stack.push(new_task(TASK_WORK, host));
                    init = true;
                    host = get_loc(term, 0);
                    continue;
                  }
                  DP0 | DP1 => {
                    let got_lock = heap.lock[get_loc(term, 0) as usize].compare_exchange_weak(0xFF, stat.tid as u8, Ordering::Acquire, Ordering::Relaxed);
                    if let Err(locker_tid) = got_lock {
                      if locker_tid != stat.tid as u8 {
                        if PARALLEL && is_local[locker_tid as usize].load(Ordering::Relaxed) == 1 {
                          loop {
                            let bid = fastrand::usize(..) % BUCKET_COUNT;
                            let old = new_bucket(BUCKET_EMPTY, 0);
                            let neo = new_bucket(BUCKET_HAS_TASK, new_task(TASK_INIT, host));
                            if let Ok(old) = buckets[bid].compare_exchange(old, neo, Ordering::Relaxed, Ordering::Relaxed) {
                              mode = Mode::Robbing;
                              continue 'main;
                            }
                          }
                        } else {
                          continue 'main;
                        }
                      }
                    }
                    let new_term = load_ptr(heap, host);
                    if term != new_term {
                      heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                      //println!("[{}] DEU RUIM", stat.tid);
                      continue 'main;
                    }
                    locks = locks + 1;
                    //println!("[{}] locks {}", stat.tid, get_loc(term, 0));
                    stack.push(new_task(TASK_WORK, host));
                    host = get_loc(term, 2);
                    continue;
                  }
                  OP2 => {
                    stack.push(new_task(TASK_WORK, host));
                    stack.push(new_task(TASK_INIT, get_loc(term, 1)));
                    host = get_loc(term, 0);
                    continue;
                  }
                  FUN => {
                    let fid = get_ext(term);
                    match fid {
                      HVM_LOG => {
                        init = false;
                        continue;
                      }
                      HVM_PUT => {
                        init = false;
                        continue;
                      }
//GENERATED-FUN-CTR-MATCH//
                      _ => {
                        // Dynamic functions
                        if let Some(Some(f)) = &info.funs.get(fid as usize) {
                          let len = f.stricts.len() as u64;
                          if len == 0 {
                            init = false;
                          } else {
                            stack.push(new_task(TASK_WORK, host));
                            for (i, strict) in f.stricts.iter().enumerate() {
                              if i < f.stricts.len() - 1 {
                                stack.push(new_task(TASK_INIT, get_loc(term, *strict)));
                              } else {
                                host = get_loc(term, *strict);
                              }
                            }
                          }
                          continue;
                        }
                      }
                    }
                  }
                  _ => {}
                }
              } else {
                match get_tag(term) {
                  APP => {
                    let arg0 = load_field(heap, term, 0);

                    // (λx(body) a)
                    // ------------ APP-LAM
                    // x <- a
                    // body
                    if get_tag(arg0) == LAM {
                      //println!("app-lam");
                      inc_cost(stat);
                      atomic_subst(heap, stat, info, Var(get_loc(arg0, 0)), take_field(heap, term, 1));
                      link(heap, host, take_field(heap, arg0, 1));
                      free(heap, stat, get_loc(term, 0), 2);
                      free(heap, stat, get_loc(arg0, 0), 2);
                      init = true;
                      continue;
                    }

                    // ({a b} c)
                    // --------------- APP-SUP
                    // dup x0 x1 = c
                    // {(a x0) (b x1)}
                    if get_tag(arg0) == SUP {
                      //println!("app-sup");
                      inc_cost(stat);
                      let app0 = get_loc(term, 0);
                      let app1 = get_loc(arg0, 0);
                      let let0 = alloc(stat, 3);
                      let par0 = alloc(stat, 2);
                      link(heap, let0 + 2, take_field(heap, term, 1));
                      link(heap, app0 + 1, Dp0(get_ext(arg0), let0));
                      link(heap, app0 + 0, take_field(heap, arg0, 0));
                      link(heap, app1 + 0, take_field(heap, arg0, 1));
                      link(heap, app1 + 1, Dp1(get_ext(arg0), let0));
                      link(heap, par0 + 0, App(app0));
                      link(heap, par0 + 1, App(app1));
                      let done = Par(get_ext(arg0), par0);
                      link(heap, host, done);
                    }
                  }
                  DP0 | DP1 => {
                    //println!("[{}] on-dup {}", stat.tid, heap.lock[get_loc(term, 0) as usize].load(Ordering::Relaxed));

                    let arg0 = load_field(heap, term, 2);
                    let tcol = get_ext(term);

                    //println!("[{}] dups {}", stat.tid, get_loc(term, 0));

                    // dup r s = λx(f)
                    // --------------- DUP-LAM
                    // dup f0 f1 = f
                    // r <- λx0(f0)
                    // s <- λx1(f1)
                    // x <- {x0 x1}
                    if get_tag(arg0) == LAM {
                      //println!("dup-lam");
                      inc_cost(stat);
                      let let0 = alloc(stat, 3);
                      let par0 = alloc(stat, 2);
                      let lam0 = alloc(stat, 2);
                      let lam1 = alloc(stat, 2);
                      link(heap, let0 + 2, take_field(heap, arg0, 1));
                      link(heap, par0 + 1, Var(lam1));

                      link(heap, par0 + 0, Var(lam0));
                      link(heap, lam0 + 1, Dp0(get_ext(term), let0));
                      link(heap, lam1 + 1, Dp1(get_ext(term), let0));

                      atomic_subst(heap, stat, info, Var(get_loc(arg0, 0)), Par(get_ext(term), par0));
                      atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Lam(lam0));
                      atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Lam(lam1));

                      let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
                      link(heap, host, done);

                      free(heap, stat, get_loc(term, 0), 3);
                      free(heap, stat, get_loc(arg0, 0), 2);

                      //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                      heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                      locks = locks - 1;

                      init = true;
                      continue;


                    // dup x y = {a b}
                    // --------------- DUP-SUP (equal)
                    // x <- a
                    // y <- b
                    //
                    // dup x y = {a b}
                    // ----------------- DUP-SUP (different)
                    // x <- {xA xB}
                    // y <- {yA yB}
                    // dup xA yA = a
                    // dup xB yB = b
                    } else if get_tag(arg0) == SUP {
                      //println!("dup-sup {}", tcol == get_ext(arg0));
                      if tcol == get_ext(arg0) {
                        inc_cost(stat);
                        //println!("A");
                        atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), take_field(heap, arg0, 0));
                        //println!("B");
                        atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), take_field(heap, arg0, 1));
                        //println!("C");
                        //link(heap, host, take_field(heap, arg0, if get_tag(term) == DP0 { 0 } else { 1 })); // <- FIXME: WTF lol
                        free(heap, stat, get_loc(term, 0), 3);
                        free(heap, stat, get_loc(arg0, 0), 2);
                        //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                        locks = locks - 1;
                        heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                        init = true;
                        continue;
                      } else {
                        inc_cost(stat);
                        let par0 = alloc(stat, 2);
                        let let0 = alloc(stat, 3);
                        let par1 = get_loc(arg0, 0);
                        let let1 = alloc(stat, 3);
                        link(heap, let0 + 2, take_field(heap, arg0, 0));
                        link(heap, let1 + 2, take_field(heap, arg0, 1));
                        link(heap, par1 + 0, Dp1(tcol, let0));
                        link(heap, par1 + 1, Dp1(tcol, let1));
                        link(heap, par0 + 0, Dp0(tcol, let0));
                        link(heap, par0 + 1, Dp0(tcol, let1));

                        //println!("A");
                        atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Par(get_ext(arg0), par0));
                        atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Par(get_ext(arg0), par1));
                        //println!("B");

                        free(heap, stat, get_loc(term, 0), 3);

                        //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                        locks = locks - 1;
                        heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);

                        init = true;
                        continue;

                        //let done = Par(get_ext(arg0), if get_tag(term) == DP0 { par0 } else { par1 });
                        //link(heap, host, done);
                      }

                    // dup x y = N
                    // ----------- DUP-NUM
                    // x <- N
                    // y <- N
                    // ~
                    } else if get_tag(arg0) == NUM {
                      //println!("dup-u32");
                      inc_cost(stat);
                      atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), arg0);
                      atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), arg0);
                      free(heap, stat, get_loc(term, 0), 3);

                      //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                      locks = locks - 1;
                      heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);

                      init = true;
                      continue;
                      //link(heap, host, arg0);

                    // dup x y = (K a b c ...)
                    // ----------------------- DUP-CTR
                    // dup a0 a1 = a
                    // dup b0 b1 = b
                    // dup c0 c1 = c
                    // ...
                    // x <- (K a0 b0 c0 ...)
                    // y <- (K a1 b1 c1 ...)
                    } else if get_tag(arg0) == CTR {
                      //println!("dup-ctr");
                      inc_cost(stat);
                      let fnid = get_ext(arg0);
                      let arit = ask_ari(info, arg0);
                      if arit == 0 {
                        atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Ctr(0, fnid, 0));
                        atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Ctr(0, fnid, 0));
                        link(heap, host, Ctr(0, fnid, 0));
                        free(heap, stat, get_loc(term, 0), 3);
                      } else {
                        let ctr0 = get_loc(arg0, 0);
                        let ctr1 = alloc(stat, arit);
                        for i in 0 .. arit - 1 {
                          let leti = alloc(stat, 3);
                          link(heap, leti + 2, take_field(heap, arg0, i));
                          link(heap, ctr0 + i, Dp0(get_ext(term), leti));
                          link(heap, ctr1 + i, Dp1(get_ext(term), leti));
                        }
                        let leti = alloc(stat, 3);
                        link(heap, leti + 2, take_field(heap, arg0, arit - 1));
                        link(heap, ctr0 + arit - 1, Dp0(get_ext(term), leti));
                        link(heap, ctr1 + arit - 1, Dp1(get_ext(term), leti));
                        atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Ctr(arit, fnid, ctr0));
                        atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Ctr(arit, fnid, ctr1));
                        //let done = Ctr(arit, fnid, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
                        //link(heap, host, done);
                        free(heap, stat, get_loc(term, 0), 3);
                      }
                      //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                      heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                      locks = locks - 1;
                      init = true;
                      continue;

                    // dup x y = *
                    // ----------- DUP-ERA
                    // x <- *
                    // y <- *
                    } else if get_tag(arg0) == ERA {
                      inc_cost(stat);
                      atomic_subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Era());
                      atomic_subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Era());
                      link(heap, host, Era());
                      free(heap, stat, get_loc(term, 0), 3);
                      //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                      heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                      locks = locks - 1;
                      init = true;
                      continue;
                    }

                    //println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
                    locks = locks - 1;
                    heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
                  }
                  OP2 => {
                    let arg0 = load_field(heap, term, 0);
                    let arg1 = load_field(heap, term, 1);

                    // (+ a b)
                    // --------- OP2-NUM
                    // add(a, b)
                    if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
                      //println!("op2-u32");
                      inc_cost(stat);
                      let a = get_num(arg0);
                      let b = get_num(arg1);
                      let c = match get_ext(term) {
                        ADD => a.wrapping_add(b) & NUM_MASK,
                        SUB => a.wrapping_sub(b) & NUM_MASK,
                        MUL => a.wrapping_mul(b) & NUM_MASK,
                        DIV => a.wrapping_div(b) & NUM_MASK,
                        MOD => a.wrapping_rem(b) & NUM_MASK,
                        AND => (a & b) & NUM_MASK,
                        OR  => (a | b) & NUM_MASK,
                        XOR => (a ^ b) & NUM_MASK,
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
                      link(heap, host, done);
                      free(heap, stat, get_loc(term, 0), 2);

                    // (+ {a0 a1} b)
                    // --------------------- OP2-SUP-0
                    // dup b0 b1 = b
                    // {(+ a0 b0) (+ a1 b1)}
                    } else if get_tag(arg0) == SUP {
                      //println!("op2-sup-0");
                      inc_cost(stat);
                      let op20 = get_loc(term, 0);
                      let op21 = get_loc(arg0, 0);
                      let let0 = alloc(stat, 3);
                      let par0 = alloc(stat, 2);
                      link(heap, let0 + 2, arg1);
                      link(heap, op20 + 1, Dp0(get_ext(arg0), let0));
                      link(heap, op20 + 0, take_field(heap, arg0, 0));
                      link(heap, op21 + 0, take_field(heap, arg0, 1));
                      link(heap, op21 + 1, Dp1(get_ext(arg0), let0));
                      link(heap, par0 + 0, Op2(get_ext(term), op20));
                      link(heap, par0 + 1, Op2(get_ext(term), op21));
                      let done = Par(get_ext(arg0), par0);
                      link(heap, host, done);

                    // (+ a {b0 b1})
                    // --------------- OP2-SUP-1
                    // dup a0 a1 = a
                    // {(+ a0 b0) (+ a1 b1)}
                    } else if get_tag(arg1) == SUP {
                      //println!("op2-sup-1");
                      inc_cost(stat);
                      let op20 = get_loc(term, 0);
                      let op21 = get_loc(arg1, 0);
                      let let0 = alloc(stat, 3);
                      let par0 = alloc(stat, 2);
                      link(heap, let0 + 2, arg0);
                      link(heap, op20 + 0, Dp0(get_ext(arg1), let0));
                      link(heap, op20 + 1, take_field(heap, arg1, 0));
                      link(heap, op21 + 1, take_field(heap, arg1, 1));
                      link(heap, op21 + 0, Dp1(get_ext(arg1), let0));
                      link(heap, par0 + 0, Op2(get_ext(term), op20));
                      link(heap, par0 + 1, Op2(get_ext(term), op21));
                      let done = Par(get_ext(arg1), par0);
                      link(heap, host, done);
                    }
                  }
                  FUN => {
                    let fid = get_ext(term);
                    match fid {
                      //HVM_LOG => {
                        //let msge = get_loc(term,0);
                        //normalize(heap, stats, info, msge);
                        //println!("{}", crate::readback::as_code(heap, stats, info, msge));
                        //link(heap, host, load_field(heap, term, 1));
                        //free(heap, stat, get_loc(term, 0), 2);
                        //collect(heap, stat, info, ask_lnk(heap, msge));
                        //init = true;
                        //continue;
                      //}
                      //HVM_PUT => {
                        //let msge = get_loc(term,0);
                        //normalize(heap, stats, info, msge);
                        //let code = crate::readback::as_code(heap, stats, info, msge);
                        //if code.chars().nth(0) == Some('"') {
                          //println!("{}", &code[1 .. code.len() - 1]);
                        //} else {
                          //println!("{}", code);
                        //}
                        //link(heap, host, load_field(heap, term, 1));
                        //free(heap, stat, get_loc(term, 0), 2);
                        //collect(heap, stat, info, ask_lnk(heap, msge));
                        //init = true;
                        //continue;
                      //}
//GENERATED-FUN-CTR-RULES//
                      
                      _ => {
                        // Dynamic functions
                        if let Some(Some(function)) = &info.funs.get(fid as usize) {
                          // Reduces function superpositions
                          for (n, is_strict) in function.is_strict.iter().enumerate() {
                            let n = n as u64;
                            if *is_strict && get_tag(load_field(heap, term, n)) == SUP {
                              fun_sup(heap, stat, info, host, term, load_field(heap, term, n), n);
                              continue 'main;
                            }
                          }

                          // For each rule condition vector
                          let mut matched = false;
                          for (r, rule) in function.rules.iter().enumerate() {
                            // Check if the rule matches
                            matched = true;
                            
                            // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
                            for (i, cond) in rule.cond.iter().enumerate() {
                              let i = i as u64;
                              match get_tag(*cond) {
                                NUM => {
                                  //println!("Didn't match because of NUM. i={} {} {}", i, get_num(ask_arg(heap, term, i)), get_num(*cond));
                                  let same_tag = get_tag(load_field(heap, term, i)) == NUM;
                                  let same_val = get_num(load_field(heap, term, i)) == get_num(*cond);
                                  matched = matched && same_tag && same_val;
                                }
                                CTR => {
                                  //println!("Didn't match because of CTR. i={} {} {}", i, get_tag(ask_arg(heap, term, i)), get_ext(*cond));
                                  let same_tag = get_tag(load_field(heap, term, i)) == CTR;
                                  let same_ext = get_ext(load_field(heap, term, i)) == get_ext(*cond);
                                  matched = matched && same_tag && same_ext;
                                }
                                VAR => {
                                  // If this is a strict argument, then we're in a default variable
                                  if function.is_strict[i as usize] {

                                    // This is a Kind2-specific optimization. Check 'HOAS_OPT'.
                                    if rule.hoas && r != function.rules.len() - 1 {

                                      // Matches number literals
                                      let is_num
                                        = get_tag(load_field(heap, term, i)) == NUM;

                                      // Matches constructor labels
                                      let is_ctr
                                        =  get_tag(load_field(heap, term, i)) == CTR
                                        && ask_ari(info, load_field(heap, term, i)) == 0;

                                      // Matches HOAS numbers and constructors
                                      let is_hoas_ctr_num
                                        =  get_tag(load_field(heap, term, i)) == CTR
                                        && get_ext(load_field(heap, term, i)) >= HOAS_CT0
                                        && get_ext(load_field(heap, term, i)) <= HOAS_NUM;

                                      matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

                                    // Only match default variables on CTRs and NUMs
                                    } else {
                                      let is_ctr = get_tag(load_field(heap, term, i)) == CTR;
                                      let is_num = get_tag(load_field(heap, term, i)) == NUM;
                                      matched = matched && (is_ctr || is_num);
                                    }
                                  }
                                }
                                _ => {}
                              }
                            }

                            // If all conditions are satisfied, the rule matched, so we must apply it
                            if matched {
                              //println!("fun-ctr {:?}", info.nams.get(&get_ext(term)));

                              // Increments the gas count
                              inc_cost(stat);

                              // Builds the right-hand side term
                              let done = alloc_body(heap, stat, term, &rule.vars, &rule.body);

                              // Links the host location to it
                              link(heap, host, done);

                              // Collects unused variables
                              for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
                                if *erase {
                                  collect(heap, stat, info, get_var(heap, term, var));
                                }
                              }

                              // free the matched ctrs
                              for (i, arity) in &rule.free {
                                let i = *i as u64;
                                free(heap, stat, get_loc(load_field(heap, term, i), 0), *arity);
                              }
                              free(heap, stat, get_loc(term, 0), function.arity);

                              break;
                            }
                          }
                          if matched {
                            init = true;
                            continue;
                          }
                        }
                      }
                    }
                  }
                  _ => {}
                }
              }

              mode = Mode::Getting;
              continue;
            }
            Mode::Getting => {
              if let Some(task) = stack.pop() {
                let task_tag = get_task_tag(task);
                let task_val = get_task_val(task);
                if PARALLEL && task_tag == TASK_WAIT {
                  // If the task was completed, pop it and get another task
                  if let Ok(old) = buckets[task_val as usize].compare_exchange(new_bucket(BUCKET_TASK_COMPLETE,0), new_bucket(BUCKET_EMPTY,0), Ordering::Relaxed, Ordering::Relaxed) {
                    //println!("[{}] received a stolen task on bucket {} (cost {})", tid, task_val, stat.cost);
                    continue;
                  // Otherwise, put it back and enter steal mode
                  } else {
                    stack.push(task);
                    mode = Mode::Robbing;
                    continue;
                  }
                } else if PARALLEL && task_tag == TASK_BACK {
                  //println!("[{}] completed a stolen task on bucket {} (cost {})", tid, task_val, stat.cost);
                  match buckets[task_val as usize].compare_exchange(new_bucket(BUCKET_TASK_STOLEN,0), new_bucket(BUCKET_TASK_COMPLETE,0), Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(old) => {
                      continue;
                    }
                    Err(old) => {
                      panic!("[{}] failed to return task on bucket {}. Found: {}.", tid, task_val, old >> 60);
                    }
                  }
                } else {
                  init = task_tag == TASK_INIT;
                  host = task_val;
                  sidx = std::cmp::min(sidx, stack.len());
                  mode = Mode::Working;
                  continue;
                }
              } else {
                if PARALLEL {
                  mode = Mode::Robbing;
                  continue;
                } else {
                  break;
                }
              }
            }
            Mode::Sharing => {
              //println!("[{}] sharing {} {}", tid, sidx, stack.len());
              while sidx < stack.len() {
                let task = unsafe { *stack.get_unchecked_mut(sidx) };
                let task_tag = get_task_tag(task);
                let task_val = get_task_val(task);
                if task_tag == TASK_INIT {
                  let bid = fastrand::usize(..) % BUCKET_COUNT;
                  if let Ok(old) = buckets[bid].compare_exchange(new_bucket(BUCKET_EMPTY,0), new_bucket(BUCKET_HAS_TASK,task), Ordering::Relaxed, Ordering::Relaxed) {
                    //println!("[{}] shared task on bucket {}", tid, bid);
                    let task_ref = unsafe { stack.get_unchecked_mut(sidx) };
                    *task_ref = new_task(TASK_WAIT, bid as u64);
                  }
                  break;
                }
                sidx = sidx + 1;
              }
              mode = Mode::Working;
              continue;
            }
            Mode::Robbing => {
              // Try to steal a task from another thread
              let bid = fastrand::usize(..) % BUCKET_COUNT;
              let buk = buckets[bid].load(Ordering::Relaxed);
              if get_bucket_tag(buk) == BUCKET_HAS_TASK {
                if let Ok(old) = buckets[bid].compare_exchange(buk, new_bucket(BUCKET_TASK_STOLEN,0), Ordering::Relaxed, Ordering::Relaxed) {
                  //println!("[{}] stolen task on bucket {} (cost {})", tid, bid, stat.cost);
                  stack.push(new_task(TASK_BACK, bid as u64));
                  let task = get_bucket_val(buk);
                  init = get_task_tag(task) == TASK_INIT;
                  host = get_task_val(task);
                  mode = Mode::Working;
                  continue;
                //} else {
                  //println!("[{}] couldn't steal task on bucket {}", tid, bid);
                }
              }
              mode = Mode::Getting;
              continue;
            }
          }
        }
        println!("[{}] halt (cost: {}, locks: {}, stack: {})", stat.tid, stat.cost, locks, stack.len());
      });
    }

  });

  load_ptr(heap, root)
}

pub fn reduce(heap: &Heap, stats: &mut Stats, info: &Info, root: u64) -> Ptr {
  reducer::<true>(heap, stats, info, root)
}

//pub fn reduce1(heap: &Heap, stats: &mut Stats, info: &Info, root: u64) -> Ptr {
  //println!("reduce root={} 1", root);
  //let stat = &mut stats[0];
  //let mut stack = vec![];
  //let mut init = true;
  //let mut host = root;
  //'main: loop {
    //let term = load_ptr(heap, host);
    ////println!("[{}] reduce:\n{}\n", tid, show_term(heap, info, ask_lnk(heap, 0), term));
    //if init {
      //match get_tag(term) {
        //APP => {
          //stack.push(new_task(TASK_WORK, host));
          //init = true;
          //host = get_loc(term, 0);
          //continue;
        //}
        //DP0 | DP1 => {
          ////let got_lock = heap.lock[get_loc(term, 0) as usize].compare_exchange(0xFF, stat.tid as u8, Ordering::Acquire, Ordering::Acquire);
          ////if let Err(old) = got_lock {
            ////panic!("todo");
          ////}
          //stack.push(new_task(TASK_WORK, host));
          //host = get_loc(term, 2);
          //continue;
        //}
        //OP2 => {
          //stack.push(new_task(TASK_WORK, host));
          //stack.push(new_task(TASK_INIT, get_loc(term, 1)));
          //host = get_loc(term, 0);
          //continue;
        //}
        //FUN => {
          //let fid = get_ext(term);
          //match fid {
            //HVM_LOG => {
              //init = false;
              //continue;
            //}
            //HVM_PUT => {
              //init = false;
              //continue;
            //}
////GENERATED-FUN-CTR-MATCH//
            //_ => {
              //// Dynamic functions
              //if let Some(Some(f)) = &info.funs.get(fid as usize) {
                //let len = f.stricts.len() as u64;
                //if len == 0 {
                  //init = false;
                //} else {
                  //stack.push(new_task(TASK_WORK, host));
                  //for (i, strict) in f.stricts.iter().enumerate() {
                    //if i < f.stricts.len() - 1 {
                      //stack.push(new_task(TASK_INIT, get_loc(term, *strict)));
                    //} else {
                      //host = get_loc(term, *strict);
                    //}
                  //}
                //}
                //continue;
              //}
            //}
          //}
        //}
        //_ => {}
      //}
    //} else {
      //match get_tag(term) {
        //APP => {
          //let arg0 = load_field(heap, term, 0);

          //// (λx(body) a)
          //// ------------ APP-LAM
          //// x <- a
          //// body
          //if get_tag(arg0) == LAM {
            ////println!("app-lam");
            //inc_cost(stat);
            //subst(heap, stat, info, Var(get_loc(arg0, 0)), take_field(heap, term, 1));
            //link(heap, host, take_field(heap, arg0, 1));
            //free(heap, stat, get_loc(term, 0), 2);
            //free(heap, stat, get_loc(arg0, 0), 2);
            //init = true;
            //continue;
          //}

          //// ({a b} c)
          //// --------------- APP-SUP
          //// dup x0 x1 = c
          //// {(a x0) (b x1)}
          //if get_tag(arg0) == SUP {
            ////println!("app-sup");
            //inc_cost(stat);
            //let app0 = get_loc(term, 0);
            //let app1 = get_loc(arg0, 0);
            //let let0 = alloc(stat, 3);
            //let par0 = alloc(stat, 2);
            //link(heap, let0 + 2, take_field(heap, term, 1));
            //link(heap, app0 + 1, Dp0(get_ext(arg0), let0));
            //link(heap, app0 + 0, take_field(heap, arg0, 0));
            //link(heap, app1 + 0, take_field(heap, arg0, 1));
            //link(heap, app1 + 1, Dp1(get_ext(arg0), let0));
            //link(heap, par0 + 0, App(app0));
            //link(heap, par0 + 1, App(app1));
            //let done = Par(get_ext(arg0), par0);
            //link(heap, host, done);
          //}
        //}
        //DP0 | DP1 => {
          ////println!("[{}] on-dup {}", stat.tid, heap.lock[get_loc(term, 0) as usize].load(Ordering::Relaxed));

          //let arg0 = load_field(heap, term, 2);
          //let tcol = get_ext(term);

          ////println!("[{}] dups {}", stat.tid, get_loc(term, 0));

          //// dup r s = λx(f)
          //// --------------- DUP-LAM
          //// dup f0 f1 = f
          //// r <- λx0(f0)
          //// s <- λx1(f1)
          //// x <- {x0 x1}
          //if get_tag(arg0) == LAM {
            ////println!("dup-lam");
            //inc_cost(stat);
            //let let0 = alloc(stat, 3);
            //let par0 = alloc(stat, 2);
            //let lam0 = alloc(stat, 2);
            //let lam1 = alloc(stat, 2);
            //link(heap, let0 + 2, take_field(heap, arg0, 1));
            //link(heap, par0 + 1, Var(lam1));

            //link(heap, par0 + 0, Var(lam0));
            //link(heap, lam0 + 1, Dp0(get_ext(term), let0));
            //link(heap, lam1 + 1, Dp1(get_ext(term), let0));

            //subst(heap, stat, info, Var(get_loc(arg0, 0)), Par(get_ext(term), par0));
            //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Lam(lam0));
            //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Lam(lam1));

            //let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
            //link(heap, host, done);

            //free(heap, stat, get_loc(term, 0), 3);
            //free(heap, stat, get_loc(arg0, 0), 2);

            ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
            ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);

            //init = true;
            //continue;


          //// dup x y = {a b}
          //// --------------- DUP-SUP (equal)
          //// x <- a
          //// y <- b
          ////
          //// dup x y = {a b}
          //// ----------------- DUP-SUP (different)
          //// x <- {xA xB}
          //// y <- {yA yB}
          //// dup xA yA = a
          //// dup xB yB = b
          //} else if get_tag(arg0) == SUP {
            ////println!("dup-sup {}", tcol == get_ext(arg0));
            //if tcol == get_ext(arg0) {
              //inc_cost(stat);
              ////println!("A");
              //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), take_field(heap, arg0, 0));
              ////println!("B");
              //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), take_field(heap, arg0, 1));
              ////println!("C");
              ////link(heap, host, take_field(heap, arg0, if get_tag(term) == DP0 { 0 } else { 1 })); // <- FIXME: WTF lol
              //free(heap, stat, get_loc(term, 0), 3);
              //free(heap, stat, get_loc(arg0, 0), 2);
              ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
              ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
              //init = true;
              //continue;
            //} else {
              //inc_cost(stat);
              //let par0 = alloc(stat, 2);
              //let let0 = alloc(stat, 3);
              //let par1 = get_loc(arg0, 0);
              //let let1 = alloc(stat, 3);
              //link(heap, let0 + 2, take_field(heap, arg0, 0));
              //link(heap, let1 + 2, take_field(heap, arg0, 1));
              //link(heap, par1 + 0, Dp1(tcol, let0));
              //link(heap, par1 + 1, Dp1(tcol, let1));
              //link(heap, par0 + 0, Dp0(tcol, let0));
              //link(heap, par0 + 1, Dp0(tcol, let1));

              ////println!("A");
              //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Par(get_ext(arg0), par0));
              //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Par(get_ext(arg0), par1));
              ////println!("B");

              //free(heap, stat, get_loc(term, 0), 3);

              ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
              ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);

              //init = true;
              //continue;

              ////let done = Par(get_ext(arg0), if get_tag(term) == DP0 { par0 } else { par1 });
              ////link(heap, host, done);
            //}

          //// dup x y = N
          //// ----------- DUP-NUM
          //// x <- N
          //// y <- N
          //// ~
          //} else if get_tag(arg0) == NUM {
            ////println!("dup-u32");
            //inc_cost(stat);
            //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), arg0);
            //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), arg0);
            //free(heap, stat, get_loc(term, 0), 3);

            ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
            ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);

            //init = true;
            //continue;
            ////link(heap, host, arg0);

          //// dup x y = (K a b c ...)
          //// ----------------------- DUP-CTR
          //// dup a0 a1 = a
          //// dup b0 b1 = b
          //// dup c0 c1 = c
          //// ...
          //// x <- (K a0 b0 c0 ...)
          //// y <- (K a1 b1 c1 ...)
          //} else if get_tag(arg0) == CTR {
            ////println!("dup-ctr");
            //inc_cost(stat);
            //let fnid = get_ext(arg0);
            //let arit = ask_ari(info, arg0);
            //if arit == 0 {
              //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Ctr(0, fnid, 0));
              //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Ctr(0, fnid, 0));
              //free(heap, stat, get_loc(term, 0), 3);
            //} else {
              //let ctr0 = get_loc(arg0, 0);
              //let ctr1 = alloc(stat, arit);
              //for i in 0 .. arit - 1 {
                //let leti = alloc(stat, 3);
                //link(heap, leti + 2, take_field(heap, arg0, i));
                //link(heap, ctr0 + i, Dp0(get_ext(term), leti));
                //link(heap, ctr1 + i, Dp1(get_ext(term), leti));
              //}
              //let leti = alloc(stat, 3);
              //link(heap, leti + 2, take_field(heap, arg0, arit - 1));
              //link(heap, ctr0 + arit - 1, Dp0(get_ext(term), leti));
              //link(heap, ctr1 + arit - 1, Dp1(get_ext(term), leti));
              //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Ctr(arit, fnid, ctr0));
              //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Ctr(arit, fnid, ctr1));
              //free(heap, stat, get_loc(term, 0), 3);
            //}
            ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
            ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
            //init = true;
            //continue;

          //// dup x y = *
          //// ----------- DUP-ERA
          //// x <- *
          //// y <- *
          //} else if get_tag(arg0) == ERA {
            //inc_cost(stat);
            //subst(heap, stat, info, Dp0(tcol, get_loc(term, 0)), Era());
            //subst(heap, stat, info, Dp1(tcol, get_loc(term, 0)), Era());
            //link(heap, host, Era());
            //free(heap, stat, get_loc(term, 0), 3);
            ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
            //init = true;
            //continue;
          //}

          ////println!("[{}] unlocks {}", stat.tid, get_loc(term, 0));
          ////heap.lock[get_loc(term, 0) as usize].store(0xFF, Ordering::Release);
        //}
        //OP2 => {
          //let arg0 = load_field(heap, term, 0);
          //let arg1 = load_field(heap, term, 1);

          //// (+ a b)
          //// --------- OP2-NUM
          //// add(a, b)
          //if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
            ////println!("op2-u32");
            //inc_cost(stat);
            //let a = get_num(arg0);
            //let b = get_num(arg1);
            //let c = match get_ext(term) {
              //ADD => a.wrapping_add(b) & NUM_MASK,
              //SUB => a.wrapping_sub(b) & NUM_MASK,
              //MUL => a.wrapping_mul(b) & NUM_MASK,
              //DIV => a.wrapping_div(b) & NUM_MASK,
              //MOD => a.wrapping_rem(b) & NUM_MASK,
              //AND => (a & b) & NUM_MASK,
              //OR  => (a | b) & NUM_MASK,
              //XOR => (a ^ b) & NUM_MASK,
              //SHL => a.wrapping_shl(b as u32) & NUM_MASK,
              //SHR => a.wrapping_shr(b as u32) & NUM_MASK,
              //LTN => u64::from(a <  b),
              //LTE => u64::from(a <= b),
              //EQL => u64::from(a == b),
              //GTE => u64::from(a >= b),
              //GTN => u64::from(a >  b),
              //NEQ => u64::from(a != b),
              //_   => panic!("Invalid operation!"),
            //};
            //let done = Num(c);
            //link(heap, host, done);
            //free(heap, stat, get_loc(term, 0), 2);

          //// (+ {a0 a1} b)
          //// --------------------- OP2-SUP-0
          //// dup b0 b1 = b
          //// {(+ a0 b0) (+ a1 b1)}
          //} else if get_tag(arg0) == SUP {
            ////println!("op2-sup-0");
            //inc_cost(stat);
            //let op20 = get_loc(term, 0);
            //let op21 = get_loc(arg0, 0);
            //let let0 = alloc(stat, 3);
            //let par0 = alloc(stat, 2);
            //link(heap, let0 + 2, arg1);
            //link(heap, op20 + 1, Dp0(get_ext(arg0), let0));
            //link(heap, op20 + 0, take_field(heap, arg0, 0));
            //link(heap, op21 + 0, take_field(heap, arg0, 1));
            //link(heap, op21 + 1, Dp1(get_ext(arg0), let0));
            //link(heap, par0 + 0, Op2(get_ext(term), op20));
            //link(heap, par0 + 1, Op2(get_ext(term), op21));
            //let done = Par(get_ext(arg0), par0);
            //link(heap, host, done);

          //// (+ a {b0 b1})
          //// --------------- OP2-SUP-1
          //// dup a0 a1 = a
          //// {(+ a0 b0) (+ a1 b1)}
          //} else if get_tag(arg1) == SUP {
            ////println!("op2-sup-1");
            //inc_cost(stat);
            //let op20 = get_loc(term, 0);
            //let op21 = get_loc(arg1, 0);
            //let let0 = alloc(stat, 3);
            //let par0 = alloc(stat, 2);
            //link(heap, let0 + 2, arg0);
            //link(heap, op20 + 0, Dp0(get_ext(arg1), let0));
            //link(heap, op20 + 1, take_field(heap, arg1, 0));
            //link(heap, op21 + 1, take_field(heap, arg1, 1));
            //link(heap, op21 + 0, Dp1(get_ext(arg1), let0));
            //link(heap, par0 + 0, Op2(get_ext(term), op20));
            //link(heap, par0 + 1, Op2(get_ext(term), op21));
            //let done = Par(get_ext(arg1), par0);
            //link(heap, host, done);
          //}
        //}
        //FUN => {
          //let fid = get_ext(term);
          //match fid {
            ////HVM_LOG => {
              ////let msge = get_loc(term,0);
              ////normalize(heap, stats, info, msge);
              ////println!("{}", crate::readback::as_code(heap, stats, info, msge));
              ////link(heap, host, load_field(heap, term, 1));
              ////free(heap, stat, get_loc(term, 0), 2);
              ////collect(heap, stat, info, ask_lnk(heap, msge));
              ////init = true;
              ////continue;
            ////}
            ////HVM_PUT => {
              ////let msge = get_loc(term,0);
              ////normalize(heap, stats, info, msge);
              ////let code = crate::readback::as_code(heap, stats, info, msge);
              ////if code.chars().nth(0) == Some('"') {
                ////println!("{}", &code[1 .. code.len() - 1]);
              ////} else {
                ////println!("{}", code);
              ////}
              ////link(heap, host, load_field(heap, term, 1));
              ////free(heap, stat, get_loc(term, 0), 2);
              ////collect(heap, stat, info, ask_lnk(heap, msge));
              ////init = true;
              ////continue;
            ////}
////GENERATED-FUN-CTR-RULES//
            
            //_ => {
              //// Dynamic functions
              //if let Some(Some(function)) = &info.funs.get(fid as usize) {
                //// Reduces function superpositions
                //for (n, is_strict) in function.is_strict.iter().enumerate() {
                  //let n = n as u64;
                  //if *is_strict && get_tag(load_field(heap, term, n)) == SUP {
                    //fun_sup(heap, stat, info, host, term, load_field(heap, term, n), n);
                    //continue 'main;
                  //}
                //}

                //// For each rule condition vector
                //let mut matched = false;
                //for (r, rule) in function.rules.iter().enumerate() {
                  //// Check if the rule matches
                  //matched = true;
                  
                  //// Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
                  //for (i, cond) in rule.cond.iter().enumerate() {
                    //let i = i as u64;
                    //match get_tag(*cond) {
                      //NUM => {
                        ////println!("Didn't match because of NUM. i={} {} {}", i, get_num(ask_arg(heap, term, i)), get_num(*cond));
                        //let same_tag = get_tag(load_field(heap, term, i)) == NUM;
                        //let same_val = get_num(load_field(heap, term, i)) == get_num(*cond);
                        //matched = matched && same_tag && same_val;
                      //}
                      //CTR => {
                        ////println!("Didn't match because of CTR. i={} {} {}", i, get_tag(ask_arg(heap, term, i)), get_ext(*cond));
                        //let same_tag = get_tag(load_field(heap, term, i)) == CTR;
                        //let same_ext = get_ext(load_field(heap, term, i)) == get_ext(*cond);
                        //matched = matched && same_tag && same_ext;
                      //}
                      //VAR => {
                        //// If this is a strict argument, then we're in a default variable
                        //if function.is_strict[i as usize] {

                          //// This is a Kind2-specific optimization. Check 'HOAS_OPT'.
                          //if rule.hoas && r != function.rules.len() - 1 {

                            //// Matches number literals
                            //let is_num
                              //= get_tag(load_field(heap, term, i)) == NUM;

                            //// Matches constructor labels
                            //let is_ctr
                              //=  get_tag(load_field(heap, term, i)) == CTR
                              //&& ask_ari(info, load_field(heap, term, i)) == 0;

                            //// Matches HOAS numbers and constructors
                            //let is_hoas_ctr_num
                              //=  get_tag(load_field(heap, term, i)) == CTR
                              //&& get_ext(load_field(heap, term, i)) >= HOAS_CT0
                              //&& get_ext(load_field(heap, term, i)) <= HOAS_NUM;

                            //matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

                          //// Only match default variables on CTRs and NUMs
                          //} else {
                            //let is_ctr = get_tag(load_field(heap, term, i)) == CTR;
                            //let is_num = get_tag(load_field(heap, term, i)) == NUM;
                            //matched = matched && (is_ctr || is_num);
                          //}
                        //}
                      //}
                      //_ => {}
                    //}
                  //}

                  //// If all conditions are satisfied, the rule matched, so we must apply it
                  //if matched {
                    ////println!("fun-ctr {:?}", info.nams.get(&get_ext(term)));

                    //// Increments the gas count
                    //inc_cost(stat);

                    //// Builds the right-hand side term
                    //let done = alloc_body(heap, stat, term, &rule.vars, &rule.body);

                    //// Links the host location to it
                    //link(heap, host, done);

                    //// Collects unused variables
                    //for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
                      //if *erase {
                        //collect(heap, stat, info, get_var(heap, term, var));
                      //}
                    //}

                    //// free the matched ctrs
                    //for (i, arity) in &rule.free {
                      //let i = *i as u64;
                      //free(heap, stat, get_loc(load_field(heap, term, i), 0), *arity);
                    //}
                    //free(heap, stat, get_loc(term, 0), function.arity);

                    //break;
                  //}
                //}
                //if matched {
                  //init = true;
                  //continue;
                //}
              //}
            //}
          //}
        //}
        //_ => {}
      //}
    //}
    //if let Some(task) = stack.pop() {
      //let task_tag = get_task_tag(task);
      //let task_val = get_task_val(task);
      //init = task_tag == TASK_INIT;
      //host = task_val;
      //continue;
    //} else {
      //break;
    //}
  //}
  //load_ptr(heap, root)
//}

pub fn normal(heap: &Heap, stats: &mut Stats, info: &Info, host: u64, visited: &Box<[AtomicU64]>) -> Ptr {
  //println!("normal host={} threads={} {}", host, stats.len(), show_term(heap, info, ask_lnk(heap,host), host));
  pub fn set_visited(visited: &Box<[AtomicU64]>, bit: u64) {
    let val = &visited[bit as usize >> 6];
    val.store(val.load(Ordering::Relaxed) | (1 << (bit & 0x3f)), Ordering::Relaxed);
  }
  pub fn was_visited(visited: &Box<[AtomicU64]>, bit: u64) -> bool {
    let val = &visited[bit as usize >> 6];
    (((val.load(Ordering::Relaxed) >> (bit & 0x3f)) as u8) & 1) == 1
  }
  let term = load_ptr(heap, host);
  if was_visited(visited, host) {
    term
  } else {
    let term = match stats.len() > 1 {
      false => { reducer::<false>(heap, stats, info, host) }
      true  => { reducer::<true>(heap, stats, info, host) }
    };
    set_visited(visited, host);
    let mut rec_locs = vec![];
    match get_tag(term) {
      LAM => {
        rec_locs.push(get_loc(term, 1));
      }
      APP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      SUP => {
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
        let arity = ask_ari(info, term);
        for i in 0 .. arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    let rec_len = rec_locs.len(); // locations where we must recurse
    let thd_len = stats.len(); // number of available threads
    let rec_loc = &rec_locs;
    //println!("~ rec_len={} thd_len={} {}", rec_len, thd_len, show_term(heap, info, ask_lnk(heap,host), host));
    if rec_len > 0 {
      std::thread::scope(|s| {
        // If there are more threads than rec_locs, splits threads for each rec_loc
        if thd_len >= rec_len {
          let spt_len = thd_len / rec_len;
          let mut stats = stats;
          for (rec_num, rec_loc) in rec_loc.iter().enumerate() {
            let (rec_stats, new_stats) = stats.split_at_mut(if rec_num == rec_len - 1 { stats.len() } else { spt_len });
            let rec_info = info.clone();
            //println!("~ rec_loc {} gets {} threads", rec_loc, rec_stats.len());
            s.spawn(move || {
              let lnk = normal(heap, rec_stats, rec_info, *rec_loc, visited);
              link(heap, *rec_loc, lnk);
            });
            stats = new_stats;
          }
        // Otherwise, splits rec_locs for each thread
        } else {
          for (thd_num, stat) in stats.iter_mut().enumerate() {
            let min_idx = thd_num * rec_len / thd_len;
            let max_idx = if thd_num < thd_len - 1 { (thd_num + 1) * rec_len / thd_len } else { rec_len };
            let rec_info = info.clone();
            //println!("~ thread {} gets rec_locs {} to {}", thd_num, min_idx, max_idx);
            s.spawn(move || {
              for idx in min_idx .. max_idx {
                let loc = rec_loc[idx];
                let lnk = normal(heap, std::slice::from_mut(stat), rec_info, loc, visited);
                link(heap, loc, lnk);
              }
            });
          }
        }
      });
    }
    term
  }
}

pub fn normalize(heap: &Heap, stats: &mut Stats, info: &Info, host: u64, run_io: bool) -> Ptr {
  println!("normalize");
  // TODO: rt::run_io(&mut heap, &mut stat, &mut info, host);
  // FIXME: reuse `visited`
  let mut cost = get_cost(stats);
  let visited = new_atomic_u64_array(heap.node.len() / 64);
  loop {
    let visited = new_atomic_u64_array(heap.node.len() / 64);
    normal(heap, stats, info, host, &visited);
    let new_cost = get_cost(stats);
    if new_cost != cost {
      cost = new_cost;
    } else {
      break;
    }
  }
  //normal(heap, &mut stats[0 .. 1], info, host, &visited);
  load_ptr(heap, host)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_io(heap: &Heap, stats: &mut Stats, info: &Info, host: u64) {
  fn read_input() -> String {
    let mut input = String::new();
    stdin().read_line(&mut input).expect("string");
    if let Some('\n') = input.chars().next_back() { input.pop(); }
    if let Some('\r') = input.chars().next_back() { input.pop(); }
    return input;
  }
  use std::io::{stdin,stdout,Write};
  loop {
    println!("??? {}", stats.len());
    let term = reduce(heap, stats, info, host); // FIXME: add parallelism
    match get_tag(term) {
      CTR => {
        match get_ext(term) {
          // IO.done a : (IO a)
          IO_DONE => {
            let done = ask_arg(heap, term, 0);
            free(heap, &mut stats[0], get_loc(term, 0), 1);
            link(heap, host, done);
            println!("");
            println!("");
            break;
          }
          // IO.do_input (String -> IO a) : (IO a)
          IO_DO_INPUT => {
            let cont = ask_arg(heap, term, 0);
            let text = make_string(heap, &mut stats[0], &read_input());
            let app0 = alloc(&mut stats[0], 2);
            link(heap, app0 + 0, cont);
            link(heap, app0 + 1, text);
            free(heap, &mut stats[0], get_loc(term, 0), 1);
            let done = App(app0);
            link(heap, host, done);
          }
          // IO.do_output String (Num -> IO a) : (IO a)
          IO_DO_OUTPUT => {
            if let Some(show) = readback_string(heap, stats, info, get_loc(term, 0)) {
              print!("{}", show);
              stdout().flush().ok();
              let cont = ask_arg(heap, term, 1);
              let app0 = alloc(&mut stats[0], 2);
              link(heap, app0 + 0, cont);
              link(heap, app0 + 1, Num(0));
              free(heap, &mut stats[0], get_loc(term, 0), 2);
              let text = ask_arg(heap, term, 0);
              collect(heap, &mut stats[0], info, text);
              let done = App(app0);
              link(heap, host, done);
            } else {
              println!("Runtime type error: attempted to print a non-string.");
              println!("{}", crate::readback::as_code(heap, stats, info, get_loc(term, 0)));
              std::process::exit(0);
            }
          }
          // IO.do_fetch String (String -> IO a) : (IO a)
          IO_DO_FETCH => {
            if let Some(url) = readback_string(heap, stats, info, get_loc(term, 0)) {
              let body = reqwest::blocking::get(url).unwrap().text().unwrap(); // FIXME: treat
              let cont = ask_arg(heap, term, 2);
              let app0 = alloc(&mut stats[0], 2);
              let text = make_string(heap, &mut stats[0], &body);
              link(heap, app0 + 0, cont);
              link(heap, app0 + 1, text);
              free(heap, &mut stats[0], get_loc(term, 0), 3);
              let opts = ask_arg(heap, term, 1); // FIXME: use options
              collect(heap, &mut stats[0], info, opts);
              let done = App(app0);
              link(heap, host, done);
            } else {
              println!("Runtime type error: attempted to print a non-string.");
              println!("{}", crate::readback::as_code(heap, stats, info, get_loc(term, 0)));
              std::process::exit(0);
            }
          }
          // IO.do_store String String (Num -> IO a) : (IO a)
          IO_DO_STORE => {
            if let Some(key) = readback_string(heap, stats, info, get_loc(term, 0)) {
              if let Some(val) = readback_string(heap, stats, info, get_loc(term, 1)) {
                std::fs::write(key, val).ok(); // TODO: Handle errors
                let cont = ask_arg(heap, term, 2);
                let app0 = alloc(&mut stats[0], 2);
                link(heap, app0 + 0, cont);
                link(heap, app0 + 1, Num(0));
                free(heap, &mut stats[0], get_loc(term, 0), 2);
                let key = ask_arg(heap, term, 0);
                collect(heap, &mut stats[0], info, key);
                free(heap, &mut stats[0], get_loc(term, 1), 2);
                let val = ask_arg(heap, term, 1);
                collect(heap, &mut stats[0], info, val);
                let done = App(app0);
                link(heap, host, done);
              } else {
                println!("Runtime type error: attempted to store a non-string.");
                println!("{}", crate::readback::as_code(heap, stats, info, get_loc(term, 1)));
                std::process::exit(0);
              }
            } else {
              println!("Runtime type error: attempted to store to a non-string key.");
              println!("{}", crate::readback::as_code(heap, stats, info, get_loc(term, 0)));
              std::process::exit(0);
            }
          }
          // IO.do_load String (String -> IO a) : (IO a)
          IO_DO_LOAD => {
            if let Some(key) = readback_string(heap, stats, info, get_loc(term, 0)) {
              let file = std::fs::read(key).unwrap(); // TODO: Handle errors
              let file = std::str::from_utf8(&file).unwrap();
              let cont = ask_arg(heap, term, 1); 
              let text = make_string(heap, &mut stats[0], file);
              let app0 = alloc(&mut stats[0], 2);
              link(heap, app0 + 0, cont);
              link(heap, app0 + 1, text);
              free(heap, &mut stats[0], get_loc(term, 0), 2);
              let done = App(app0);
              link(heap, host, done);
            } else {
              println!("Runtime type error: attempted to read from a non-string key.");
              println!("{}", crate::readback::as_code(heap, stats, info, get_loc(term, 0)));
              std::process::exit(0);
            }
          }
          _ => { break; }
        }
      }
      _ => { break; }
    }
  }
}

pub fn make_string(heap: &Heap, stat: &mut Stat, text: &str) -> Ptr {
  let mut term = Ctr(0, STRING_NIL, 0);
  for chr in text.chars().rev() { // TODO: reverse
    let ctr0 = alloc(stat, 2);
    link(heap, ctr0 + 0, Num(chr as u64));
    link(heap, ctr0 + 1, term);
    term = Ctr(2, STRING_CONS, ctr0);
  }
  return term;
}

// TODO: finish this
pub fn readback_string(heap: &Heap, stats: &mut Stats, info: &Info, host: u64) -> Option<String> {
  let mut host = host;
  let mut text = String::new();
  loop {
    let term = reduce(heap, stats, info, host);
    match get_tag(term) {
      CTR => {
        match get_ext(term) {
          STRING_NIL => {
            break;
          }
          STRING_CONS => {
            let chr = reduce(heap, stats, info, get_loc(term, 0));
            if get_tag(chr) == NUM {
              text.push(std::char::from_u32(get_num(chr) as u32).unwrap_or('?'));
              host = get_loc(term, 1);
              continue;
            } else {
              return None;
            }
          }
          _ => {
            return None;
          }
        }
      }
      _ => {
        return None;
      }
    }
  }
  return Some(text);
}


// Debug
// -----

pub fn show_ptr(x: Ptr) -> String {
  if x == 0 {
    String::from("~")
  } else {
    let tag = get_tag(x);
    let ext = get_ext(x);
    let val = get_val(x);
    let tgs = match tag {
      DP0 => "Dp0",
      DP1 => "Dp1",
      VAR => "Var",
      ARG => "Arg",
      ERA => "Era",
      LAM => "Lam",
      APP => "App",
      SUP => "Sup",
      CTR => "Ctr",
      FUN => "Fun",
      OP2 => "Op2",
      NUM => "Num",
      _ => "?",
    };
    format!("{}({:07x}, {:08x})", tgs, ext, val)
  }
}


pub fn validate_heap(heap: &Heap) {
  for idx in 0 .. HEAP_SIZE {
    // If it is an ARG, it must be pointing to a VAR/DP0/DP1 that points to it
    let arg = heap.node[idx].load(Ordering::Relaxed);
    if get_tag(arg) == ARG {
      let var = load_ptr(heap, get_loc(arg, 0));
      let oks = match get_tag(var) {
        VAR => { get_loc(var, 0) == idx as u64 }
        DP0 => { get_loc(var, 0) == idx as u64 }
        DP1 => { get_loc(var, 0) == idx as u64 - 1 }
        _   => { false }
      };
      if !oks {
        panic!("Invalid heap state, due to arg at '{:04x}' of:\n{}", idx, show_heap(heap));
      }
    }
  }
}

pub fn show_heap(heap: &Heap) -> String {
  let mut text: String = String::new();
  for idx in 0 .. HEAP_SIZE {
    let ptr = heap.node[idx].load(Ordering::Relaxed);
    if ptr != 0 {
      text.push_str(&format!("{:04x} | ", idx));
      text.push_str(&show_ptr(ptr));
      text.push('\n');
    }
  }
  text
}

pub fn show_term(heap: &Heap, info: &Info, term: Ptr, focus: u64) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    heap: &Heap,
    info: &Info,
    term: Ptr,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(heap, info, load_field(heap, term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(heap, info, load_field(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, info, load_field(heap, term, 1), lets, kinds, names, count);
      }
      SUP => {
        find_lets(heap, info, load_field(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, info, load_field(heap, term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, info, load_field(heap, term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, info, load_field(heap, term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(heap, info, load_field(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, info, load_field(heap, term, 1), lets, kinds, names, count);
      }
      CTR | FUN => {
        let arity = ask_ari(info, term);
        for i in 0..arity {
          find_lets(heap, info, load_field(heap, term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    heap: &Heap,
    info: &Info,
    term: Ptr,
    names: &HashMap<u64, String>,
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
        format!("λ{} {}", name, go(heap, info, load_field(heap, term, 1), names, focus))
      }
      APP => {
        let func = go(heap, info, load_field(heap, term, 0), names, focus);
        let argm = go(heap, info, load_field(heap, term, 1), names, focus);
        format!("({} {})", func, argm)
      }
      SUP => {
        //let kind = get_ext(term);
        let func = go(heap, info, load_field(heap, term, 0), names, focus);
        let argm = go(heap, info, load_field(heap, term, 1), names, focus);
        format!("{{{} {}}}", func, argm)
      }
      OP2 => {
        let oper = get_ext(term);
        let val0 = go(heap, info, load_field(heap, term, 0), names, focus);
        let val1 = go(heap, info, load_field(heap, term, 1), names, focus);
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
      CTR | FUN => {
        let func = get_ext(term);
        let arit = ask_ari(info, term);
        let args: Vec<String> = (0..arit).map(|i| go(heap, info, load_field(heap, term, i), names, focus)).collect();
        let name = &info.nams.get(&func).unwrap_or(&String::from("<?>")).clone();
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
  find_lets(heap, info, term, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(heap, info, term, &names, focus);
  for (_key, pos) in lets {
    // todo: reverse
    let what = String::from("?h");
    //let kind = kinds.get(&pos).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 = if load_ptr(heap, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 = if load_ptr(heap, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!("\ndup[{:x}] {} {} = {};", pos, nam0, nam1, go(heap, info, load_ptr(heap, pos + 2), &names, focus)));
  }
  text
}
