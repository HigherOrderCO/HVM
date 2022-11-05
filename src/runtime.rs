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
//   Core:
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
//   Core:
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
//   Core:
//     
//     λx dup x0 x1 = x; (* x0 x1)
//
//   Memory:
//
//     Root : Ptr(LAM, 0x0000000, 0x00000000)
//     0x00 | Ptr(ARG, 0x0000000, 0x00000004) // the lambda's argument
//     0x01 | Ptr(OP2, 0x0000002, 0x00000005) // the lambda's body
//     0x02 | Ptr(ARG, 0x0000000, 0x00000005) // the duplication's 1st argument
//     0x03 | Ptr(ARG, 0x0000000, 0x00000006) // the duplication's 2nd argument
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

use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicI64, AtomicUsize, Ordering};
use crossbeam::utils::{CachePadded, Backoff};

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

// Reserved Names
// --------------

pub struct EntryInfo {
  pub id    : u64,
  pub name  : &'static str,
  pub arity : usize,
  pub is_fn : bool,
}

pub const ENTRY_INFO : &[EntryInfo] = &[
  EntryInfo {
    id    : 0,
    name  : "HVM.log",
    arity : 2,
    is_fn : true,
  },
  EntryInfo {
    id    : 1,
    name  : "HVM.put",
    arity : 2,
    is_fn : true,
  },
  EntryInfo {
    id    : 2,
    name  : "String.nil",
    arity : 0,
    is_fn : false,
  },
  EntryInfo {
    id    : 3,
    name  : "String.cons",
    arity : 2,
    is_fn : false,
  },
  EntryInfo {
    id    : 4,
    name  : "IO.done",
    arity : 1,
    is_fn : false,
  },
  EntryInfo {
    id    : 5,
    name  : "IO.do_input",
    arity : 1,
    is_fn : false,
  },
  EntryInfo {
    id    : 6,
    name  : "IO.do_output",
    arity : 2,
    is_fn : false,
  },
  EntryInfo {
    id    : 7,
    name  : "IO.do_fetch",
    arity : 3,
    is_fn : false,
  },
  EntryInfo {
    id    : 8,
    name  : "IO.do_store",
    arity : 3,
    is_fn : false,
  },
  EntryInfo {
    id    : 9,
    name  : "IO.do_load",
    arity : 2,
    is_fn : false,
  },
  EntryInfo {
    id    : 10,
    name  : "Kind.Term.ct0",
    arity : 2,
    is_fn : false,
  },
  EntryInfo {
    id    : 11,
    name  : "Kind.Term.ct1",
    arity : 3,
    is_fn : false,
  },
  EntryInfo {
    id    : 12,
    name  : "Kind.Term.ct2",
    arity : 4,
    is_fn : false,
  },
  EntryInfo {
    id    : 13,
    name  : "Kind.Term.ct3",
    arity : 5,
    is_fn : false,
  },
  EntryInfo {
    id    : 14,
    name  : "Kind.Term.ct4",
    arity : 6,
    is_fn : false,
  },
  EntryInfo {
    id    : 15,
    name  : "Kind.Term.ct5",
    arity : 7,
    is_fn : false,
  },
  EntryInfo {
    id    : 16,
    name  : "Kind.Term.ct6",
    arity : 8,
    is_fn : false,
  },
  EntryInfo {
    id    : 17,
    name  : "Kind.Term.ct7",
    arity : 9,
    is_fn : false,
  },
  EntryInfo {
    id    : 18,
    name  : "Kind.Term.ct8",
    arity : 10,
    is_fn : false,
  },
  EntryInfo {
    id    : 19,
    name  : "Kind.Term.ct9",
    arity : 11,
    is_fn : false,
  },
  EntryInfo {
    id    : 20,
    name  : "Kind.Term.ctA",
    arity : 12,
    is_fn : false,
  },
  EntryInfo {
    id    : 21,
    name  : "Kind.Term.ctB",
    arity : 13,
    is_fn : false,
  },
  EntryInfo {
    id    : 22,
    name  : "Kind.Term.ctC",
    arity : 14,
    is_fn : false,
  },
  EntryInfo {
    id    : 23,
    name  : "Kind.Term.ctD",
    arity : 15,
    is_fn : false,
  },
  EntryInfo {
    id    : 24,
    name  : "Kind.Term.ctE",
    arity : 16,
    is_fn : false,
  },
  EntryInfo {
    id    : 25,
    name  : "Kind.Term.ctF",
    arity : 17,
    is_fn : false,
  },
  EntryInfo {
    id    : 26,
    name  : "Kind.Term.ctG",
    arity : 18,
    is_fn : false,
  },
  EntryInfo {
    id    : 27,
    name  : "Kind.Term.num",
    arity : 2,
    is_fn : false,
  },
//GENERATED-ENTRY-INFOS//
];

const HVM_LOG : u64 = 0;
const HVM_PUT : u64 = 1;
const STRING_NIL : u64 = 2;
const STRING_CONS : u64 = 3;
const IO_DONE : u64 = 4;
const IO_DO_INPUT : u64 = 5;
const IO_DO_OUTPUT : u64 = 6;
const IO_DO_FETCH : u64 = 7;
const IO_DO_STORE : u64 = 8;
const IO_DO_LOAD : u64 = 9;
const KIND_TERM_CT0 : u64 = 10;
const KIND_TERM_CT1 : u64 = 11;
const KIND_TERM_CT2 : u64 = 12;
const KIND_TERM_CT3 : u64 = 13;
const KIND_TERM_CT4 : u64 = 14;
const KIND_TERM_CT5 : u64 = 15;
const KIND_TERM_CT6 : u64 = 16;
const KIND_TERM_CT7 : u64 = 17;
const KIND_TERM_CT8 : u64 = 18;
const KIND_TERM_CT9 : u64 = 19;
const KIND_TERM_CTA : u64 = 20;
const KIND_TERM_CTB : u64 = 21;
const KIND_TERM_CTC : u64 = 22;
const KIND_TERM_CTD : u64 = 23;
const KIND_TERM_CTE : u64 = 24;
const KIND_TERM_CTF : u64 = 25;
const KIND_TERM_CTG : u64 = 26;
const KIND_TERM_NUM : u64 = 27;
//GENERATED-ENTRY-IDS//

pub const MAX_RESERVED_ID : u64 = KIND_TERM_NUM;

pub const HEAP_SIZE : usize = 4 * CELLS_PER_GB;

// Types
// -----

pub type Ptr = u64;
pub type AtomicPtr = AtomicU64;

// A runtime term
#[derive(Clone, Debug)]
pub enum Core {
  Var { bidx: u64 },
  Glo { glob: u64, misc: u64 },
  Dup { eras: (bool, bool), glob: u64, expr: Box<Core>, body: Box<Core> },
  Let { expr: Box<Core>, body: Box<Core> },
  Lam { eras: bool, glob: u64, body: Box<Core> },
  App { func: Box<Core>, argm: Box<Core> },
  Fun { func: u64, args: Vec<Core> },
  Ctr { func: u64, args: Vec<Core> },
  Num { numb: u64 },
  Op2 { oper: u64, val0: Box<Core>, val1: Box<Core> },
}

// A runtime rule
#[derive(Clone, Debug)]
pub struct Rule {
  pub hoas: bool,
  pub cond: Vec<Ptr>,
  pub vars: Vec<RuleVar>,
  pub core: Core,
  pub body: RuleBody,
  pub free: Vec<(u64, u64)>,
}

// A rule left-hand side variable
#[derive(Clone, Debug)]
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
  Val { value: u64 }, // Fixed value, doesn't require adjustment
  Var { index: u64 }, // Link to an external variable
  Ptr { value: u64, targ: u64, slot: u64 }, // Local link, requires adjustment
}

#[derive(Clone)]
pub struct Function {
  pub arity: u64,
  pub is_strict: Vec<bool>,
  pub stricts: Vec<u64>,
  pub rules: Vec<Rule>,
}

#[derive(Clone)]
pub struct Arity(pub u64);

pub type Funs = Vec<Option<Function>>;
pub type Aris = Vec<Arity>;
pub type Nams = HashMap<u64, String>;

#[derive(Clone)]
pub struct Program {
  pub funs: Funs,
  pub aris: Aris,
  pub nams: Nams,
}

// Thread local data and stats
#[derive(Debug)]
pub struct LocalVars {
  pub tid: usize,
  pub used: AtomicI64, // number of used memory cells
  pub next: AtomicU64, // next alloc index
  pub amin: AtomicU64, // min alloc index
  pub amax: AtomicU64, // max alloc index
  pub dups: AtomicU64, // next dup label to be created
  pub cost: AtomicU64, // total number of rewrite rules
}

// Global memory buffer
pub struct Heap {
  node: Memory,
  //node: Box<[AtomicPtr]>,
  lvar: Box<[CachePadded<LocalVars>]>,
  lock: Box<[AtomicU8]>,
  vstk: Box<[VisitQueue]>,
  aloc: Box<[Box<[AtomicU64]>]>,
  rbag: RedexBag,
}

// Initializers
// ------------

fn available_parallelism() -> usize {
  //return 1;
  return std::thread::available_parallelism().unwrap().get();
}

pub fn new_atomic_u8_array(size: usize) -> Box<[AtomicU8]> {
  return unsafe { Box::from_raw(AtomicU8::from_mut_slice(Box::leak(vec![0xFFu8; size].into_boxed_slice()))) }
}

pub fn new_atomic_u64_array(size: usize) -> Box<[AtomicU64]> {
  return unsafe { Box::from_raw(AtomicU64::from_mut_slice(Box::leak(vec![0u64; size].into_boxed_slice()))) }
}

pub fn new_heap() -> Heap {
  let tids = available_parallelism();
  let mut lvar = vec![];
  for tid in 0 .. tids {
    lvar.push(CachePadded::new(LocalVars {
      tid: tid,
      used: AtomicI64::new(0),
      next: AtomicU64::new((HEAP_SIZE / tids * (tid + 0)) as u64),
      amin: AtomicU64::new((HEAP_SIZE / tids * (tid + 0)) as u64),
      amax: AtomicU64::new((HEAP_SIZE / tids * (tid + 1)) as u64),
      dups: AtomicU64::new(((1 << 28) / tids * tid) as u64),
      cost: AtomicU64::new(0),
    }))
  }
  let size = HEAP_SIZE; // FIXME: accept size param
  //let node = new_atomic_u64_array(size);
  let node = Memory::new();
  let lvar = lvar.into_boxed_slice();
  let lock = new_atomic_u8_array(size);
  let rbag = RedexBag::new();
  let aloc = (0 .. tids).map(|x| new_atomic_u64_array(1 << 20)).collect::<Vec<Box<[AtomicU64]>>>().into_boxed_slice();
  let vstk = (0 .. tids).map(|x| VisitQueue::new()).collect::<Vec<VisitQueue>>().into_boxed_slice();
  return Heap { node, lvar, lock, rbag, aloc, vstk };
}

pub fn new_tids() -> Box<[usize]> {
  return (0 .. available_parallelism()).collect::<Vec<usize>>().into_boxed_slice();
}

pub fn new_program() -> Program {
  Program {
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

pub fn Sup(col: u64, pos: u64) -> Ptr {
  (SUP * TAG) | (col * EXT) | pos
}

pub fn Op2(ope: u64, pos: u64) -> Ptr {
  (OP2 * TAG) | (ope * EXT) | pos
}

pub fn Num(val: u64) -> Ptr {
  (NUM * TAG) | (val & NUM_MASK)
}

pub fn Ctr(fun: u64, pos: u64) -> Ptr {
  (CTR * TAG) | (fun * EXT) | pos
}

// FIXME: update name to Fun
pub fn Fun(fun: u64, pos: u64) -> Ptr {
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

pub fn get_cost(heap: &Heap) -> u64 {
  heap.lvar.iter().map(|x| x.cost.load(Ordering::Relaxed)).sum()
}

pub fn get_used(heap: &Heap) -> i64 {
  heap.lvar.iter().map(|x| x.used.load(Ordering::Relaxed)).sum()
}

pub fn inc_cost(heap: &Heap, tid: usize) {
  heap.lvar[tid].cost.fetch_add(1, Ordering::Relaxed);
}

pub fn gen_dup(heap: &Heap, tid: usize) -> u64 {
  return heap.lvar[tid].dups.fetch_add(1, Ordering::Relaxed) & 0xFFFFFFF;
}

// Program
// -------

pub fn ask_ari(prog: &Program, lnk: Ptr) -> u64 {
  let fid = get_ext(lnk);
  if let Some(node_info) = ENTRY_INFO.get(fid as usize) {
    return node_info.arity as u64;
  }
  if let Some(Arity(arit)) = prog.aris.get(fid as usize) {
    return *arit;
  }
  return 0;
}

// Pointers
// --------

// DEPRECATED
pub fn ask_lnk(heap: &Heap, loc: u64) -> Ptr {
  unsafe { *(*heap.node.data.get_unchecked(loc as usize)).as_mut_ptr() }
}

// DEPRECATED
pub fn ask_arg(heap: &Heap, term: Ptr, arg: u64) -> Ptr {
  ask_lnk(heap, get_loc(term, arg))
}

// Given a location, loads the ptr stored on it
pub fn load_ptr(heap: &Heap, loc: u64) -> Ptr {
  unsafe { heap.node.data.get_unchecked(loc as usize).load(Ordering::Relaxed) }
}

// Moves a pointer to another location
pub fn move_ptr(heap: &Heap, old_loc: u64, new_loc: u64) -> Ptr {
  link(heap, new_loc, take_ptr(heap, old_loc))
}

// Given a pointer to a node, loads its nth arg
pub fn load_arg(heap: &Heap, term: Ptr, arg: u64) -> Ptr {
  load_ptr(heap, get_loc(term, arg))
}

// Given a location, takes the ptr stored on it
pub fn take_ptr(heap: &Heap, loc: u64) -> Ptr {
  unsafe { heap.node.data.get_unchecked(loc as usize).swap(0, Ordering::Relaxed) }
}

// Given a pointer to a node, takes its nth arg
pub fn take_arg(heap: &Heap, term: Ptr, arg: u64) -> Ptr {
  take_ptr(heap, get_loc(term, arg))
}

// Writes a ptr to memory. Updates binders.
pub fn link(heap: &Heap, loc: u64, ptr: Ptr) -> Ptr {
  unsafe {
    heap.node.data.get_unchecked(loc as usize).store(ptr, Ordering::Relaxed);
    if get_tag(ptr) <= VAR {
      let arg_loc = get_loc(ptr, get_tag(ptr) & 0x01);
      heap.node.data.get_unchecked(arg_loc as usize).store(Arg(loc), Ordering::Relaxed);
    }
  }
  ptr
}

// Allocator
// ---------

pub fn alloc(heap: &Heap, tid: usize, arity: u64) -> u64 {
  heap.node.alloc(tid, arity)
  //unsafe {
    //let lvar = &heap.lvar[tid];
    //if arity == 0 {
      //0
    //} else {
      //let mut length = 0;
      //loop {
        //// Loads value on cursor
        //let val = heap.node.data.get_unchecked(*lvar.next.as_mut_ptr() as usize).load(Ordering::Relaxed);
        //// If it is empty, increment length
        //if val == 0 {
          //length += 1;
        //// Otherwise, reset length
        //} else {
          //length = 0;
        //};
        //// Moves cursor right
        //*lvar.next.as_mut_ptr() += 1;
        //// If it is out of bounds, warp around
        //if *lvar.next.as_mut_ptr() >= *lvar.amax.as_mut_ptr() {
          //length = 0;
          //*lvar.next.as_mut_ptr() = *lvar.amin.as_mut_ptr();
          ////println!("[{}] loop", lvar.tid);
        //}
        //// If length equals arity, allocate that space
        //if length == arity {
          ////println!("[{}] return", lvar.tid);
          ////println!("[{}] alloc {} at {}", lvar.tid, arity, lvar.next - length);
          ////lvar.used.fetch_add(arity as i64, Ordering::Relaxed);
          //return *lvar.next.as_mut_ptr() - length;
        //}
      //}
    //}
  //}
}

pub fn free(heap: &Heap, tid: usize, loc: u64, arity: u64) {
  heap.node.free(tid, loc, arity)
  //heap.lvar[tid].used.fetch_sub(arity as i64, Ordering::Relaxed);
  //for i in 0 .. arity {
    //unsafe { heap.node.data.get_unchecked((loc + i) as usize) }.store(0, Ordering::Relaxed);
  //}
}

// Substitution
// ------------

// Atomically replaces a ptr by another. Updates binders.
pub fn atomic_relink(heap: &Heap, loc: u64, old: Ptr, neo: Ptr) -> Result<Ptr, Ptr> {
  unsafe {
    let got = heap.node.data.get_unchecked(loc as usize).compare_exchange_weak(old, neo, Ordering::Relaxed, Ordering::Relaxed)?;
    if get_tag(neo) <= VAR {
      let arg_loc = get_loc(neo, get_tag(neo) & 0x01);
      heap.node.data.get_unchecked(arg_loc as usize).store(Arg(loc), Ordering::Relaxed);
    }
    return Ok(got);
  }
}

// Performs a global [x <- val] substitution atomically.
pub fn atomic_subst(heap: &Heap, prog: &Program, tid: usize, var: Ptr, val: Ptr) {
  loop {
    let arg_ptr = load_ptr(heap, get_loc(var, get_tag(var) & 0x01));
    if get_tag(arg_ptr) == ARG {
      match atomic_relink(heap, get_loc(arg_ptr, 0), var, val) {
        Ok(_) => { return; }
        Err(_) => { continue; }
      }
    }
    if get_tag(arg_ptr) == ERA {
      collect(heap, prog, tid, val); // safe, since `val` is owned by this thread
      return;
    }
  }
}

// Garbage Collection
// ------------------

// As soon as we detect an expression is unreachable, i.e., when it is applied to a lambda or
// function that doesn't use its argument, we call `collect()` on it. Since the expression is now
// implicitly "owned" by this thread, we're allowed to traverse the structure and fully free its
// memory. There are some complications, though: lambdas, duplications, and their respective
// variables. When a lam is collected, we must first substitute its bound variable by `Era()`, and
// then recurse. When a lam-bound variable is collected, we just link its argument to `Era()`. This
// will allow lams to be collected properly in all scenarios.
//
// A. When the lam is collected before the var. Ex: λx (Pair 42 x)
//    1. We substitute [x <- Era()] and recurse into the lam's body.
//    2. When we reach x, it will be Era(), so there is nothing to do.
//    3. All memory related to this lambda is freed.
//    This is safe, because both are owned by this thread
//
// B. When the var is collected before the lam. Ex: (Pair x λx(42))
//    1. We reach x and link the lam's argument to Era().
//    2. When we reach the lam, its var will be Era(), so [Era() <- Era()] will do nothing.
//    3. All memory related to this lambda is freed.
//    This is safe, because both are owned by this thread.
//
// C. When the var is collected, but the lam isn't. Ex: (Pair x 42)
//    1. We reach x and link the lam's argument to Era().
//    2. The owner of the lam can still use it, and applying it will trigger collect().
//    This is safe, because the lam arg field is owned by the thread that owns the var (this one).
//
// D. When the lam is collected, but the var isn't. Ex: (Pair λx(42) 777)
//    1. We reach the lam and substitute [x <- Era()].
//    2. The owner of var will now have an Era(), rather than an unbound variable.
//    This is safe because, subst is atomic.
//
// As for dup nodes, the same idea applies. When a dup-bound variable is collected, we just link
// its argument to Era(). The problem is, it is impossible to reach a dup node directly. Because
// of that, if two threads collected the same dup, we'd have a memory leak: the dup node wouldn't
// be freed, and the dup expression wouldn't be collected. As such, when we reach a dup-bound
// variable, we also visit the dup node. Visiting dup nodes doesn't imply ownership, since a dup
// node can be accessed through two different dup-bound variables. As such, we attempt to lock it.
// If we can't have the lock, that means another thread is handling that dup, so we let it decide
// what to do with it, and return. If we get the lock, then we now have ownership, so we check the
// other argument. If it is Era(), that means this dup node was collected twice, so, we clear it
// and collect its expression. Otherwise, we release the lock and let the owner of the other
// variable decide what to do with it in a future. This covers most cases, but the is still a
// problem: what if the other variable is contained inside the duplicated expression? For example,
// the normal form of `(λf λx (f (f x)) λf λx (f (f x)))` is:
//
// λf λx b0
// dup f0 f1 = f
// dup b0 b1 = (f0 (f1 {b1 x}))
// 
// If we attempt to collect it with the algorithm above, we'll have:
//
// dup f0 f1 = ~
// dup ~  b1 = (f0 (f1 {b1 ~}))
//
// That's because, once we reached `b0`, we replaced its respective arg by `Era()`, then locked its
// dup node and checked the other arg, `b1`; since it isn't `Era()`, we released the lock and let
// the owner of `b1` decide what to do. But `b1` is contained inside the expression, so it has no
// owner anymore; it forms a cycle, and no other part of the program will access it! This will not
// be handled by HVM's automatic collector and will be left as a memory leak. Under normal
// circumstances, the leak is too minimal to be a problem. It could be eliminated by enabling an
// external garbage collector (which would rarely need to be triggered), or avoided altogether by
// not allowing inputs that can result in self-referential clones on the input language's type
// system. Sadly, it is an issue that exists, and, for the time being, I'm not aware of a good
// solution that maintains HVM philosophy of only including constant-time compute primitives.

pub fn collect(heap: &Heap, prog: &Program, tid: usize, term: Ptr) {
  let mut coll = Vec::new();
  let mut next = term;
  loop {
    let term = next;
    match get_tag(term) {
      DP0 => {
        link(heap, get_loc(term, 0), Era());
        if acquire_lock(heap, tid, term).is_ok() {
          if get_tag(load_arg(heap, term, 1)) == ERA {
            collect(heap, prog, tid, load_arg(heap, term, 2));
            free(heap, tid, get_loc(term, 0), 3);
          }
          release_lock(heap, tid, term);
        }
      }
      DP1 => {
        link(heap, get_loc(term, 1), Era());
        if acquire_lock(heap, tid, term).is_ok() {
          if get_tag(load_arg(heap, term, 0)) == ERA {
            collect(heap, prog, tid, load_arg(heap, term, 2));
            free(heap, tid, get_loc(term, 0), 3);
          }
          release_lock(heap, tid, term);
        }
      }
      VAR => {
        link(heap, get_loc(term, 0), Era());
      }
      LAM => {
        atomic_subst(heap, prog, tid, Var(get_loc(term,0)), Era());
        next = take_arg(heap, term, 1);
        free(heap, tid, get_loc(term, 0), 2);
        continue;
      }
      APP => {
        coll.push(take_arg(heap, term, 0));
        next = take_arg(heap, term, 1);
        free(heap, tid, get_loc(term, 0), 2);
        continue;
      }
      SUP => {
        coll.push(take_arg(heap, term, 0));
        next = take_arg(heap, term, 1);
        free(heap, tid, get_loc(term, 0), 2);
        continue;
      }
      OP2 => {
        coll.push(take_arg(heap, term, 0));
        next = take_arg(heap, term, 1);
        free(heap, tid, get_loc(term, 0), 2);
        continue;
      }
      NUM => {}
      CTR | FUN => {
        let arity = ask_ari(prog, term);
        for i in 0 .. arity {
          if i < arity - 1 {
            coll.push(take_arg(heap, term, i));
          } else {
            next = take_arg(heap, term, i);
          }
        }
        free(heap, tid, get_loc(term, 0), arity);
        if arity > 0 {
          continue;
        }
      }
      _ => {}
    }
    if let Some(got) = coll.pop() {
      next = got;
    } else {
      break;
    }
  }
}

pub fn alloc_body(heap: &Heap, tid: usize, term: Ptr, vars: &[RuleVar], body: &RuleBody) -> Ptr {
  // FIXME: verify the use of get_unchecked
  let (elem, nodes, dupk) = body;
  fn elem_to_ptr(heap: &Heap, lvar: &LocalVars, aloc: &[AtomicU64], term: Ptr, vars: &[RuleVar], elem: &RuleBodyCell) -> Ptr {
    unsafe {
      match elem {
        RuleBodyCell::Val { value } => {
          *value
        },
        RuleBodyCell::Var { index } => {
          get_var(heap, term, vars.get_unchecked(*index as usize))
        },
        RuleBodyCell::Ptr { value, targ, slot } => {
          let mut val = value + *aloc.get_unchecked(*targ as usize).as_mut_ptr() + slot;
          // should be changed if the pointer format changes
          if get_tag(*value) == DP0 {
            val += (*lvar.dups.as_mut_ptr() & 0xFFFFFFF) * EXT;
          }
          if get_tag(*value) == DP1 {
            val += (*lvar.dups.as_mut_ptr() & 0xFFFFFFF) * EXT;
          }
          val
        }
      }
    }
  }
  unsafe {
    let aloc = &heap.aloc[tid];
    let lvar = &heap.lvar[tid];
    for i in 0 .. nodes.len() {
      *aloc.get_unchecked(i).as_mut_ptr() = alloc(heap, tid, (*nodes.get_unchecked(i)).len() as u64);
    };
    for i in 0 .. nodes.len() {
      let host = *aloc.get_unchecked(i).as_mut_ptr() as usize;
      for j in 0 .. (*nodes.get_unchecked(i)).len() {
        let elem = (*nodes.get_unchecked(i)).get_unchecked(j);
        let ptr = elem_to_ptr(heap, lvar, aloc, term, vars, elem);
        if let RuleBodyCell::Var { .. } = elem {
          link(heap, (host + j) as u64, ptr);
        } else {
          *heap.node.data.get_unchecked(host + j).as_mut_ptr() = ptr;
        }
      };
    };
    let done = elem_to_ptr(heap, lvar, aloc, term, vars, elem);
    *lvar.dups.as_mut_ptr() += dupk;
    return done;
  }
}

pub fn get_var(heap: &Heap, term: Ptr, var: &RuleVar) -> Ptr {
  let RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => take_arg(heap, load_arg(heap, term, *param), *i),
    None    => take_arg(heap, term, *param),
  }
}

// Rewrite Rules
// -------------

// (λx(body) a)
// ------------ APP-LAM
// x <- a
// body
#[inline(always)]
fn app_lam(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr) {
  //println!("app-lam");
  inc_cost(heap, tid);
  atomic_subst(heap, prog, tid, Var(get_loc(arg0, 0)), take_arg(heap, term, 1));
  link(heap, host, take_arg(heap, arg0, 1));
  free(heap, tid, get_loc(term, 0), 2);
  free(heap, tid, get_loc(arg0, 0), 2);
}

// ({a b} c)
// --------------- APP-SUP
// dup x0 x1 = c
// {(a x0) (b x1)}
#[inline(always)]
fn app_sup(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr) {
  inc_cost(heap, tid);
  let app0 = get_loc(term, 0);
  let app1 = get_loc(arg0, 0);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, term, 1));
  link(heap, app0 + 1, Dp0(get_ext(arg0), let0));
  link(heap, app0 + 0, take_arg(heap, arg0, 0));
  link(heap, app1 + 0, take_arg(heap, arg0, 1));
  link(heap, app1 + 1, Dp1(get_ext(arg0), let0));
  link(heap, par0 + 0, App(app0));
  link(heap, par0 + 1, App(app1));
  let done = Sup(get_ext(arg0), par0);
  link(heap, host, done);
}

// dup r s = λx(f)
// --------------- DUP-LAM
// dup f0 f1 = f
// r <- λx0(f0)
// s <- λx1(f1)
// x <- {x0 x1}
#[inline(always)]
fn dup_lam(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  let lam0 = alloc(heap, tid, 2);
  let lam1 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, arg0, 1));
  link(heap, par0 + 1, Var(lam1));
  link(heap, par0 + 0, Var(lam0));
  link(heap, lam0 + 1, Dp0(get_ext(term), let0));
  link(heap, lam1 + 1, Dp1(get_ext(term), let0));
  atomic_subst(heap, prog, tid, Var(get_loc(arg0, 0)), Sup(get_ext(term), par0));
  atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), Lam(lam0));
  atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), Lam(lam1));
  let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
  link(heap, host, done);
  free(heap, tid, get_loc(term, 0), 3);
  free(heap, tid, get_loc(arg0, 0), 2);
}

// dup x y = {a b}
// --------------- DUP-SUP (equal)
// x <- a
// y <- b
#[inline(always)]
fn dup_sup_0(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), take_arg(heap, arg0, 0));
  atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), take_arg(heap, arg0, 1));
  free(heap, tid, get_loc(term, 0), 3);
  free(heap, tid, get_loc(arg0, 0), 2);
}

// dup x y = {a b}
// --------------- DUP-SUP (different)
// x <- {xA xB}
// y <- {yA yB}
// dup xA yA = a
// dup xB yB = b
#[inline(always)]
fn dup_sup_1(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let par0 = alloc(heap, tid, 2);
  let let0 = alloc(heap, tid, 3);
  let par1 = get_loc(arg0, 0);
  let let1 = alloc(heap, tid, 3);
  link(heap, let0 + 2, take_arg(heap, arg0, 0));
  link(heap, let1 + 2, take_arg(heap, arg0, 1));
  link(heap, par1 + 0, Dp1(tcol, let0));
  link(heap, par1 + 1, Dp1(tcol, let1));
  link(heap, par0 + 0, Dp0(tcol, let0));
  link(heap, par0 + 1, Dp0(tcol, let1));
  atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par0));
  atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par1));
  free(heap, tid, get_loc(term, 0), 3);
}

// dup x y = N
// ----------- DUP-NUM
// x <- N
// y <- N
// ~
#[inline(always)]
fn dup_num(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), arg0);
  atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), arg0);
  free(heap, tid, get_loc(term, 0), 3);
}

// dup x y = (K a b c ...)
// ----------------------- DUP-CTR
// dup a0 a1 = a
// dup b0 b1 = b
// dup c0 c1 = c
// ...
// x <- (K a0 b0 c0 ...)
// y <- (K a1 b1 c1 ...)
#[inline(always)]
fn dup_ctr(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let fnid = get_ext(arg0);
  let arit = ask_ari(prog, arg0);
  if arit == 0 {
    atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnid, 0));
    atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnid, 0));
    link(heap, host, Ctr(fnid, 0));
    free(heap, tid, get_loc(term, 0), 3);
  } else {
    let ctr0 = get_loc(arg0, 0);
    let ctr1 = alloc(heap, tid, arit);
    for i in 0 .. arit - 1 {
      let leti = alloc(heap, tid, 3);
      link(heap, leti + 2, take_arg(heap, arg0, i));
      link(heap, ctr0 + i, Dp0(get_ext(term), leti));
      link(heap, ctr1 + i, Dp1(get_ext(term), leti));
    }
    let leti = alloc(heap, tid, 3);
    link(heap, leti + 2, take_arg(heap, arg0, arit - 1));
    link(heap, ctr0 + arit - 1, Dp0(get_ext(term), leti));
    link(heap, ctr1 + arit - 1, Dp1(get_ext(term), leti));
    atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnid, ctr0));
    atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnid, ctr1));
    //let done = Ctr(fnid, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
    //link(heap, host, done);
    free(heap, tid, get_loc(term, 0), 3);
  }
}

// dup x y = *
// ----------- DUP-ERA
// x <- *
// y <- *
#[inline(always)]
fn dup_era(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  atomic_subst(heap, prog, tid, Dp0(tcol, get_loc(term, 0)), Era());
  atomic_subst(heap, prog, tid, Dp1(tcol, get_loc(term, 0)), Era());
  link(heap, host, Era());
  free(heap, tid, get_loc(term, 0), 3);
}

// (+ a b)
// --------- OP2-NUM
// add(a, b)
#[inline(always)]
fn op2_num(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, arg1: Ptr) {
  inc_cost(heap, tid);
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
  free(heap, tid, get_loc(term, 0), 2);
}

// (+ {a0 a1} b)
// --------------------- OP2-SUP-0
// dup b0 b1 = b
// {(+ a0 b0) (+ a1 b1)}
#[inline(always)]
fn op2_sup_0(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, arg1: Ptr) {
  inc_cost(heap, tid);
  let op20 = get_loc(term, 0);
  let op21 = get_loc(arg0, 0);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, arg1);
  link(heap, op20 + 1, Dp0(get_ext(arg0), let0));
  link(heap, op20 + 0, take_arg(heap, arg0, 0));
  link(heap, op21 + 0, take_arg(heap, arg0, 1));
  link(heap, op21 + 1, Dp1(get_ext(arg0), let0));
  link(heap, par0 + 0, Op2(get_ext(term), op20));
  link(heap, par0 + 1, Op2(get_ext(term), op21));
  let done = Sup(get_ext(arg0), par0);
  link(heap, host, done);
}

// (+ a {b0 b1})
// --------------- OP2-SUP-1
// dup a0 a1 = a
// {(+ a0 b0) (+ a1 b1)}
#[inline(always)]
fn op2_sup_1(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, arg0: Ptr, arg1: Ptr) {
  inc_cost(heap, tid);
  let op20 = get_loc(term, 0);
  let op21 = get_loc(arg1, 0);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, arg0);
  link(heap, op20 + 0, Dp0(get_ext(arg1), let0));
  link(heap, op20 + 1, take_arg(heap, arg1, 0));
  link(heap, op21 + 1, take_arg(heap, arg1, 1));
  link(heap, op21 + 0, Dp1(get_ext(arg1), let0));
  link(heap, par0 + 0, Op2(get_ext(term), op20));
  link(heap, par0 + 1, Op2(get_ext(term), op21));
  let done = Sup(get_ext(arg1), par0);
  link(heap, host, done);
}

#[inline(always)]
pub fn fun_sup(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, argn: Ptr, n: u64) -> Ptr {
  inc_cost(heap, tid);
  let arit = ask_ari(prog, term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(heap, tid, arit);
  let par0 = get_loc(argn, 0);
  for i in 0 .. arit {
    if i != n {
      let leti = alloc(heap, tid, 3);
      let argi = take_arg(heap, term, i);
      link(heap, fun0 + i, Dp0(get_ext(argn), leti));
      link(heap, fun1 + i, Dp1(get_ext(argn), leti));
      link(heap, leti + 2, argi);
    } else {
      link(heap, fun0 + i, take_arg(heap, argn, 0));
      link(heap, fun1 + i, take_arg(heap, argn, 1));
    }
  }
  link(heap, par0 + 0, Fun(func, fun0));
  link(heap, par0 + 1, Fun(func, fun1));
  let done = Sup(get_ext(argn), par0);
  link(heap, host, done);
  done
}

#[inline(always)]
fn fun_ctr(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, fid: u64) -> bool {

  if let Some(Some(function)) = &prog.funs.get(fid as usize) {
    // Reduces function superpositions
    for (n, is_strict) in function.is_strict.iter().enumerate() {
      let n = n as u64;
      if *is_strict && get_tag(load_arg(heap, term, n)) == SUP {
        fun_sup(heap, prog, tid, host, term, load_arg(heap, term, n), n);
        return true;
      }
    }

    // For each rule condition vector
    let mut matched;
    for (r, rule) in function.rules.iter().enumerate() {
      // Check if the rule matches
      matched = true;
      
      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      for (i, cond) in rule.cond.iter().enumerate() {
        let i = i as u64;
        match get_tag(*cond) {
          NUM => {
            let same_tag = get_tag(load_arg(heap, term, i)) == NUM;
            let same_val = get_num(load_arg(heap, term, i)) == get_num(*cond);
            matched = matched && same_tag && same_val;
          }
          CTR => {
            let same_tag = get_tag(load_arg(heap, term, i)) == CTR;
            let same_ext = get_ext(load_arg(heap, term, i)) == get_ext(*cond);
            matched = matched && same_tag && same_ext;
          }
          VAR => {
            // If this is a strict argument, then we're in a default variable
            if unsafe { *function.is_strict.get_unchecked(i as usize) } {

              // This is a Kind2-specific optimization. Check 'KIND_TERM_OPT'.
              if rule.hoas && r != function.rules.len() - 1 {

                // Matches number literals
                let is_num
                  = get_tag(load_arg(heap, term, i)) == NUM;

                // Matches constructor labels
                let is_ctr
                  =  get_tag(load_arg(heap, term, i)) == CTR
                  && ask_ari(prog, load_arg(heap, term, i)) == 0;

                // Matches KIND_TERM numbers and constructors
                let is_hoas_ctr_num
                  =  get_tag(load_arg(heap, term, i)) == CTR
                  && get_ext(load_arg(heap, term, i)) >= KIND_TERM_CT0
                  && get_ext(load_arg(heap, term, i)) <= KIND_TERM_NUM;

                matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

              // Only match default variables on CTRs and NUMs
              } else {
                let is_ctr = get_tag(load_arg(heap, term, i)) == CTR;
                let is_num = get_tag(load_arg(heap, term, i)) == NUM;
                matched = matched && (is_ctr || is_num);
              }
            }
          }
          _ => {}
        }
      }

      // If all conditions are satisfied, the rule matched, so we must apply it
      if matched {
        // Increments the gas count
        inc_cost(heap, tid);

        // Builds the right-hand side term
        let done = alloc_body(heap, tid, term, &rule.vars, &rule.body);

        // Links the host location to it
        link(heap, host, done);

        // Collects unused variables
        for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
          if *erase {
            collect(heap, prog, tid, get_var(heap, term, var));
          }
        }

        // free the matched ctrs
        for (i, arity) in &rule.free {
          free(heap, tid, get_loc(load_arg(heap, term, *i as u64), 0), *arity);
        }
        free(heap, tid, get_loc(term, 0), function.arity);

        return true;
      }
    }
  }
  return false;
}

// Allocator
// ---------

struct MemoryNext {
  cell: AtomicU64,
  area: AtomicU64,
}

struct Memory {
  tids: usize,
  data: Box<[AtomicU64]>,
  used: Box<[AtomicU64]>,
  next: Box<[CachePadded<MemoryNext>]>,
}

const PAGE_SIZE : usize = 4096;

impl Memory {

  pub fn new() -> Memory {
    let tids = available_parallelism();
    let mut next = vec![];
    for i in 0 .. tids {
      let cell = AtomicU64::new(u64::MAX);
      let area = AtomicU64::new((HEAP_SIZE / PAGE_SIZE / tids * i) as u64);
      next.push(CachePadded::new(MemoryNext { cell, area }));
    }
    let data = new_atomic_u64_array(HEAP_SIZE);
    let used = new_atomic_u64_array(HEAP_SIZE / PAGE_SIZE);
    let next = next.into_boxed_slice();
    Memory { tids, data, used, next }
  }

  pub fn alloc(&self, tid: usize, arity: u64) -> u64 {
    let next = unsafe { self.next.get_unchecked(tid) };
    // Attempts to allocate on this thread's owned area
    let aloc = next.cell.fetch_add(arity, Ordering::Relaxed);
    let area = aloc / PAGE_SIZE as u64;
    if aloc != u64::MAX && (aloc + arity) / PAGE_SIZE as u64 == area {
      unsafe { self.used.get_unchecked(area as usize) }.fetch_add(arity, Ordering::Relaxed);
      //println!("[{}] old_alloc {} at {}, used={} ({} {})", tid, arity, aloc, self.used[area as usize].load(Ordering::Relaxed), area, (aloc + arity) / PAGE_SIZE as u64);
      return aloc;
    }
    // If we can't, attempt to allocate on a new area
    let mut area = next.area.load(Ordering::Relaxed)  % ((HEAP_SIZE / PAGE_SIZE) as u64);
    loop {
      if unsafe { self.used.get_unchecked(area as usize) }.compare_exchange_weak(0, arity, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        let aloc = area * PAGE_SIZE as u64;
        next.cell.store(aloc + arity, Ordering::Relaxed);
        next.area.store((area + 1) % ((HEAP_SIZE / PAGE_SIZE) as u64), Ordering::Relaxed);
        //println!("[{}] new_alloc {} at {}, used={}", tid, arity, aloc, self.used[area as usize].load(Ordering::Relaxed));
        return aloc;
      } else {
        area = (area + 1) % ((HEAP_SIZE / PAGE_SIZE) as u64);
      }
    }
  }

  pub fn free(&self, tid: usize, loc: u64, arity: u64) {
    //for i in 0 .. arity { unsafe { self.data.get_unchecked((loc + i) as usize) }.store(0, Ordering::Relaxed); }
    let area = loc / PAGE_SIZE as u64;
    let used = unsafe { self.used.get_unchecked(area as usize) }.fetch_sub(arity, Ordering::Relaxed);
    //println!("[{}] free {} at {}, used={}", tid, arity, loc, self.used[area as usize].load(Ordering::Relaxed));
  }

}

// Redex Bag
// ---------
// Concurrent bag featuring insert, read and modify. No pop.

const REDEX_BAG_SIZE : usize = 1 << 24;
const REDEX_CONT_RET : u64 = 0xFFFFFF; // signals to return

// - 32 bits: host
// - 24 bits: cont
// -  8 bits: left
type Redex = u64;

struct RedexBag {
  tids: usize,
  next: Box<[CachePadded<AtomicUsize>]>,
  data: Box<[AtomicU64]>,
}

fn new_redex(host: u64, cont: u64, left: u64) -> Redex {
  return (host << 32) | (cont << 8) | left;
}

fn get_redex_host(redex: Redex) -> u64 {
  return redex >> 32;
}

fn get_redex_cont(redex: Redex) -> u64 {
  return (redex >> 8) & 0xFFFFFF;
}

fn get_redex_left(redex: Redex) -> u64 {
  return redex & 0xFF;
}

impl RedexBag {
  fn new() -> RedexBag {
    let tids = available_parallelism();
    let mut next = vec![];
    for _ in 0 .. tids {
      next.push(CachePadded::new(AtomicUsize::new(0)));
    }
    let next = next.into_boxed_slice();
    let data = new_atomic_u64_array(REDEX_BAG_SIZE);
    return RedexBag { tids, next, data };
  }

  fn min_index(&self, tid: usize) -> usize {
    return REDEX_BAG_SIZE / self.tids * (tid + 0);
  }

  fn max_index(&self, tid: usize) -> usize {
    return std::cmp::min(REDEX_BAG_SIZE / self.tids * (tid + 1), REDEX_CONT_RET as usize - 1);
  }

  fn insert(&self, tid: usize, redex: u64) -> u64 {
    loop {
      let index = self.next[tid].fetch_add(1, Ordering::Relaxed);
      if index + 1 >= self.max_index(tid) { 
        self.next[tid].store(self.min_index(tid), Ordering::Relaxed);
      }
      if self.data[index].compare_exchange_weak(0, redex, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        return index as u64;
      }
    }
  }

  fn complete(&self, index: u64) -> Option<(u64,u64)> {
    let redex = self.data[index as usize].fetch_sub(1, Ordering::Relaxed);
    if get_redex_left(redex) == 1 {
      self.data[index as usize].store(0, Ordering::Relaxed);
      return Some((get_redex_host(redex), get_redex_cont(redex)));
    } else {
      return None;
    }
  }
}

// Visit Queue
// -----------
// A concurrent task-stealing queue featuring push, pop and steal.

const VISIT_QUEUE_SIZE : usize = 1 << 24;

// - 32 bits: host
// - 32 bits: cont
type Visit = u64;

struct VisitQueue {
  init: CachePadded<AtomicUsize>,
  last: CachePadded<AtomicUsize>,
  data: Box<[AtomicU64]>,
}

fn new_visit(host: u64, cont: u64) -> Visit {
  return (host << 32) | cont;
}

fn get_visit_host(visit: Visit) -> u64 {
  return visit >> 32;
}

fn get_visit_cont(visit: Visit) -> u64 {
  return visit & 0xFFFFFFFF;
}

impl VisitQueue {
  fn new() -> VisitQueue {
    return VisitQueue {
      init: CachePadded::new(AtomicUsize::new(0)),
      last: CachePadded::new(AtomicUsize::new(0)),
      data: new_atomic_u64_array(VISIT_QUEUE_SIZE),
    }
  }

  fn push(&self, value: u64) {
    let index = self.last.fetch_add(1, Ordering::Relaxed);
    self.data[index].store(value, Ordering::Relaxed);
  }

  fn pop(&self) -> Option<(u64, u64)> {
    loop {
      let last = self.last.load(Ordering::Relaxed);
      if last > 0 {
        self.last.fetch_sub(1, Ordering::Relaxed);
        self.init.fetch_min(last - 1, Ordering::Relaxed);
        let visit = self.data[last - 1].swap(0, Ordering::Relaxed);
        if visit == 0 {
          continue;
        } else {
          return Some((get_visit_host(visit), get_visit_cont(visit)));
        }
      } else {
        return None;
      }
    }
  }

  fn steal(&self) -> Option<(u64, u64)> {
    let index = self.init.load(Ordering::Relaxed);
    let visit = self.data[index].load(Ordering::Relaxed);
    if visit != 0 {
      if let Ok(visit) = self.data[index].compare_exchange(visit, 0, Ordering::Relaxed, Ordering::Relaxed) {
        self.init.fetch_add(1, Ordering::Relaxed);
        return Some((get_visit_host(visit), get_visit_cont(visit)));
      }
    }
    return None;
  }
}

// Locks
// -----

const LOCK_OPEN : u8 = 0xFF;

fn acquire_lock(heap: &Heap, tid: usize, term: Ptr) -> Result<u8, u8> {
  let locker = unsafe { heap.lock.get_unchecked(get_loc(term, 0) as usize) };
  locker.compare_exchange_weak(LOCK_OPEN, tid as u8, Ordering::Acquire, Ordering::Relaxed)
}

fn release_lock(heap: &Heap, tid: usize, term: Ptr) {
  let locker = unsafe { heap.lock.get_unchecked(get_loc(term, 0) as usize) };
  locker.store(LOCK_OPEN, Ordering::Release)
}


// Reduce
// ------

pub fn reduce(heap: &Heap, prog: &Program, tids: &[usize], root: u64) -> Ptr {
  // Halting flag
  let stop = &AtomicBool::new(false);

  // Spawn a thread for each worker
  std::thread::scope(|s| {
    for tid in tids {
      let tid = *tid;
      s.spawn(move || {
        // Visit stacks
        let redex = &heap.rbag;
        let visit = &heap.vstk[tid];
        let mut delay = vec![];

        // Backoff
        let backoff = &Backoff::new();

        // State variables
        let mut work = if tid == tids[0] { true } else { false };
        let mut init = if tid == tids[0] { true } else { false };
        let mut cont = if tid == tids[0] { REDEX_CONT_RET } else { 0 };
        let mut host = if tid == tids[0] { root } else { 0 };

        'main: loop {
          //println!("[{}] reduce\n{}\n", tid, show_term(heap, prog, load_ptr(heap, root), load_ptr(heap, host)));
          //println!("[{}] loop {:?}", tid, &heap.node[0 .. 256]);
          //println!("[{}] loop work={} init={} cont={} host={} visit={} delay={} stop={} count={} | {}", tid, work, init, cont, host, visit.len(), delay.len(), stop.load(Ordering::Relaxed), count, show_ptr(load_ptr(heap, host)));
          if work {
            if init {
              let term = load_ptr(heap, host);
              //println!("[{}] work={} init={} cont={} host={}", tid, work, init, cont, host);
              match get_tag(term) {
                APP => {
                  let goup = redex.insert(tid, new_redex(host, cont, 1));
                  work = true;
                  init = true;
                  cont = goup;
                  host = get_loc(term, 0);
                  continue 'main;
                }
                DP0 | DP1 => {
                  match acquire_lock(heap, tid, term) {
                    Err(locker_tid) => {
                      delay.push(new_visit(host, cont));
                      work = false;
                      init = true;
                      continue 'main;
                    }
                    Ok(_) => {
                      // If the term changed, release lock and try again
                      if term != load_ptr(heap, host) {
                        release_lock(heap, tid, term);
                        continue 'main;
                      }
                      let goup = redex.insert(tid, new_redex(host, cont, 1));
                      work = true;
                      init = true;
                      cont = goup;
                      host = get_loc(term, 2);
                      continue 'main;
                    }
                  }
                }
                OP2 => {
                  let goup = redex.insert(tid, new_redex(host, cont, 2));
                  visit.push(new_visit(get_loc(term, 1), goup));
                  work = true;
                  init = true;
                  cont = goup;
                  host = get_loc(term, 0);
                  continue 'main;
                }
                FUN => {
                  let fid = get_ext(term);
                  if fid == HVM_LOG || fid == HVM_PUT {
                    work = true;
                    init = false;
                    continue 'main;
                  }
                  match fid {
//GENERATED-FUN-CTR-MATCH//
                    _ => {
                      // Dynamic functions
                      if let Some(Some(f)) = &prog.funs.get(fid as usize) {
                        let len = f.stricts.len() as u64;
                        if len == 0 {
                          work = true;
                          init = false;
                          continue 'main;
                        } else {
                          let goup = redex.insert(tid, new_redex(host, cont, f.stricts.len() as u64));
                          for (i, arg_idx) in f.stricts.iter().enumerate() {
                            if i < f.stricts.len() - 1 {
                              visit.push(new_visit(get_loc(term, *arg_idx), goup));
                            } else {
                              work = true;
                              init = true;
                              cont = goup;
                              host = get_loc(term, *arg_idx);
                              continue 'main;
                            }
                          }
                        }
                      }
                    }
                  }
                }
                _ => {}
              }
              work = true;
              init = false;
              continue 'main;
            } else {
              let term = load_ptr(heap, host);
              //println!("[{}] reduce {} | {}", tid, host, show_ptr(term));
              
              // Apply rewrite rules
              match get_tag(term) {
                APP => {
                  let arg0 = load_arg(heap, term, 0);
                  if get_tag(arg0) == LAM {
                    app_lam(heap, prog, tid, host, term, arg0);
                    work = true;
                    init = true;
                    continue 'main;
                  }
                  if get_tag(arg0) == SUP {
                    app_sup(heap, prog, tid, host, term, arg0);
                  }
                }
                DP0 | DP1 => {
                  let arg0 = load_arg(heap, term, 2);
                  let tcol = get_ext(term);
                  //println!("[{}] dups {}", lvar.tid, get_loc(term, 0));
                  if get_tag(arg0) == LAM {
                    dup_lam(heap, prog, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    work = true;
                    init = true;
                    continue 'main;
                  } else if get_tag(arg0) == SUP {
                    //println!("dup-sup {}", tcol == get_ext(arg0));
                    if tcol == get_ext(arg0) {
                      dup_sup_0(heap, prog, tid, host, term, arg0, tcol);
                      release_lock(heap, tid, term);
                      work = true;
                      init = true;
                      continue 'main;
                    } else {
                      dup_sup_1(heap, prog, tid, host, term, arg0, tcol);
                      release_lock(heap, tid, term);
                      work = true;
                      init = true;
                      continue 'main;
                    }
                  } else if get_tag(arg0) == NUM {
                    dup_num(heap, prog, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    work = true;
                    init = true;
                    continue 'main;
                  } else if get_tag(arg0) == CTR {
                    dup_ctr(heap, prog, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    work = true;
                    init = true;
                    continue 'main;
                  } else if get_tag(arg0) == ERA {
                    dup_era(heap, prog, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    work = true;
                    init = true;
                    continue 'main;
                  } else {
                    release_lock(heap, tid, term);
                  }
                }
                OP2 => {
                  let arg0 = load_arg(heap, term, 0);
                  let arg1 = load_arg(heap, term, 1);
                  if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
                    op2_num(heap, prog, tid, host, term, arg0, arg1);
                  } else if get_tag(arg0) == SUP {
                    op2_sup_0(heap, prog, tid, host, term, arg0, arg1);
                  } else if get_tag(arg1) == SUP {
                    op2_sup_1(heap, prog, tid, host, term, arg0, arg1);
                  }
                }
                FUN => {
                  let fid = get_ext(term);
                  if fid == HVM_LOG || fid == HVM_PUT {
                    normalize(heap, prog, &[tid], get_loc(term, 0), false);
                    let code = crate::readback::as_code(heap, prog, get_loc(term, 0));
                    if fid == HVM_PUT && code.chars().nth(0) == Some('"') {
                      println!("{}", &code[1 .. code.len() - 1]);
                    } else {
                      println!("{}", code);
                    }
                    collect(heap, prog, tid, load_ptr(heap, get_loc(term, 0)));
                    free(heap, tids[0], get_loc(term, 0), 2);
                    link(heap, host, load_arg(heap, term, 1));
                    init = true;
                    continue;
                  }
                  match fid {
//GENERATED-FUN-CTR-RULES//
                    _ => {
                      // Dynamic functions
                      if fun_ctr(heap, prog, tid, host, term, fid) {
                        work = true;
                        init = true;
                        continue 'main;
                      }
                    }
                  }
                }
                _ => {}
              }

              // If root is on WHNF, halt
              if cont == REDEX_CONT_RET {
                stop.store(true, Ordering::Relaxed);
                break;
              }

              // Otherwise, try reducing the parent redex
              if let Some((new_host, new_cont)) = redex.complete(cont) {
                work = true;
                init = false;
                host = new_host;
                cont = new_cont;
                continue 'main;
              }

              // Otherwise, visit next pointer
              work = false;
              init = true;
              continue 'main;
            }
          } else {
            if init {
              // If available, visit a new location
              if let Some((new_host, new_cont)) = visit.pop() {
                work = true;
                init = true;
                host = new_host;
                cont = new_cont;
                continue 'main;
              }
              // If available, visit a delayed location
              if delay.len() > 0 {
                for next in delay.drain(0..).rev() {
                  visit.push(next);
                }
                work = false;
                init = true;
                continue 'main;
              }
              // Otherwise, we have nothing to do
              work = false;
              init = false;
              continue 'main;
            } else {
              if stop.load(Ordering::Relaxed) {
                break;
              } else {
                for victim_tid in tids {
                  if *victim_tid != tid {
                    if let Some((new_host, new_cont)) = heap.vstk[*victim_tid].steal() {
                      //println!("[{}] stole {} {} from {}", tid, new_host, new_cont, victim_tid);
                      work = true;
                      init = true;
                      host = new_host;
                      cont = new_cont;
                      continue 'main;
                    }
                  }
                }
                backoff.snooze();
                continue 'main;
              }
            }
          }
        }
      });
    }
  });

  return load_ptr(heap, root);
}

// Normal
// ------

pub fn normal(heap: &Heap, prog: &Program, tids: &[usize], host: u64, visited: &Box<[AtomicU64]>) -> Ptr {
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
    //let term = reduce2(heap, lvars, prog, host);
    let term = reduce(heap, prog, tids, host);
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
        let arity = ask_ari(prog, term);
        for i in 0 .. arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    let rec_len = rec_locs.len(); // locations where we must recurse
    let thd_len = tids.len(); // number of available threads
    let rec_loc = &rec_locs;
    //println!("~ rec_len={} thd_len={} {}", rec_len, thd_len, show_term(heap, prog, ask_lnk(heap,host), host));
    if rec_len > 0 {
      std::thread::scope(|s| {
        // If there are more threads than rec_locs, splits threads for each rec_loc
        if thd_len >= rec_len {
          //panic!("b");
          let spt_len = thd_len / rec_len;
          let mut tids = tids;
          for (rec_num, rec_loc) in rec_loc.iter().enumerate() {
            let (rec_tids, new_tids) = tids.split_at(if rec_num == rec_len - 1 { tids.len() } else { spt_len });
            //println!("~ rec_loc {} gets {} threads", rec_loc, rec_lvars.len());
            //let new_loc;
            //if thd_len == rec_len {
              //new_loc = alloc(heap, rec_tids[0], 1);
              //move_ptr(heap, *rec_loc, new_loc);
            //} else {
              //new_loc = *rec_loc;
            //}
            let new_loc = *rec_loc;
            s.spawn(move || {
              let ptr = normal(heap, prog, rec_tids, new_loc, visited);
              //if thd_len == rec_len {
                //move_ptr(heap, new_loc, *rec_loc);
              //}
              link(heap, *rec_loc, ptr);
            });
            tids = new_tids;
          }
        // Otherwise, splits rec_locs for each thread
        } else {
          //panic!("c");
          for (thd_num, tid) in tids.iter().enumerate() {
            let min_idx = thd_num * rec_len / thd_len;
            let max_idx = if thd_num < thd_len - 1 { (thd_num + 1) * rec_len / thd_len } else { rec_len };
            //println!("~ thread {} gets rec_locs {} to {}", thd_num, min_idx, max_idx);
            s.spawn(move || {
              for idx in min_idx .. max_idx {
                let loc = rec_loc[idx];
                let lnk = normal(heap, prog, std::slice::from_ref(tid), loc, visited);
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

pub fn normalize(heap: &Heap, prog: &Program, tids: &[usize], host: u64, run_io: bool) -> Ptr {
  // TODO: rt::run_io(&mut heap, &mut lvar, &mut prog, host);
  // FIXME: reuse `visited`
  let mut cost = get_cost(heap);
  let visited = new_atomic_u64_array(HEAP_SIZE / 64);
  loop {
    let visited = new_atomic_u64_array(HEAP_SIZE / 64);
    normal(&heap, prog, tids, host, &visited);
    let new_cost = get_cost(heap);
    if new_cost != cost {
      cost = new_cost;
    } else {
      break;
    }
  }
  //println!("normalize cost: {}", get_cost(heap));
  load_ptr(heap, host)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_io(heap: &Heap, prog: &Program, tids: &[usize], host: u64) {
  fn read_input() -> String {
    let mut input = String::new();
    stdin().read_line(&mut input).expect("string");
    if let Some('\n') = input.chars().next_back() { input.pop(); }
    if let Some('\r') = input.chars().next_back() { input.pop(); }
    return input;
  }
  use std::io::{stdin,stdout,Write};
  loop {
    let term = reduce(heap, prog, tids, host); // FIXME: add parallelism
    match get_tag(term) {
      CTR => {
        let fid = get_ext(term);
        // IO.done a : (IO a)
        if fid == IO_DONE {
          let done = ask_arg(heap, term, 0);
          free(heap, 0, get_loc(term, 0), 1);
          link(heap, host, done);
          println!("");
          println!("");
          break;
        }
        // IO.do_input (String -> IO a) : (IO a)
        if fid == IO_DO_INPUT {
          let cont = ask_arg(heap, term, 0);
          let text = make_string(heap, tids[0], &read_input());
          let app0 = alloc(heap, tids[0], 2);
          link(heap, app0 + 0, cont);
          link(heap, app0 + 1, text);
          free(heap, 0, get_loc(term, 0), 1);
          let done = App(app0);
          link(heap, host, done);
        }
        // IO.do_output String (Num -> IO a) : (IO a)
        if fid == IO_DO_OUTPUT {
          if let Some(show) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            print!("{}", show);
            stdout().flush().ok();
            let cont = ask_arg(heap, term, 1);
            let app0 = alloc(heap, tids[0], 2);
            link(heap, app0 + 0, cont);
            link(heap, app0 + 1, Num(0));
            free(heap, 0, get_loc(term, 0), 2);
            let text = ask_arg(heap, term, 0);
            collect(heap, prog, 0, text);
            let done = App(app0);
            link(heap, host, done);
          } else {
            println!("Runtime type error: attempted to print a non-string.");
            println!("{}", crate::readback::as_code(heap, prog, get_loc(term, 0)));
            std::process::exit(0);
          }
        }
        // IO.do_fetch String (String -> IO a) : (IO a)
        if fid == IO_DO_FETCH {
          if let Some(url) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            let body = reqwest::blocking::get(url).unwrap().text().unwrap(); // FIXME: treat
            let cont = ask_arg(heap, term, 2);
            let app0 = alloc(heap, tids[0], 2);
            let text = make_string(heap, tids[0], &body);
            link(heap, app0 + 0, cont);
            link(heap, app0 + 1, text);
            free(heap, 0, get_loc(term, 0), 3);
            let opts = ask_arg(heap, term, 1); // FIXME: use options
            collect(heap, prog, 0, opts);
            let done = App(app0);
            link(heap, host, done);
          } else {
            println!("Runtime type error: attempted to print a non-string.");
            println!("{}", crate::readback::as_code(heap, prog, get_loc(term, 0)));
            std::process::exit(0);
          }
        }
        // IO.do_store String String (Num -> IO a) : (IO a)
        if fid == IO_DO_STORE {
          if let Some(key) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            if let Some(val) = readback_string(heap, prog, tids, get_loc(term, 1)) {
              std::fs::write(key, val).ok(); // TODO: Handle errors
              let cont = ask_arg(heap, term, 2);
              let app0 = alloc(heap, tids[0], 2);
              link(heap, app0 + 0, cont);
              link(heap, app0 + 1, Num(0));
              free(heap, 0, get_loc(term, 0), 2);
              let key = ask_arg(heap, term, 0);
              collect(heap, prog, 0, key);
              free(heap, 0, get_loc(term, 1), 2);
              let val = ask_arg(heap, term, 1);
              collect(heap, prog, 0, val);
              let done = App(app0);
              link(heap, host, done);
            } else {
              println!("Runtime type error: attempted to store a non-string.");
              println!("{}", crate::readback::as_code(heap, prog, get_loc(term, 1)));
              std::process::exit(0);
            }
          } else {
            println!("Runtime type error: attempted to store to a non-string key.");
            println!("{}", crate::readback::as_code(heap, prog, get_loc(term, 0)));
            std::process::exit(0);
          }
        }
        // IO.do_load String (String -> IO a) : (IO a)
        if fid == IO_DO_LOAD {
          if let Some(key) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            let file = std::fs::read(key).unwrap(); // TODO: Handle errors
            let file = std::str::from_utf8(&file).unwrap();
            let cont = ask_arg(heap, term, 1); 
            let text = make_string(heap, tids[0], file);
            let app0 = alloc(heap, tids[0], 2);
            link(heap, app0 + 0, cont);
            link(heap, app0 + 1, text);
            free(heap, 0, get_loc(term, 0), 2);
            let done = App(app0);
            link(heap, host, done);
          } else {
            println!("Runtime type error: attempted to read from a non-string key.");
            println!("{}", crate::readback::as_code(heap, prog, get_loc(term, 0)));
            std::process::exit(0);
          }
        }
        break;
      }
      _ => {
        break;
      }
    }
  }
}

pub fn make_string(heap: &Heap, tid: usize, text: &str) -> Ptr {
  let mut term = Ctr(STRING_NIL, 0);
  for chr in text.chars().rev() { // TODO: reverse
    let ctr0 = alloc(heap, tid, 2);
    link(heap, ctr0 + 0, Num(chr as u64));
    link(heap, ctr0 + 1, term);
    term = Ctr(STRING_CONS, ctr0);
  }
  return term;
}

// TODO: finish this
pub fn readback_string(heap: &Heap, prog: &Program, tids: &[usize], host: u64) -> Option<String> {
  let mut host = host;
  let mut text = String::new();
  loop {
    let term = reduce(heap, prog, tids, host);
    if get_tag(term) == CTR {
      let fid = get_ext(term);
      if fid == STRING_NIL {
        break;
      }
      if fid == STRING_CONS {
        let chr = reduce(heap, prog, tids, get_loc(term, 0));
        if get_tag(chr) == NUM {
          text.push(std::char::from_u32(get_num(chr) as u32).unwrap_or('?'));
          host = get_loc(term, 1);
          continue;
        } else {
          return None;
        }
      }
      return None;
    } else {
      return None;
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

pub fn show_heap(heap: &Heap) -> String {
  let mut text: String = String::new();
  for idx in 0 .. HEAP_SIZE {
    let ptr = heap.node.data[idx].load(Ordering::Relaxed);
    if ptr != 0 {
      text.push_str(&format!("{:04x} | ", idx));
      text.push_str(&show_ptr(ptr));
      text.push('\n');
    }
  }
  text
}

pub fn show_term(heap: &Heap, prog: &Program, term: Ptr, focus: u64) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    heap: &Heap,
    prog: &Program,
    term: Ptr,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    if term == 0 {
      return;
    }
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      SUP => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, load_arg(heap, term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, load_arg(heap, term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      CTR | FUN => {
        let arity = ask_ari(prog, term);
        for i in 0..arity {
          find_lets(heap, prog, load_arg(heap, term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    heap: &Heap,
    prog: &Program,
    term: Ptr,
    names: &HashMap<u64, String>,
    focus: u64,
  ) -> String {
    if term == 0 {
      return format!("<>");
    }
    let done = match get_tag(term) {
      DP0 => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("a{}", name);
        } else {
          return format!("a^{}", get_loc(term, 0));
        }
      }
      DP1 => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("b{}", name);
        } else {
          return format!("b^{}", get_loc(term, 0));
        }
      }
      VAR => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("x{}", name);
        } else {
          return format!("x^{}", get_loc(term, 0));
        }
      }
      LAM => {
        let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("<lam>")));
        format!("λ{} {}", name, go(heap, prog, load_arg(heap, term, 1), names, focus))
      }
      APP => {
        let func = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let argm = go(heap, prog, load_arg(heap, term, 1), names, focus);
        format!("({} {})", func, argm)
      }
      SUP => {
        let kind = get_ext(term);
        let func = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let argm = go(heap, prog, load_arg(heap, term, 1), names, focus);
        format!("#{}{{{} {}}}", kind, func, argm)
      }
      OP2 => {
        let oper = get_ext(term);
        let val0 = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let val1 = go(heap, prog, load_arg(heap, term, 1), names, focus);
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
          _   => "<oper>",
        };
        format!("({} {} {})", symb, val0, val1)
      }
      NUM => {
        format!("{}", get_val(term))
      }
      CTR | FUN => {
        let func = get_ext(term);
        let arit = ask_ari(prog, term);
        let args: Vec<String> = (0..arit).map(|i| go(heap, prog, load_arg(heap, term, i), names, focus)).collect();
        let name = &prog.nams.get(&func).unwrap_or(&String::from("<?>")).clone();
        format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
      ERA => "*".to_string(),
      _ => format!("<era:{}>", get_tag(term)),
    };
    if term == focus {
      format!("${}", done)
    } else {
      done
    }
  }
  find_lets(heap, prog, term, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(heap, prog, term, &names, focus);
  for (_key, pos) in itertools::sorted(lets.iter()) {
    // todo: reverse
    let what = String::from("?h");
    let kind = kinds.get(&pos).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 = if load_ptr(heap, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 = if load_ptr(heap, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!("\ndup#{}[{:x}] {} {} = {};", kind, pos, nam0, nam1, go(heap, prog, load_ptr(heap, pos + 2), &names, focus)));
  }
  text
}

pub fn debug_validate_heap(heap: &Heap) {
  for idx in 0 .. HEAP_SIZE {
    // If it is an ARG, it must be pointing to a VAR/DP0/DP1 that points to it
    let arg = heap.node.data[idx].load(Ordering::Relaxed);
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
