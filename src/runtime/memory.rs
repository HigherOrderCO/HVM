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

pub use crate::runtime::structures::allocator::{*};
pub use crate::runtime::structures::redex_bag::{*};
pub use crate::runtime::structures::visit_queue::{*};

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicI64, Ordering};
use crossbeam::utils::{CachePadded, Backoff};

// Types
// -----

pub type Ptr = u64;
pub type AtomicPtr = AtomicU64;
pub type ArityMap = crate::runtime::U64Map<u64>;

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
  pub tids: usize,
  pub node: Allocator,
  pub lvar: Box<[CachePadded<LocalVars>]>,
  pub lock: Box<[AtomicU8]>,
  pub vstk: Box<[VisitQueue]>,
  pub aloc: Box<[Box<[AtomicU64]>]>,
  pub rbag: RedexBag,
}

// Pointer Constructors
// --------------------

pub const VAL: u64 = 1;
pub const EXT: u64 = 0x100000000;
pub const TAG: u64 = 0x1000000000000000;

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

pub const NUM_MASK: u64 = 0xFFF_FFFF_FFFF_FFFF;

// Pointer Constructors
// --------------------

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

pub fn arity_of(arit: &ArityMap, lnk: Ptr) -> u64 {
  return *arit.get(&get_ext(lnk)).unwrap_or(&0);
}

// Pointers
// --------

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

// Heap Constructors
// -----------------

pub fn new_atomic_u8_array(size: usize) -> Box<[AtomicU8]> {
  return unsafe { Box::from_raw(AtomicU8::from_mut_slice(Box::leak(vec![0xFFu8; size].into_boxed_slice()))) }
}

pub fn new_atomic_u64_array(size: usize) -> Box<[AtomicU64]> {
  return unsafe { Box::from_raw(AtomicU64::from_mut_slice(Box::leak(vec![0u64; size].into_boxed_slice()))) }
}

pub fn new_heap(size: usize, tids: usize) -> Heap {
  let mut lvar = vec![];
  for tid in 0 .. tids {
    lvar.push(CachePadded::new(LocalVars {
      tid: tid,
      used: AtomicI64::new(0),
      next: AtomicU64::new((size / tids * (tid + 0)) as u64),
      amin: AtomicU64::new((size / tids * (tid + 0)) as u64),
      amax: AtomicU64::new((size / tids * (tid + 1)) as u64),
      dups: AtomicU64::new(((1 << 28) / tids * tid) as u64),
      cost: AtomicU64::new(0),
    }))
  }
  let node = Allocator::new(tids);
  let lvar = lvar.into_boxed_slice();
  let lock = new_atomic_u8_array(size);
  let rbag = RedexBag::new(tids);
  let aloc = (0 .. tids).map(|x| new_atomic_u64_array(1 << 20)).collect::<Vec<Box<[AtomicU64]>>>().into_boxed_slice();
  let vstk = (0 .. tids).map(|x| VisitQueue::new()).collect::<Vec<VisitQueue>>().into_boxed_slice();
  return Heap { tids, node, lvar, lock, rbag, aloc, vstk };
}

// Allocator
// ---------

pub fn alloc(heap: &Heap, tid: usize, arity: u64) -> u64 {
  //heap.node.alloc(tid, arity)
  unsafe {
    let lvar = &heap.lvar[tid];
    if arity == 0 {
      0
    } else {
      let mut length = 0;
      //let mut count = 0;
      loop {
        //count += 1;
        //if tid == 9 && count > 5000000 {
          //println!("[9] slow-alloc {} | {}", count, *lvar.next.as_mut_ptr());
        //}
        // Loads value on cursor
        let val = heap.node.data.get_unchecked(*lvar.next.as_mut_ptr() as usize).load(Ordering::Relaxed);
        // If it is empty, increment length
        if val == 0 {
          length += 1;
        // Otherwise, reset length
        } else {
          length = 0;
        };
        // Moves cursor right
        *lvar.next.as_mut_ptr() += 1;
        // If it is out of bounds, warp around
        if *lvar.next.as_mut_ptr() >= *lvar.amax.as_mut_ptr() {
          length = 0;
          *lvar.next.as_mut_ptr() = *lvar.amin.as_mut_ptr();
        }
        // If length equals arity, allocate that space
        if length == arity {
          //println!("[{}] return", lvar.tid);
          //println!("[{}] alloc {} at {}", lvar.tid, arity, lvar.next - length);
          //lvar.used.fetch_add(arity as i64, Ordering::Relaxed);
          //if tid == 9 && count > 50000 {
            //println!("[{}] allocated {}! {}", 9, length, *lvar.next.as_mut_ptr() - length);
          //}
          return *lvar.next.as_mut_ptr() - length;
        }
      }
    }
  }
}

pub fn free(heap: &Heap, tid: usize, loc: u64, arity: u64) {
  //heap.node.free(tid, loc, arity)
  for i in 0 .. arity {
    unsafe { heap.node.data.get_unchecked((loc + i) as usize) }.store(0, Ordering::Relaxed);
  }
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
pub fn atomic_subst(heap: &Heap, arit: &ArityMap, tid: usize, var: Ptr, val: Ptr) {
  //let mut count = 0;
  loop {
    let arg_ptr = load_ptr(heap, get_loc(var, get_tag(var) & 0x01));
    //count = count + 1;
    //if count > 100000 {
      //println!("[{}] aaaa {} | {} {}", tid, count, crate::runtime::show_ptr(var), crate::runtime::show_ptr(val));
      //link(heap, get_loc(arg_ptr, 0), val);
      //return;
    //}
    if get_tag(arg_ptr) == ARG {
      match atomic_relink(heap, get_loc(arg_ptr, 0), var, val) {
        Ok(_) => { return; }
        Err(_) => { continue; }
      }
    }
    if get_tag(arg_ptr) == ERA {
      collect(heap, arit, tid, val); // safe, since `val` is owned by this thread
      return;
    }
  }
}

// Locks
// -----

pub const LOCK_OPEN : u8 = 0xFF;

pub fn acquire_lock(heap: &Heap, tid: usize, term: Ptr) -> Result<u8, u8> {
  let locker = unsafe { heap.lock.get_unchecked(get_loc(term, 0) as usize) };
  locker.compare_exchange_weak(LOCK_OPEN, tid as u8, Ordering::Acquire, Ordering::Relaxed)
}

pub fn release_lock(heap: &Heap, tid: usize, term: Ptr) {
  let locker = unsafe { heap.lock.get_unchecked(get_loc(term, 0) as usize) };
  locker.store(LOCK_OPEN, Ordering::Release)
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

pub fn collect(heap: &Heap, arit: &ArityMap, tid: usize, term: Ptr) {
  let mut coll = Vec::new();
  let mut next = term;
  loop {
    let term = next;
    match get_tag(term) {
      DP0 => {
        link(heap, get_loc(term, 0), Era());
        if acquire_lock(heap, tid, term).is_ok() {
          if get_tag(load_arg(heap, term, 1)) == ERA {
            collect(heap, arit, tid, load_arg(heap, term, 2));
            free(heap, tid, get_loc(term, 0), 3);
          }
          release_lock(heap, tid, term);
        }
      }
      DP1 => {
        link(heap, get_loc(term, 1), Era());
        if acquire_lock(heap, tid, term).is_ok() {
          if get_tag(load_arg(heap, term, 0)) == ERA {
            collect(heap, arit, tid, load_arg(heap, term, 2));
            free(heap, tid, get_loc(term, 0), 3);
          }
          release_lock(heap, tid, term);
        }
      }
      VAR => {
        link(heap, get_loc(term, 0), Era());
      }
      LAM => {
        atomic_subst(heap, arit, tid, Var(get_loc(term,0)), Era());
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
        let arity = arity_of(arit, term);
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
