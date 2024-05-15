use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::alloc::{alloc, dealloc, Layout};
use std::mem;

//ok

// Runtime
// =======

// Types
pub type Tag  = u8;  // Tag  ::= 3-bit (rounded up to u8)
pub type Lab  = u32; // Lab  ::= 29-bit (rounded up to u32)
pub type Val  = u32; // Val  ::= 29-bit (rounded up to u32)
pub type Rule = u8;  // Rule ::= 8-bit (fits a u8)


// Port
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct Port(pub Val);

// Pair
pub struct Pair(pub u64);

// Atomics
pub type AVal = AtomicU32;
pub struct APort(pub AVal);
pub struct APair(pub AtomicU64);

// Number
pub struct Numb(pub Val);

// Tags
pub const VAR : Tag = 0x0; // variable
pub const REF : Tag = 0x1; // reference
pub const ERA : Tag = 0x2; // eraser
pub const NUM : Tag = 0x3; // number
pub const CON : Tag = 0x4; // constructor
pub const DUP : Tag = 0x5; // duplicator
pub const OPR : Tag = 0x6; // operator
pub const SWI : Tag = 0x7; // switch

// Rules
pub const LINK : Rule = 0x0;
pub const CALL : Rule = 0x1;
pub const VOID : Rule = 0x2;
pub const ERAS : Rule = 0x3;
pub const ANNI : Rule = 0x4;
pub const COMM : Rule = 0x5;
pub const OPER : Rule = 0x6;
pub const SWIT : Rule = 0x7;

// Numbs
pub const SYM : Tag = 0x0;
pub const U24 : Tag = 0x1;
pub const I24 : Tag = 0x2;
pub const F24 : Tag = 0x3;
pub const ADD : Tag = 0x4;
pub const SUB : Tag = 0x5;
pub const MUL : Tag = 0x6;
pub const DIV : Tag = 0x7;
pub const REM : Tag = 0x8;
pub const EQ  : Tag = 0x9;
pub const NEQ : Tag = 0xA;
pub const LT  : Tag = 0xB;
pub const GT  : Tag = 0xC;
pub const AND : Tag = 0xD;
pub const OR  : Tag = 0xE;
pub const XOR : Tag = 0xF;

// Constants
pub const FREE : Port = Port(0x0);
pub const ROOT : Port = Port(0xFFFFFFF8);
pub const NONE : Port = Port(0xFFFFFFFF);

// RBag
pub struct RBag {
  pub lo: Vec<Pair>,
  pub hi: Vec<Pair>,
}

// Global Net
pub struct GNet<'a> {
  pub nlen: usize, // length of the node buffer
  pub vlen: usize, // length of the vars buffer
  pub node: &'a mut [APair], // node buffer
  pub vars: &'a mut [APort], // vars buffer
  pub itrs: AtomicU64, // interaction count
}

// Thread Memory
pub struct TMem {
  pub tid: u32, // thread id
  pub tids: u32, // thread count
  pub tick: u32, // tick counter
  pub itrs: u32, // interaction count
  pub nput: usize, // next node allocation index
  pub vput: usize, // next vars allocation index
  pub nloc: Vec<usize>, // allocated node locations
  pub vloc: Vec<usize>, // allocated vars locations
  pub rbag: RBag, // local redex bag
}

// Top-Level Definition
pub struct Def {
  pub name: String, // def name
  pub safe: bool, // has no dups
  pub root: Port, // root port
  pub rbag: Vec<Pair>, // def redex bag
  pub node: Vec<Pair>, // def node buffer
  pub vars: usize, // def vars count
}

// Book of Definitions
pub struct Book {
  pub defs: Vec<Def>,
}

impl Port {
  pub fn new(tag: Tag, val: Val) -> Self {
    Port((val << 3) | tag as Val)
  }

  pub fn get_tag(&self) -> Tag {
    (self.0 & 7) as Tag
  }

  pub fn get_val(&self) -> Val {
    self.0 >> 3
  }

  pub fn is_nod(&self) -> bool {
    self.get_tag() >= CON
  }

  pub fn is_var(&self) -> bool {
    self.get_tag() == VAR
  }

  pub fn get_rule(a: Port, b: Port) -> Rule {
    const TABLE: [[Rule; 8]; 8] = [
      //VAR  REF  ERA  NUM  CON  DUP  OPR  SWI
      [LINK,LINK,LINK,LINK,LINK,LINK,LINK,LINK], // VAR
      [LINK,VOID,VOID,VOID,CALL,CALL,CALL,CALL], // REF
      [LINK,VOID,VOID,VOID,ERAS,ERAS,ERAS,ERAS], // ERA
      [LINK,VOID,VOID,VOID,ERAS,ERAS,OPER,SWIT], // NUM
      [LINK,CALL,ERAS,ERAS,ANNI,COMM,COMM,COMM], // CON
      [LINK,CALL,ERAS,ERAS,COMM,ANNI,COMM,COMM], // DUP
      [LINK,CALL,ERAS,OPER,COMM,COMM,ANNI,COMM], // OPR
      [LINK,CALL,ERAS,SWIT,COMM,COMM,COMM,ANNI], // SWI
    ];
    return TABLE[a.get_tag() as usize][b.get_tag() as usize];
  }

  pub fn should_swap(a: Port, b: Port) -> bool {
    b.get_tag() < a.get_tag()
  }

  pub fn is_high_priority(rule: Rule) -> bool {
    (0b00011101 >> rule) & 1 != 0
  }

  pub fn adjust_port(&self, tm: &TMem) -> Port {
    let tag = self.get_tag();
    let val = self.get_val();
    if self.is_nod() {
      Port::new(tag, tm.nloc[val as usize] as u32)
    } else if self.is_var() {
      Port::new(tag, tm.vloc[val as usize] as u32)
    } else {
      Port::new(tag, val)
    }
  }

}

impl Pair {
  pub fn new(fst: Port, snd: Port) -> Self {
    Pair(((snd.0 as u64) << 32) | fst.0 as u64)
  }

  pub fn get_fst(&self) -> Port {
    Port((self.0 & 0xFFFFFFFF) as u32)
  }

  pub fn get_snd(&self) -> Port {
    Port((self.0 >> 32) as u32)
  }

  pub fn adjust_pair(&self, tm: &TMem) -> Pair {
    let p1 = self.get_fst().adjust_port(tm);
    let p2 = self.get_snd().adjust_port(tm);
    Pair::new(p1, p2)
  }

  pub fn set_par_flag(&self) -> Self {
    let p1 : Port = self.get_fst();
    let p2 : Port = self.get_snd();
    if p1.get_tag() == REF {
      return Pair::new(Port::new(p1.get_tag(), p1.get_val() | 0x10000000), p2);
    } else {
      return Pair::new(p1, p2);
    }
  }

  pub fn get_par_flag(&self) -> bool {
    let p1 : Port = self.get_fst();
    if p1.get_tag() == REF {
      return p1.get_val() >> 28 == 1;
    } else {
      return false;
    }
  }
}

impl Numb {

  // SYM: a symbolic operator

  pub fn new_sym(val: u32) -> Self {
    Numb(((val & 0xF) << 4) as Val | (SYM as Val))
  }

  pub fn get_sym(&self) -> u32 {
    ((self.0 >> 4) & 0xF) as u32
  }

  // U24: unsigned 24-bit integer
  
  pub fn new_u24(val: u32) -> Self {
    Numb(((val & 0xFFFFFF) << 4) as Val | (U24 as Val))
  }

  pub fn get_u24(&self) -> u32 {
    ((self.0 >> 4) & 0xFFFFFF) as u32
  }

  // I24: signed 24-bit integer

  pub fn new_i24(val: i32) -> Self {
    Numb((((val as u32) & 0xFFFFFF) << 4) as Val | (I24 as Val))
  }

  pub fn get_i24(&self) -> i32 {
    (((self.0 >> 4) & 0xFFFFFF) as i32) << 8 >> 8
  }

  // F24: 24-bit float
  
  pub fn new_f24(val: f32) -> Self {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 0x1;
    let expo = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x7FFFFF;
    assert!(expo >= -63 && expo <= 63);
    let bits = (expo + 63) as u32;
    let bits = (sign << 23) | (bits << 16) | (mant >> 7);
    Numb((bits << 4) as Val | (F24 as Val))
  }

  pub fn get_f24(&self) -> f32 {
    let bits = (self.0 >> 4) & 0xFFFFFF;
    let sign = (bits >> 23) & 0x1;
    let expo = (bits >> 16) & 0x7F;
    let mant = bits & 0xFFFF;
    let iexp = (expo as i32) - 63;
    let bits = (sign << 31) | (((iexp + 127) as u32) << 23) | (mant << 7);
    let bits = if mant == 0 && iexp == -63 { sign << 31 } else { bits };
    f32::from_bits(bits)
  }

  // Gets the numeric type.

  pub fn get_typ(&self) -> Tag {
    return (self.0 & 0xF) as Tag;
  }

  // Flip flag.

  pub fn get_flp(&self) -> bool {
    return (self.0 >> 28) & 1 == 1;
  }

  pub fn set_flp(&self) -> Self {
    return Numb(self.0 | 0x1000_0000);
  }

  pub fn flp_flp(&self) -> Self {
    Numb(self.0 ^ 0x1000_0000)
  }

  // Partial application.
  pub fn partial(a: Self, b: Self) -> Self {
    return Numb(b.0 & 0xFFFFFFF0 | a.get_sym());
  }

  pub fn operate(mut a: Self, mut b: Self) -> Self {
    //println!("operate {} {}", crate::ast::Numb(a.0).show(), crate::ast::Numb(b.0).show());
    if a.get_flp() ^ b.get_flp() {
      (a,b) = (b,a);
    }
    let at = a.get_typ();
    let bt = b.get_typ();
    if at == SYM && bt == SYM {
      return Numb::new_u24(0);
    }
    if at == SYM && bt != SYM {
      return Numb::partial(a, b);
    }
    if at != SYM && bt == SYM {
      return Numb::partial(b, a);
    }
    if at >= ADD && bt >= ADD {
      return Numb::new_u24(0);
    }
    if at < ADD && bt < ADD {
      return Numb::new_u24(0);
    }
    let op = if at >= ADD { at } else { bt };
    let ty = if at >= ADD { bt } else { at };
    match ty {
      U24 => {
        let av = a.get_u24();
        let bv = b.get_u24();
        match op {
          ADD => Numb::new_u24(av.wrapping_add(bv)),
          SUB => Numb::new_u24(av.wrapping_sub(bv)),
          MUL => Numb::new_u24(av.wrapping_mul(bv)),
          DIV => Numb::new_u24(av.wrapping_div(bv)),
          REM => Numb::new_u24(av.wrapping_rem(bv)),
          EQ  => Numb::new_u24((av == bv) as u32),
          NEQ => Numb::new_u24((av != bv) as u32),
          LT  => Numb::new_u24((av <  bv) as u32),
          GT  => Numb::new_u24((av >  bv) as u32),
          AND => Numb::new_u24(av & bv),
          OR  => Numb::new_u24(av | bv),
          XOR => Numb::new_u24(av ^ bv),
          _   => unreachable!(),
        }
      }
      I24 => {
        let av = a.get_i24();
        let bv = b.get_i24();
        match op {
          ADD => Numb::new_i24(av.wrapping_add(bv)),
          SUB => Numb::new_i24(av.wrapping_sub(bv)),
          MUL => Numb::new_i24(av.wrapping_mul(bv)),
          DIV => Numb::new_i24(av.wrapping_div(bv)),
          REM => Numb::new_i24(av.wrapping_rem(bv)),
          EQ  => Numb::new_i24((av == bv) as i32),
          NEQ => Numb::new_i24((av != bv) as i32),
          LT  => Numb::new_i24((av <  bv) as i32),
          GT  => Numb::new_i24((av >  bv) as i32),
          AND => Numb::new_i24(av & bv),
          OR  => Numb::new_i24(av | bv),
          XOR => Numb::new_i24(av ^ bv),
          _   => unreachable!(),
        }
      }
      F24 => {
        let av = a.get_f24();
        let bv = b.get_f24();
        match op {
          ADD => Numb::new_f24(av + bv),
          SUB => Numb::new_f24(av - bv),
          MUL => Numb::new_f24(av * bv),
          DIV => Numb::new_f24(av / bv),
          REM => Numb::new_f24(av % bv),
          EQ  => Numb::new_u24((av == bv) as u32),
          NEQ => Numb::new_u24((av != bv) as u32),
          LT  => Numb::new_u24((av <  bv) as u32),
          GT  => Numb::new_u24((av >  bv) as u32),
          AND => Numb::new_f24(av.atan2(bv)),
          OR  => Numb::new_f24(bv.log(av)),
          XOR => Numb::new_f24(av.powf(bv)),
          _   => unreachable!(),
        }
      }
      _ => Numb::new_u24(0),
    }
  }
}

impl RBag {
  pub fn new() -> Self {
    RBag {
      lo: Vec::new(),
      hi: Vec::new(),
    }
  }

  pub fn push_redex(&mut self, redex: Pair) {
    let rule = Port::get_rule(redex.get_fst(), redex.get_snd());
    if Port::is_high_priority(rule) {
      self.hi.push(redex);
    } else {
      self.lo.push(redex);
    }
  }

  pub fn pop_redex(&mut self) -> Option<Pair> {
    if !self.hi.is_empty() {
      self.hi.pop()
    } else {
      self.lo.pop()
    }
  }

  pub fn len(&self) -> usize {
    self.lo.len() + self.hi.len()
  }

  pub fn has_highs(&self) -> bool {
    !self.hi.is_empty()
  }
}

impl<'a> GNet<'a> {
  pub fn new(nlen: usize, vlen: usize) -> Self {
    let nlay = Layout::array::<APair>(nlen).unwrap();
    let vlay = Layout::array::<APort>(vlen).unwrap();
    let nptr = unsafe { alloc(nlay) as *mut APair };
    let vptr = unsafe { alloc(vlay) as *mut APort };
    let node = unsafe { std::slice::from_raw_parts_mut(nptr, nlen) };
    let vars = unsafe { std::slice::from_raw_parts_mut(vptr, vlen) };
    GNet { nlen, vlen, node, vars, itrs: AtomicU64::new(0) }
  }

  pub fn node_create(&self, loc: usize, val: Pair) {
    self.node[loc].0.store(val.0, Ordering::Relaxed);
  }

  pub fn vars_create(&self, var: usize, val: Port) {
    self.vars[var].0.store(val.0, Ordering::Relaxed);
  }

  pub fn node_load(&self, loc: usize) -> Pair {
    Pair(self.node[loc].0.load(Ordering::Relaxed))
  }

  pub fn vars_load(&self, var: usize) -> Port {
    Port(self.vars[var].0.load(Ordering::Relaxed) as u32)
  }

  pub fn node_store(&self, loc: usize, val: Pair) {
    self.node[loc].0.store(val.0, Ordering::Relaxed);
  }

  pub fn vars_store(&self, var: usize, val: Port) {
    self.vars[var].0.store(val.0, Ordering::Relaxed);
  }
  
  pub fn node_exchange(&self, loc: usize, val: Pair) -> Pair {
    Pair(self.node[loc].0.swap(val.0, Ordering::Relaxed))
  }

  pub fn vars_exchange(&self, var: usize, val: Port) -> Port {
    Port(self.vars[var].0.swap(val.0, Ordering::Relaxed) as u32)
  }

  pub fn node_take(&self, loc: usize) -> Pair {
    self.node_exchange(loc, Pair(0))
  }

  pub fn vars_take(&self, var: usize) -> Port {
    self.vars_exchange(var, Port(0))
  }

  pub fn is_node_free(&self, loc: usize) -> bool {
    self.node_load(loc).0 == 0
  }

  pub fn is_vars_free(&self, var: usize) -> bool {
    self.vars_load(var).0 == 0
  }

  pub fn enter(&self, mut var: Port) -> Port {
    // While `B` is VAR: extend it (as an optimization)
    while var.get_tag() == VAR {
      // Takes the current `B` substitution as `B'`
      let val = self.vars_exchange(var.get_val() as usize, NONE);
      // If there was no `B'`, stop, as there is no extension
      if val == NONE || val == Port(0) {
        break;
      }
      // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
      self.vars_take(var.get_val() as usize);
      var = val;
    }
    return var;
  }

}

impl<'a> Drop for GNet<'a> {
  fn drop(&mut self) {
    let nlay = Layout::array::<APair>(self.nlen).unwrap();
    let vlay = Layout::array::<APair>(self.vlen).unwrap();
    unsafe {
      dealloc(self.node.as_mut_ptr() as *mut u8, nlay);
      dealloc(self.vars.as_mut_ptr() as *mut u8, vlay);
    }
  }

}

impl TMem {
  // TODO: implement a TMem::new() fn
  pub fn new(tid: u32, tids: u32) -> Self {
    TMem {
      tid,
      tids,
      tick: 0,
      itrs: 0,
      nput: 0,
      vput: 0,
      nloc: vec![0; 32],
      vloc: vec![0; 32],
      rbag: RBag::new(),
    }
  }
  
  pub fn node_alloc(&mut self, net: &GNet, num: usize) -> usize {
    let mut got = 0;
    for _ in 0..net.nlen {
      self.nput += 1; // index 0 reserved
      if self.nput < net.nlen-1 || net.is_node_free(self.nput % net.nlen) {
        self.nloc[got] = self.nput % net.nlen;
        got += 1;
        //println!("ALLOC NODE {} {}", got, self.nput);
      }
      if got >= num {
        break;
      }
    }
    return got
  }

  pub fn vars_alloc(&mut self, net: &GNet, num: usize) -> usize {
    let mut got = 0;
    for _ in 0..net.vlen {
      self.vput += 1; // index 0 reserved for FREE
      if self.vput < net.vlen-1 || net.is_vars_free(self.vput % net.vlen) {
        self.vloc[got] = self.vput % net.nlen;
        //println!("ALLOC VARS {} {}", got, self.vput);
        got += 1;
      }
      if got >= num {
        break;
      }
    }
    got
  }

  pub fn get_resources(&mut self, net: &GNet, _need_rbag: usize, need_node: usize, need_vars: usize) -> bool {
    let got_node = self.node_alloc(net, need_node);
    let got_vars = self.vars_alloc(net, need_vars);
    got_node >= need_node && got_vars >= need_vars
  }

  // Atomically Links `A ~ B`.
  pub fn link(&mut self, net: &GNet, a: Port, b: Port) {
    //println!("link {} ~ {}", a.show(), b.show());
    let mut a = a;
    let mut b = b;

    // Attempts to directionally point `A ~> B`
    loop {
      // If `A` is NODE: swap `A` and `B`, and continue
      if a.get_tag() != VAR && a.get_tag() == VAR {
        let x = a; a = b; b = x;
      }

      // If `A` is NODE: create the `A ~ B` redex
      if a.get_tag() != VAR {
        self.rbag.push_redex(Pair::new(a, b));
        break;
      }

      // While `B` is VAR: extend it (as an optimization)
      b = net.enter(b);

      // Since `A` is VAR: point `A ~> B`.
      if true {
        // Stores `A -> B`, taking the current `A` subst as `A'`
        let a_ = net.vars_exchange(a.get_val() as usize, b);
        // If there was no `A'`, stop, as we lost B's ownership
        if a_ == NONE {
          break;
        }
        // Otherwise, delete `A` (we own both) and link `A' ~ B`
        net.vars_take(a.get_val() as usize);
        a = a_;
      }
    }
  }

  // Links `A ~ B` (as a pair).
  pub fn link_pair(&mut self, net: &GNet, ab: Pair) {
    self.link(net, ab.get_fst(), ab.get_snd());
    //println!("link_pair {:016X}", ab.0);
  }

  // The Link Interaction.
  pub fn interact_link(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 1, 0, 0) {
      return false;
    }

    // Links.
    self.link_pair(net, Pair::new(a, b));
    
    true
  }

  // The Call Interaction.
  pub fn interact_call(&mut self, net: &GNet, a: Port, b: Port, book: &Book) -> bool {
    let fid = (a.get_val() as usize) & 0xFFFFFFF;
    let def = &book.defs[fid];

    // Copy Optimization.
    if def.safe && b.get_tag() == DUP {
      return self.interact_eras(net, a, b);
    }

    // Allocates needed nodes and vars.
    if !self.get_resources(net, def.rbag.len() + 1, def.node.len(), def.vars as usize) {
      return false;
    }

    // Stores new vars.
    for i in 0..def.vars {
      net.vars_create(self.vloc[i], NONE);
      //println!("vars_create vars_loc[{:04X}] {:04X}", i, self.vloc[i]);
    }

    // Stores new nodes.
    for i in 0..def.node.len() {
      net.node_create(self.nloc[i], def.node[i].adjust_pair(self));
      //println!("node_create node_loc[{:04X}] {:016X}", i-1, def.node[i].0);
    }

    // Links.
    for pair in &def.rbag {
      self.link_pair(net, pair.adjust_pair(self));
    }
    self.link_pair(net, Pair::new(def.root.adjust_port(self), b));
  
    true
  }

  // The Void Interaction.
  pub fn interact_void(&mut self, _net: &GNet, _a: Port, _b: Port) -> bool {
    true
  }

  // The Eras Interaction.
  pub fn interact_eras(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 2, 0, 0) {
      return false;
    }

    // Checks availability
    if net.node_load(b.get_val() as usize).0 == 0 {
      return false;
    }

    // Loads ports.
    let b_ = net.node_exchange(b.get_val() as usize, Pair(0));
    let b1 = b_.get_fst();
    let b2 = b_.get_snd();

    // Links.
    self.link_pair(net, Pair::new(a, b1));
    self.link_pair(net, Pair::new(a, b2));
    
    true
  }

  // The Anni Interaction.
  pub fn interact_anni(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 2, 0, 0) {
      return false;
    }

    // Checks availability
    if net.node_load(a.get_val() as usize).0 == 0 || net.node_load(b.get_val() as usize).0 == 0 {
      return false;
    }

    // Loads ports.
    let a_ = net.node_take(a.get_val() as usize);
    let a1 = a_.get_fst();
    let a2 = a_.get_snd();
    let b_ = net.node_take(b.get_val() as usize);
    let b1 = b_.get_fst();
    let b2 = b_.get_snd();

    // Links.
    self.link_pair(net, Pair::new(a1, b1));
    self.link_pair(net, Pair::new(a2, b2));

    return true;
  }

  // The Comm Interaction.
  pub fn interact_comm(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 4, 4, 4) {
      return false;
    }

    // Checks availability
    if net.node_load(a.get_val() as usize).0 == 0 || net.node_load(b.get_val() as usize).0 == 0 {
      return false;
    }

    // Loads ports.
    let a_ = net.node_take(a.get_val() as usize);
    let a1 = a_.get_fst();
    let a2 = a_.get_snd();
    let b_ = net.node_take(b.get_val() as usize);
    let b1 = b_.get_fst();
    let b2 = b_.get_snd();
      
    // Stores new vars.
    net.vars_create(self.vloc[0], NONE);
    net.vars_create(self.vloc[1], NONE);
    net.vars_create(self.vloc[2], NONE);
    net.vars_create(self.vloc[3], NONE);

    // Stores new nodes.
    net.node_create(self.nloc[0], Pair::new(Port::new(VAR, self.vloc[0] as u32), Port::new(VAR, self.vloc[1] as u32)));
    net.node_create(self.nloc[1], Pair::new(Port::new(VAR, self.vloc[2] as u32), Port::new(VAR, self.vloc[3] as u32)));
    net.node_create(self.nloc[2], Pair::new(Port::new(VAR, self.vloc[0] as u32), Port::new(VAR, self.vloc[2] as u32)));
    net.node_create(self.nloc[3], Pair::new(Port::new(VAR, self.vloc[1] as u32), Port::new(VAR, self.vloc[3] as u32)));

    // Links.
    self.link_pair(net, Pair::new(Port::new(b.get_tag(), self.nloc[0] as u32), a1));
    self.link_pair(net, Pair::new(Port::new(b.get_tag(), self.nloc[1] as u32), a2));
    self.link_pair(net, Pair::new(Port::new(a.get_tag(), self.nloc[2] as u32), b1));
    self.link_pair(net, Pair::new(Port::new(a.get_tag(), self.nloc[3] as u32), b2));
    
    true
  }

  // The Oper Interaction.
  pub fn interact_oper(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 1, 1, 0) {
      return false;
    }

    // Checks availability
    if net.node_load(b.get_val() as usize).0 == 0 {
      return false;
    }

    // Loads ports.
    let av = a.get_val();
    let b_ = net.node_take(b.get_val() as usize);
    let b1 = b_.get_fst();
    let b2 = net.enter(b_.get_snd());
     
    // Performs operation.
    if b1.get_tag() == NUM {
      let bv = b1.get_val();
      let cv = Numb::operate(Numb(av), Numb(bv));
      self.link_pair(net, Pair::new(Port::new(NUM, cv.0), b2));
    } else {
      net.node_create(self.nloc[0], Pair::new(Port::new(a.get_tag(), Numb(a.get_val()).flp_flp().0), b2));
      self.link_pair(net, Pair::new(b1, Port::new(OPR, self.nloc[0] as u32)));
    }

    true
  }

  // The Swit Interaction.
  pub fn interact_swit(&mut self, net: &GNet, a: Port, b: Port) -> bool {
    // Allocates needed nodes and vars.
    if !self.get_resources(net, 1, 2, 0) {
      return false;
    }
  
    // Checks availability
    if net.node_load(b.get_val() as usize).0 == 0 {
      return false;
    }

    // Loads ports.
    let av = Numb(a.get_val()).get_u24();
    let b_ = net.node_take(b.get_val() as usize);
    let b1 = b_.get_fst();
    let b2 = b_.get_snd();
 
    // Stores new nodes.
    if av == 0 {
      net.node_create(self.nloc[0], Pair::new(b2, Port::new(ERA,0)));
      self.link_pair(net, Pair::new(Port::new(CON, self.nloc[0] as u32), b1));
    } else {
      net.node_create(self.nloc[0], Pair::new(Port::new(ERA,0), Port::new(CON, self.nloc[1] as u32)));
      net.node_create(self.nloc[1], Pair::new(Port::new(NUM, Numb::new_u24(av-1).0), b2));
      self.link_pair(net, Pair::new(Port::new(CON, self.nloc[0] as u32), b1));
    }

    true
  }

  // Pops a local redex and performs a single interaction.
  pub fn interact(&mut self, net: &GNet, book: &Book) -> bool {
    // Pops a redex.
    let redex = match self.rbag.pop_redex() {
      Some(redex) => redex,
      None => return true, // If there is no redex, stop
    };

    // Gets redex ports A and B.
    let mut a = redex.get_fst();
    let mut b = redex.get_snd();

    // Gets the rule type.
    let mut rule = Port::get_rule(a, b);

    // Used for root redex.
    if a.get_tag() == REF && b == ROOT {
      rule = CALL;
    // Swaps ports if necessary.
    } else if Port::should_swap(a,b) {
      let x = a; a = b; b = x;
    }

    let success = match rule {
      LINK => self.interact_link(net, a, b),
      CALL => self.interact_call(net, a, b, book),
      VOID => self.interact_void(net, a, b),
      ERAS => self.interact_eras(net, a, b),
      ANNI => self.interact_anni(net, a, b),
      COMM => self.interact_comm(net, a, b),
      OPER => self.interact_oper(net, a, b),
      SWIT => self.interact_swit(net, a, b),
      _    => panic!("Invalid rule"),
    };

    // If error, pushes redex back.
    if !success {
      self.rbag.push_redex(redex);
      false
    // Else, increments the interaction count.
    } else if rule != LINK {
      self.itrs += 1;
      true
    } else {
      true
    }
  }

  pub fn evaluator(&mut self, net: &GNet, book: &Book) {
    // Increments the tick
    self.tick += 1;

    // DEBUG:
    //let mut max_rlen = 0;
    //let mut max_nlen = 0;
    //let mut max_vlen = 0;

    // Performs some interactions
    while self.rbag.len() > 0 {
      self.interact(net, book);

      // DEBUG:
      //println!("{}{}", self.rbag.show(), net.show());
      //println!("");
      //let rlen = self.rbag.lo.len() + self.rbag.hi.len();
      //let mut nlen = 0;
      //for i in 0 .. 256 {
        //if net.node_load(i).0 != 0 {
          //nlen += 1;
        //}
      //}
      //let mut vlen = 0;
      //for i in 0..256 {
        //if net.vars_load(i).0 != 0 {
          //vlen += 1;
        //}
      //}
      //max_rlen = max_rlen.max(rlen);
      //max_nlen = max_nlen.max(nlen);
      //max_vlen = max_vlen.max(vlen);

    }

    // DEBUG:
    //println!("MAX_RLEN: {}", max_rlen);
    //println!("MAX_NLEN: {}", max_nlen);
    //println!("MAX_VLEN: {}", max_vlen);

    net.itrs.fetch_add(self.itrs as u64, Ordering::Relaxed);
    self.itrs = 0;
  }
}

// Serialization
// -------------

impl Book {
  pub fn to_buffer(&self, buf: &mut Vec<u8>) {
    // Writes the number of defs
    buf.extend_from_slice(&(self.defs.len() as u32).to_ne_bytes());

    // For each def
    for (fid, def) in self.defs.iter().enumerate() {
      // Writes the safe flag
      buf.extend_from_slice(&(fid as u32).to_ne_bytes());

      // Writes the name
      let name_bytes = def.name.as_bytes();
      buf.extend_from_slice(&name_bytes[..32.min(name_bytes.len())]);
      buf.resize(buf.len() + (32 - name_bytes.len()), 0);

      // Writes the safe flag
      buf.extend_from_slice(&(def.safe as u32).to_ne_bytes());

      // Writes the rbag length
      buf.extend_from_slice(&(def.rbag.len() as u32).to_ne_bytes());
      
      // Writes the node length
      buf.extend_from_slice(&(def.node.len() as u32).to_ne_bytes());

      // Writes the vars length
      buf.extend_from_slice(&(def.vars as u32).to_ne_bytes());
      
      // Writes the root
      buf.extend_from_slice(&def.root.0.to_ne_bytes());

      // Writes the rbag buffer
      for pair in &def.rbag {
        buf.extend_from_slice(&pair.0.to_ne_bytes());
      }

      // Writes the node buffer
      for pair in &def.node {
        buf.extend_from_slice(&pair.0.to_ne_bytes());
      }
    }
  }
}

// Debug
// -----

impl Port {
  pub fn show(&self) -> String {
    match self.get_tag() {
      VAR => format!("VAR:{:08X}", self.get_val()),
      REF => format!("REF:{:08X}", self.get_val()),
      ERA => format!("ERA:{:08X}", self.get_val()),
      NUM => format!("NUM:{:08X}", self.get_val()),
      CON => format!("CON:{:08X}", self.get_val()),
      DUP => format!("DUP:{:08X}", self.get_val()),
      OPR => format!("OPR:{:08X}", self.get_val()),
      SWI => format!("SWI:{:08X}", self.get_val()),
      _   => panic!("Invalid tag"),
    }
  }
}

impl Pair {
  pub fn show(&self) -> String {
    format!("{} ~ {}", self.get_fst().show(), self.get_snd().show())
  }
}

impl RBag {
  pub fn show(&self) -> String {
    let mut s = String::new();
    s.push_str("RBAG | FST-TREE     | SND-TREE    \n");
    s.push_str("---- | ------------ | ------------\n");
    for (i, pair) in self.hi.iter().enumerate() {
      s.push_str(&format!("{:04X} | {} | {}\n", i, pair.get_fst().show(), pair.get_snd().show()));
    }
    s.push_str("~~~~ | ~~~~~~~~~~~~ | ~~~~~~~~~~~~\n");
    for (i, pair) in self.lo.iter().enumerate() {
      s.push_str(&format!("{:04X} | {} | {}\n", i + self.hi.len(), pair.get_fst().show(), pair.get_snd().show()));
    }
    s.push_str("==== | ============ | ============\n");
    return s;
  }
}

impl<'a> GNet<'a> {
  pub fn show(&self) -> String {
    let mut s = String::new();
    s.push_str("NODE | FST-PORT     | SND-PORT     \n");
    s.push_str("---- | ------------ | ------------\n");
    //for i in 0..256 {
    for i in 0..self.nlen-1 {
      let node = self.node_load(i);
      if node.0 != 0 {
        s.push_str(&format!("{:04X} | {} | {}\n", i, node.get_fst().show(), node.get_snd().show()));
      }
    }
    s.push_str("==== | ============ | ============\n");
    s.push_str("VARS | VALUE        |\n");
    s.push_str("---- | ------------ |\n");
    //for i in 0..256 {
    for i in 0..self.vlen-1 {
      let var = self.vars_load(i);
      if var.0 != 0 {
        s.push_str(&format!("{:04X} | {} |\n", i, var.show()));
      }
    }
    let root = self.vars_load(0x1FFFFFFF);
    s.push_str(&format!("ROOT | {} |\n", root.show()));
    s.push_str("==== | ============ |\n");
    return s;
  }
}

impl Book {
  pub fn show(&self) -> String {
    let mut s = String::new();
    for def in &self.defs {
      s.push_str(&format!("==== | ============ | ============ {} (vars={},safe={})\n", def.name, def.vars, def.safe));
      s.push_str("NODE | FST-PORT     | SND-PORT     \n");
      s.push_str("---- | ------------ | ------------\n");
      for (i, node) in def.node.iter().enumerate() {
        s.push_str(&format!("{:04X} | {} | {}\n", i, node.get_fst().show(), node.get_snd().show()));
      }
      s.push_str("==== | ============ | ============\n");
      s.push_str("RBAG | FST-TREE     | SND-TREE    \n");
      s.push_str("---- | ------------ | ------------\n");
      for (i, node) in def.rbag.iter().enumerate() {
        s.push_str(&format!("{:04X} | {} | {}\n", i, node.get_fst().show(), node.get_snd().show()));
      }
      s.push_str("==== | ============ | ============\n");

    }
    return s;
  }
}

impl Book {
  // Creates a demo program that is equivalent to:
  //   lop  = λn switch n { 0: 0; _: (lop n-1) }
  //   fun  = λn switch n { 0: (lop LOOPS); _: (+ (fun n-1) (fun n-1)) }
  //   main = (fun DEPTH)
  // Or, in core syntax:
  //   @fun  = (?<(@fun0 @fun1) a> a)
  //   @fun0 = a & @lop ~ (#65536 a)
  //   @fun1 = ({a b} c) & @fun ~ (a <+ d c>) & @fun ~ (b d)
  //   @lop  = (?<(#0 @lop0) a> a)
  //   @lop0 = (a b) & @lop ~ (a b)
  //   @main = a & @fun ~ (#10 a)
  //pub fn new_demo(depth: u32, loops: u32) -> Self {
    //let fun = Def {
      //name: "fun".to_string(),
      //safe: true,
      //rbag: vec![],
      //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x09),Port(0x11)), Pair::new(Port(0x14),Port(0x00))],
      //vars: 1,
    //};
    //let fun0 = Def {
      //name: "fun0".to_string(),
      //safe: true,
      //rbag: vec![Pair::new(Port(0x19),Port(0x0C))],
      //node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,loops),Port(0x00))],
      //vars: 1,
    //};
    //let fun1 = Def {
      //name: "fun1".to_string(),
      //safe: false,
      //rbag: vec![Pair::new(Port(0x01),Port(0x1C)), Pair::new(Port(0x01),Port(0x2C))],
      //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x15),Port(0x10)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x26)), Pair::new(Port(0x18),Port(0x10)), Pair::new(Port(0x08),Port(0x18))],
      //vars: 4,
    //};
    //let lop = Def {
      //name: "lop".to_string(),
      //safe: true,
      //rbag: vec![],
      //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x03),Port(0x21)), Pair::new(Port(0x14),Port(0x00))],
      //vars: 1,
    //};
    //let lop0 = Def {
      //name: "lop0".to_string(),
      //safe: true,
      //rbag: vec![Pair::new(Port(0x19),Port(0x14))],
      //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x08))],
      //vars: 2,
    //};
    //let main = Def {
      //name: "main".to_string(),
      //safe: true,
      //rbag: vec![Pair::new(Port(0x01),Port(0x0C))],
      //node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,depth),Port(0x00))],
      //vars: 1,
    //};
    //return Book {
      //defs: vec![fun, fun0, fun1, lop, lop0, main],
    //};
  //}
}
