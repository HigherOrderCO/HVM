use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::alloc::{alloc, dealloc, Layout};
use std::mem;

// Runtime
// =======

// Types
pub type Tag  = u8;
pub type Lab  = u32;
pub type Val  = u32;
pub type AVal = AtomicU32;
pub type Rule = u8;

// Port
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct Port(pub Val);
pub struct APort(pub AVal);

// Pair
pub struct Pair(pub u64);
pub struct APair(pub AtomicU64);

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

// None
pub const NONE : Port = Port(0xFFFFFFF9);

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
  pub vars: &'a mut [APair], // vars buffer
  pub itrs: AtomicU64, // interaction count
}

// Thread Memory
pub struct TMem {
  pub tid: u32, // thread id
  pub tids: u32, // thread count
  pub tick: u32, // tick counter
  pub itrs: u32, // interaction count
  pub nidx: usize, // next node allocation index
  pub vidx: usize, // next vars allocation index
  pub nloc: Vec<usize>, // allocated node locations
  pub vloc: Vec<usize>, // allocated vars locations
  pub rbag: RBag, // local redex bag
}

// Top-Level Definition
pub struct Def {
  pub name: String, // def name
  pub safe: bool, // has no dups
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
      Port::new(tag, tm.nloc[val as usize - 1] as u32)
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
    let vlay = Layout::array::<APair>(vlen).unwrap();
    let nptr = unsafe { alloc(nlay) as *mut APair };
    let vptr = unsafe { alloc(vlay) as *mut APair };
    let node = unsafe { std::slice::from_raw_parts_mut(nptr, nlen) };
    let vars = unsafe { std::slice::from_raw_parts_mut(vptr, vlen) };
    GNet { nlen, vlen, node, vars, itrs: AtomicU64::new(0) }
  }

  pub fn node_create(&self, loc: usize, val: Pair) {
    self.node[loc].0.store(val.0, Ordering::Relaxed);
  }

  pub fn vars_create(&self, var: usize, val: Port) {
    self.vars[var].0.store(val.0 as u64, Ordering::Relaxed);
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
    self.vars[var].0.store(val.0 as u64, Ordering::Relaxed);
  }
  
  pub fn node_exchange(&self, loc: usize, val: Pair) -> Pair {
    Pair(self.node[loc].0.swap(val.0, Ordering::Relaxed))
  }

  pub fn vars_exchange(&self, var: usize, val: Port) -> Port {
    Port(self.vars[var].0.swap(val.0 as u64, Ordering::Relaxed) as u32)
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
      nidx: 0,
      vidx: 0,
      nloc: vec![0; 32],
      vloc: vec![0; 32],  
      rbag: RBag::new(),
    }
  }
  
  pub fn node_alloc(&mut self, net: &GNet, num: usize) -> usize {
    let mut got = 0;
    for _ in 0..net.nlen {
      self.nidx += 1;
      if self.nidx < net.nlen || net.is_node_free(self.nidx % net.nlen) {
        self.nloc[got] = self.nidx % net.nlen;
        got += 1;
        //println!("ALLOC NODE {} {}", got, self.nidx);
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
      self.vidx += 1;
      if self.vidx < net.vlen || net.is_vars_free(self.vidx % net.vlen) {
        self.vloc[got] = self.vidx % net.nlen;
        //println!("ALLOC VARS {} {}", got, self.vidx);
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
    let mut a = a;
    let mut b = b;

    // Attempts to directionally point `A ~> B` 
    loop {
      // If `A` is NODE: swap `A` and `B`, and continue
      if a.get_tag() != VAR {
        let x = a; a = b; b = x;
      }

      // If `A` is NODE: create the `A ~ B` redex
      if a.get_tag() != VAR {
        self.rbag.push_redex(Pair::new(a, b));
        break;
      }

      // While `B` is VAR: extend it (as an optimization)
      while b.get_tag() == VAR {
        // Takes the current `B` substitution as `B'`
        let b_ = net.vars_exchange(b.get_val() as usize, NONE);
        // If there was no `B'`, stop, as there is no extension
        if b_ == NONE || b_ == Port(0) {
          break;
        }
        // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
        net.vars_take(b.get_val() as usize);
        b = b_;
      }

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

  // TODO: implement all the INTERACTION functions below.

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
    let fid = a.get_val() as usize;
    let def = &book.defs[fid];

    // Copy Optimization.
    if def.safe && b.get_tag() == DUP {
      return self.interact_eras(net, a, b);
    }

    // Allocates needed nodes and vars.
    if !self.get_resources(net, def.rbag.len() + 1, def.node.len() - 1, def.vars as usize) {
      return false;
    }

    // Stores new vars.
    for i in 0..def.vars {
      net.vars_create(self.vloc[i], NONE);
      //println!("vars_create vars_loc[{:04X}] {:04X}", i, self.vloc[i]);
    }

    // Stores new nodes.
    for i in 1..def.node.len() {
      net.node_create(self.nloc[i-1], def.node[i].adjust_pair(self));
      //println!("node_create node_loc[{:04X}] {:016X}", i-1, def.node[i].0);
    }

    // Links.
    self.link_pair(net, Pair::new(b, def.node[0].get_fst().adjust_port(self)));
    for pair in &def.rbag {
      self.link_pair(net, pair.adjust_pair(self));
    }
  
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
    self.link_pair(net, Pair::new(a1, Port::new(b.get_tag(), self.nloc[0] as u32)));
    self.link_pair(net, Pair::new(a2, Port::new(b.get_tag(), self.nloc[1] as u32)));  
    self.link_pair(net, Pair::new(b1, Port::new(a.get_tag(), self.nloc[2] as u32)));
    self.link_pair(net, Pair::new(b2, Port::new(a.get_tag(), self.nloc[3] as u32)));
    
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
    let b2 = b_.get_snd();
     
    // Performs operation.
    if b1.get_tag() == NUM {
      let bv = b1.get_val();
      let rv = av + bv;
      self.link_pair(net, Pair::new(b2, Port::new(NUM, rv))); 
    } else {
      net.node_create(self.nloc[0], Pair::new(a, b2));
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
    let av = a.get_val();
    let b_ = net.node_take(b.get_val() as usize);
    let b1 = b_.get_fst();
    let b2 = b_.get_snd();
 
    // Stores new nodes.  
    if av == 0 {
      net.node_create(self.nloc[0], Pair::new(b2, Port::new(ERA,0)));
      self.link_pair(net, Pair::new(Port::new(CON, self.nloc[0] as u32), b1));
    } else {
      net.node_create(self.nloc[0], Pair::new(Port::new(ERA,0), Port::new(CON, self.nloc[1] as u32)));
      net.node_create(self.nloc[1], Pair::new(Port::new(NUM, av-1), b2));
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
    if a.get_tag() == REF && b == NONE {
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
    } else {
      self.itrs += 1;
      true
    }
  }

  pub fn evaluator(&mut self, net: &GNet, book: &Book) {
    // Increments the tick
    self.tick += 1;

    // Performs some interactions  
    while self.rbag.len() > 0 {
      self.interact(net, book);
    }

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
    for i in 0..self.nlen {
      let node = self.node_load(i);
      if node.0 != 0 {
        s.push_str(&format!("{:04X} | {} | {}\n", i, node.get_fst().show(), node.get_snd().show()));
      }
    }
    s.push_str("==== | ============ | ============\n");  
    s.push_str("VARS | VALUE        |\n");
    s.push_str("---- | ------------ |\n");  
    for i in 0..self.vlen {
      let var = self.vars_load(i);
      if var.0 != 0 {
        s.push_str(&format!("{:04X} | {} |\n", i, var.show()));
      }
    }
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
  pub fn new_demo(depth: u32, loops: u32) -> Self {
    let fun = Def {
      name: "fun".to_string(),
      safe: true,
      rbag: vec![],
      node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x09),Port(0x11)), Pair::new(Port(0x14),Port(0x00))],
      vars: 1,
    };
    let fun0 = Def {
      name: "fun0".to_string(),
      safe: true,
      rbag: vec![Pair::new(Port(0x19),Port(0x0C))],
      node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,loops),Port(0x00))],
      vars: 1,
    };
    let fun1 = Def {
      name: "fun1".to_string(),
      safe: false,
      rbag: vec![Pair::new(Port(0x01),Port(0x1C)), Pair::new(Port(0x01),Port(0x2C))],
      node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x15),Port(0x10)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x26)), Pair::new(Port(0x18),Port(0x10)), Pair::new(Port(0x08),Port(0x18))],
      vars: 4,
    };
    let lop = Def {
      name: "lop".to_string(),
      safe: true,
      rbag: vec![],
      node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x03),Port(0x21)), Pair::new(Port(0x14),Port(0x00))],
      vars: 1,  
    };
    let lop0 = Def {
      name: "lop0".to_string(),
      safe: true,
      rbag: vec![Pair::new(Port(0x19),Port(0x14))],
      node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x08))],
      vars: 2,
    };
    let main = Def {
      name: "main".to_string(),
      safe: true,
      rbag: vec![Pair::new(Port(0x01),Port(0x0C))],
      node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,depth),Port(0x00))],
      vars: 1,
    };
    return Book {
      defs: vec![fun, fun0, fun1, lop, lop0, main],
    };
  }
}

pub fn run(book: &Book) {
  // Initializes the global net
  let net = GNet::new(1 << 29, 1 << 29);

  // Initializes threads
  let mut tm = TMem::new(0, 1);

  // Creates an initial redex that calls main
  let main_id = book.defs.iter().position(|def| def.name == "main").unwrap();
  tm.rbag.push_redex(Pair::new(Port::new(REF, main_id as u32), NONE));

  // Starts the timer
  let start = std::time::Instant::now();

  // Evaluates
  tm.evaluator(&net, &book);
  
  // Stops the timer
  let duration = start.elapsed();

  // Prints interactions and time
  let itrs = net.itrs.load(Ordering::Relaxed);
  println!("itrs: {}", itrs);
  println!("time: {:.2}s", duration.as_secs_f64());
  println!("MIPS: {:.2}", itrs as f64 / duration.as_secs_f64() / 1_000_000.0);
}

pub fn run_demo() {
  run(&Book::new_demo(10, 65536));
}
