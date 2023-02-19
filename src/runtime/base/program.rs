use crate::language;
use crate::prelude::*;
use crate::runtime::*;
use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// A runtime term
#[derive(Clone, Debug)]
pub enum Core {
  Var { bidx: u64 },
  Glo { glob: u64, misc: Tag },
  Dup { eras: (bool, bool), glob: u64, expr: Box<Core>, body: Box<Core> },
  Sup { val0: Box<Core>, val1: Box<Core> },
  Let { expr: Box<Core>, body: Box<Core> },
  Lam { eras: bool, glob: u64, body: Box<Core> },
  App { func: Box<Core>, argm: Box<Core> },
  Fun { func: u64, args: Vec<Core> },
  Ctr { func: u64, args: Vec<Core> },
  U6O { numb: u64 },
  F6O { numb: u64 },
  Op2 { oper: Oper, val0: Box<Core>, val1: Box<Core> },
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

pub type VisitFun = fn(ReduceCtx) -> bool;
pub type ApplyFun = fn(ReduceCtx) -> bool;

pub struct VisitObj {
  pub strict_map: Vec<bool>,
  pub strict_idx: Vec<u64>,
}

pub struct ApplyObj {
  pub rules: Vec<Rule>,
}

pub enum Function {
  Interpreted { smap: Box<[bool]>, visit: VisitObj, apply: ApplyObj },
  Compiled { smap: Box<[bool]>, visit: VisitFun, apply: ApplyFun },
}

pub type Funs = U64Map<Function>;
pub type Aris = U64Map<u64>;
pub type Nams = U64Map<String>;

pub struct Program {
  pub funs: Funs,
  pub aris: Aris,
  pub nams: Nams,
}

impl Default for Program {
  fn default() -> Self {
    Self::new()
  }
}

impl Program {
  pub fn new() -> Self {
    let mut funs = U64Map::new();
    let mut aris = U64Map::new();
    let mut nams = U64Map::new();
    // Adds the built-in functions
    for fid in 0..crate::runtime::precomp::PRECOMP_COUNT as usize {
      if let Some(precomp) = PRECOMP.get(fid) {
        if let Some(fs) = &precomp.funs {
          funs.insert(
            fid as u64,
            Function::Compiled {
              smap: precomp.smap.to_vec().into_boxed_slice(),
              visit: fs.visit,
              apply: fs.apply,
            },
          );
        }
        nams.insert(fid as u64, precomp.name.to_string());
        aris.insert(fid as u64, precomp.smap.len() as u64);
      }
    }
    Self { funs, aris, nams }
  }

  pub fn add_book(&mut self, book: &RuleBook) {
    let funs: &mut Funs = &mut book.gen_functions();
    let nams: &mut Nams = &mut gen_names(book);
    let aris: &mut Aris = &mut U64Map::new();
    for (fid, fun) in funs.data.drain(0..).enumerate() {
      if let Some(fun) = fun {
        self.funs.insert(fid as u64, fun);
      }
    }
    for (fid, nam) in nams.data.iter().enumerate() {
      if let Some(nam) = nam {
        self.nams.insert(fid as u64, nam.clone());
      }
    }
    for (fid, smp) in &book.id_to_smap {
      self.aris.insert(*fid, smp.len() as u64);
    }
  }

  pub fn add_function(&mut self, name: String, function: Function) {
    self.nams.push(name);
    self.funs.push(function);
  }
}

impl Heap {
  pub fn get_var(&self, term: Ptr, var: &RuleVar) -> Ptr {
    let RuleVar { param, field, erase: _ } = var;
    match field {
      Some(i) => self.take_arg(self.load_arg(term, *param), *i),
      None => self.take_arg(term, *param),
    }
  }
}

impl Heap {
  pub fn alloc_body(
    &self,
    prog: &Program,
    tid: usize,
    term: Ptr,
    vars: &[RuleVar],
    body: &RuleBody,
  ) -> Ptr {
    //#[inline(always)]
    fn cell_to_ptr(
      heap: &Heap,
      lvar: &LocalVars,
      aloc: &[AtomicU64],
      term: Ptr,
      vars: &[RuleVar],
      cell: &RuleBodyCell,
    ) -> Ptr {
      unsafe {
        match cell {
          RuleBodyCell::Val { value } => *value,
          RuleBodyCell::Var { index } => heap.get_var(term, vars.get_unchecked(*index as usize)),
          RuleBodyCell::Ptr { value, targ, slot } => {
            let mut val = value + *aloc.get_unchecked(*targ as usize).as_mut_ptr() + slot;
            // should be changed if the pointer format changes
            if value.tag() <= Tag::DP1 {
              val += (*lvar.dups.as_mut_ptr() & 0xFFF_FFFF) * EXT;
            }
            val
          }
        }
      }
    }
    // FIXME: verify the use of get_unchecked
    unsafe {
      let (cell, nodes, dupk) = body;
      let aloc = &self.aloc[tid];
      let lvar = &self.lvar[tid];
      for i in 0..nodes.len() {
        *aloc.get_unchecked(i).as_mut_ptr() =
          self.alloc(tid, (*nodes.get_unchecked(i)).len() as u64);
      }
      if *lvar.dups.as_mut_ptr() + dupk >= (1 << 28) {
        *lvar.dups.as_mut_ptr() = 0;
      }
      for i in 0..nodes.len() {
        let host = *aloc.get_unchecked(i).as_mut_ptr() as usize;
        for j in 0..(*nodes.get_unchecked(i)).len() {
          let cell = (*nodes.get_unchecked(i)).get_unchecked(j);
          let ptr = cell_to_ptr(self, lvar, aloc, term, vars, cell);
          if let RuleBodyCell::Var { .. } = cell {
            self.link((host + j) as u64, ptr);
          } else {
            *self.node.get_unchecked(host + j).as_mut_ptr() = ptr;
          }
        }
      }
      let done = cell_to_ptr(self, lvar, aloc, term, vars, cell);
      *lvar.dups.as_mut_ptr() += dupk;
      //println!("result: {}\n{}\n", show_ptr(done), show_term(heap, prog, done, 0));
      done
    }
  }
}

// todo: "dups" still needs to be moved out on `alloc_body` etc.
pub fn build_function(
  book: &RuleBook,
  fn_name: &str,
  rules: &[language::syntax::Rule],
) -> Function {
  let hoas = fn_name.starts_with("F$");
  let dynrules = rules
    .iter()
    .filter_map(|rule| {
      if let Term::Ctr { ref name, ref args } = *rule.lhs {
        let mut cond = vec![];
        let mut vars = vec![];
        let mut inps = vec![];
        let mut free = vec![];
        for (i, arg) in args.iter().enumerate() {
          match &**arg {
            Term::Ctr { name, args } => {
              cond.push(Ctr(*book.name_to_id.get(name).unwrap_or(&0), 0));
              free.push((i as u64, args.len() as u64));
              for (j, arg) in args.iter().enumerate() {
                if let Term::Var { ref name } = **arg {
                  vars.push(RuleVar { param: i as u64, field: Some(j as u64), erase: name == "*" });
                  inps.push(name.clone());
                } else {
                  panic!("sorry, left-hand sides can't have nested constructors yet.");
                }
              }
            }
            Term::U6O { numb } => {
              cond.push(U6O(*numb));
            }
            Term::F6O { numb } => {
              cond.push(F6O(*numb));
            }
            Term::Var { name } => {
              cond.push(Var(0));
              vars.push(RuleVar { param: i as u64, field: None, erase: name == "*" });
              inps.push(name.clone());
            }
            _ => {
              panic!("invalid left-hand side.");
            }
          }
        }

        let core = rule.rhs.to_core(book, &inps);
        let body = build_body(&core, vars.len() as u64);

        Some(Rule { hoas, cond, vars, core, body, free })
      } else {
        None
      }
    })
    .collect();

  let fnid = book.name_to_id.get(fn_name).unwrap();
  let smap = book.id_to_smap.get(fnid).unwrap().clone().into_boxed_slice();

  let strict_map = smap.to_vec();
  let mut strict_idx = vec![];
  for (i, is_strict) in smap.iter().enumerate() {
    if *is_strict {
      strict_idx.push(i as u64);
    }
  }

  Function::Interpreted {
    smap,
    visit: VisitObj { strict_map, strict_idx },
    apply: ApplyObj { rules: dynrules },
  }
}

pub fn hash<T: std::hash::Hash>(t: &T) -> u64 {
  use std::hash::Hasher;
  let mut s = std::collections::hash_map::DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

impl RuleBook {
  pub fn gen_functions(&self) -> U64Map<Function> {
    let mut funs: U64Map<Function> = U64Map::new();
    for (name, rules_info) in &self.rule_group {
      let fnid = self.name_to_id.get(name).unwrap_or(&0);
      let func = build_function(self, name, &rules_info.1);
      funs.insert(*fnid, func);
    }
    funs
  }
}

pub fn gen_names(book: &RuleBook) -> U64Map<String> {
  U64Map::from_hashmap(&mut book.id_to_name.clone())
}

impl Term {
  fn convert_term(&self, book: &RuleBook, depth: u64, vars: &mut Vec<String>) -> Core {
    match self {
      Self::Var { name } => {
        if let Some((idx, _)) = vars.iter().enumerate().rev().find(|(_, var)| var == &name) {
          Core::Var { bidx: idx as u64 }
        } else {
          match Tag::global_name_misc(name) {
            Some(Tag::VAR) => Core::Glo { glob: hash(name), misc: Tag::VAR },
            Some(Tag::DP0) => Core::Glo { glob: hash(&name[2..].to_string()), misc: Tag::DP0 },
            Some(Tag::DP1) => Core::Glo { glob: hash(&name[2..].to_string()), misc: Tag::DP1 },
            _ => panic!("Unexpected error."),
          }
        }
      }
      Self::Dup { nam0, nam1, expr, body } => {
        let eras = (nam0 == "*", nam1 == "*");
        let glob =
          if Tag::global_name_misc(nam0).is_some() { hash(&nam0[2..].to_string()) } else { 0 };
        let expr = Box::new(expr.convert_term(book, depth + 0, vars));
        vars.push(nam0.clone());
        vars.push(nam1.clone());
        let body = Box::new(body.convert_term(book, depth + 2, vars));
        vars.pop();
        vars.pop();
        Core::Dup { eras, glob, expr, body }
      }
      Self::Sup { val0, val1 } => {
        let val0 = Box::new(val0.convert_term(book, depth + 0, vars));
        let val1 = Box::new(val1.convert_term(book, depth + 0, vars));
        Core::Sup { val0, val1 }
      }
      Self::Lam { name, body } => {
        let glob = if Tag::global_name_misc(name).is_some() { hash(name) } else { 0 };
        let eras = name == "*";
        vars.push(name.clone());
        let body = Box::new(body.convert_term(book, depth + 1, vars));
        vars.pop();
        Core::Lam { eras, glob, body }
      }
      Self::Let { name, expr, body } => {
        let expr = Box::new(expr.convert_term(book, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(body.convert_term(book, depth + 1, vars));
        vars.pop();
        Core::Let { expr, body }
      }
      Self::App { func, argm } => {
        let func = Box::new(func.convert_term(book, depth + 0, vars));
        let argm = Box::new(argm.convert_term(book, depth + 0, vars));
        Core::App { func, argm }
      }
      Self::Ctr { name, args } => {
        let func = *book.name_to_id.get(name).unwrap_or_else(|| panic!("unbound symbol: {}", name));
        let args = args.iter().map(|arg| arg.convert_term(book, depth + 0, vars)).collect();
        if *book.ctr_is_fun.get(name).unwrap_or(&false) {
          Core::Fun { func, args }
        } else {
          Core::Ctr { func, args }
        }
      }
      Self::U6O { numb } => Core::U6O { numb: *numb },
      Self::F6O { numb } => Core::F6O { numb: *numb },
      Self::Op2 { oper, val0, val1 } => {
        let val0 = Box::new(val0.convert_term(book, depth + 0, vars));
        let val1 = Box::new(val1.convert_term(book, depth + 1, vars));
        Core::Op2 { oper: *oper, val0, val1 }
      }
    }
  }

  pub fn to_core(&self, book: &RuleBook, inps: &[String]) -> Core {
    #[allow(clippy::identity_op)]
    let mut vars = inps.to_vec();
    self.convert_term(book, 0, &mut vars)
  }
}
/// converts a language term to a runtime term

pub fn build_body(term: &Core, free_vars: u64) -> RuleBody {
  fn link(nodes: &mut [RuleBodyNode], targ: u64, slot: u64, elem: RuleBodyCell) {
    nodes[targ as usize][slot as usize] = elem;
    if let RuleBodyCell::Ptr { value, targ: var_targ, slot: var_slot } = elem {
      let tag = value.tag();
      if tag <= Tag::VAR {
        nodes[var_targ as usize][(var_slot + (tag.something())) as usize] =
          RuleBodyCell::Ptr { value: Arg(0), targ, slot };
      }
    }
  }
  fn alloc_lam(
    lams: &mut std::collections::HashMap<u64, u64>,
    nodes: &mut Vec<RuleBodyNode>,
    glob: u64,
  ) -> u64 {
    if let Some(targ) = lams.get(&glob) {
      return *targ;
    }
    let targ = nodes.len() as u64;
    nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
    link(nodes, targ, 0, RuleBodyCell::Val { value: Era() });
    if glob != 0 {
      lams.insert(glob, targ);
    }
    targ
  }

  fn alloc_dup(
    dups: &mut HashMap<u64, (u64, u64)>,
    nodes: &mut Vec<RuleBodyNode>,
    links: &mut Vec<(u64, u64, RuleBodyCell)>,
    dupk: &mut u64,
    glob: u64,
  ) -> (u64, u64) {
    if let Some(got) = dups.get(&glob) {
      return *got;
    }
    let dupc = *dupk;
    let targ = nodes.len() as u64;
    *dupk += 1;
    nodes.push(vec![RuleBodyCell::Val { value: 0 }; 3]);
    links.push((targ, 0, RuleBodyCell::Val { value: Era() }));
    links.push((targ, 1, RuleBodyCell::Val { value: Era() }));
    if glob != 0 {
      dups.insert(glob, (targ, dupc));
    }
    (targ, dupc)
  }

  fn gen_elems(
    term: &Core,
    dupk: &mut u64,
    vars: &mut Vec<RuleBodyCell>,
    lams: &mut HashMap<u64, u64>,
    dups: &mut HashMap<u64, (u64, u64)>,
    nodes: &mut Vec<RuleBodyNode>,
    links: &mut Vec<(u64, u64, RuleBodyCell)>,
  ) -> RuleBodyCell {
    match term {
      Core::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize]
        } else {
          panic!("unbound variable.");
        }
      }
      Core::Glo { glob, misc } => match *misc {
        Tag::VAR => {
          let targ = alloc_lam(lams, nodes, *glob);
          RuleBodyCell::Ptr { value: Var(0), targ, slot: 0 }
        }
        Tag::DP0 => {
          let (targ, dupc) = alloc_dup(dups, nodes, links, dupk, *glob);
          RuleBodyCell::Ptr { value: Dp0(dupc, 0), targ, slot: 0 }
        }
        Tag::DP1 => {
          let (targ, dupc) = alloc_dup(dups, nodes, links, dupk, *glob);
          RuleBodyCell::Ptr { value: Dp1(dupc, 0), targ, slot: 0 }
        }
        _ => {
          panic!("Unexpected error.");
        }
      },
      Core::Dup { eras: _, glob, expr, body } => {
        let (targ, dupc) = alloc_dup(dups, nodes, links, dupk, *glob);
        let expr = gen_elems(expr, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 2, expr));
        //let dupc = 0; // FIXME remove
        vars.push(RuleBodyCell::Ptr { value: Dp0(dupc, 0), targ, slot: 0 });
        vars.push(RuleBodyCell::Ptr { value: Dp1(dupc, 0), targ, slot: 0 });
        let body = gen_elems(body, dupk, vars, lams, dups, nodes, links);
        vars.pop();
        vars.pop();
        body
      }
      Core::Sup { val0, val1 } => {
        let dupc = *dupk;
        let targ = nodes.len() as u64;
        *dupk += 1;
        nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
        let val0 = gen_elems(val0, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 0, val0));
        let val1 = gen_elems(val1, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 1, val1));
        //let dupc = 0; // FIXME remove
        RuleBodyCell::Ptr { value: Sup(dupc, 0), targ, slot: 0 }
      }
      Core::Let { expr, body } => {
        let expr = gen_elems(expr, dupk, vars, lams, dups, nodes, links);
        vars.push(expr);
        let body = gen_elems(body, dupk, vars, lams, dups, nodes, links);
        vars.pop();
        body
      }
      Core::Lam { eras: _, glob, body } => {
        let targ = alloc_lam(lams, nodes, *glob);
        let var = RuleBodyCell::Ptr { value: Var(0), targ, slot: 0 };
        vars.push(var);
        let body = gen_elems(body, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 1, body));
        vars.pop();
        RuleBodyCell::Ptr { value: Lam(0), targ, slot: 0 }
      }
      Core::App { func, argm } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
        let func = gen_elems(func, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 0, func));
        let argm = gen_elems(argm, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 1, argm));
        RuleBodyCell::Ptr { value: App(0), targ, slot: 0 }
      }
      Core::Fun { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![RuleBodyCell::Val { value: 0 }; args.len()]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, lams, dups, nodes, links);
            links.push((targ, i as u64, arg));
          }
          RuleBodyCell::Ptr { value: Fun(*func, 0), targ, slot: 0 }
        } else {
          RuleBodyCell::Val { value: Fun(*func, 0) }
        }
      }
      Core::Ctr { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![RuleBodyCell::Val { value: 0 }; args.len()]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, lams, dups, nodes, links);
            links.push((targ, i as u64, arg));
          }
          RuleBodyCell::Ptr { value: Ctr(*func, 0), targ, slot: 0 }
        } else {
          RuleBodyCell::Val { value: Ctr(*func, 0) }
        }
      }
      Core::U6O { numb } => RuleBodyCell::Val { value: U6O(*numb) },
      Core::F6O { numb } => RuleBodyCell::Val { value: F6O(*numb) },
      Core::Op2 { oper, val0, val1 } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
        let val0 = gen_elems(val0, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 0, val0));
        let val1 = gen_elems(val1, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 1, val1));
        RuleBodyCell::Ptr { value: Op2(oper.as_u64(), 0), targ, slot: 0 }
      }
    }
  }

  let mut links: Vec<(u64, u64, RuleBodyCell)> = vec![];
  let mut nodes: Vec<RuleBodyNode> = vec![];
  let mut lams: HashMap<u64, u64> = HashMap::new();
  let mut dups: HashMap<u64, (u64, u64)> = HashMap::new();
  let mut vars: Vec<RuleBodyCell> =
    (0..free_vars).map(|i| RuleBodyCell::Var { index: i }).collect();
  let mut dupk: u64 = 0;

  let elem = gen_elems(term, &mut dupk, &mut vars, &mut lams, &mut dups, &mut nodes, &mut links);
  for (targ, slot, elem) in links {
    link(&mut nodes, targ, slot, elem);
  }

  (elem, nodes, dupk)
}

pub fn alloc_closed_core(heap: &Heap, prog: &Program, tid: usize, term: &Core) -> u64 {
  let host = heap.alloc(tid, 1);
  let body = build_body(term, 0);
  let term = heap.alloc_body(prog, tid, 0, &[], &body);
  heap.link(host, term);
  host
}

pub fn alloc_term(heap: &Heap, prog: &Program, tid: usize, book: &RuleBook, term: &Term) -> u64 {
  alloc_closed_core(heap, prog, tid, &term.to_core(book, &[]))
}

pub fn make_string(heap: &Heap, tid: usize, text: &str) -> Ptr {
  let mut term = Ctr(STRING_NIL, 0);
  for chr in text.chars().rev() {
    // TODO: reverse
    let ctr0 = heap.alloc(tid, 2);
    heap.link(ctr0 + 0, U6O(chr as u64));
    heap.link(ctr0 + 1, term);
    term = Ctr(STRING_CONS, ctr0);
  }
  term
}
