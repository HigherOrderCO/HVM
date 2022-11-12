use crate::runtime::{*};
use crate::language;
use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// A runtime term
#[derive(Clone, Debug)]
pub enum Core {
  Var { bidx: u64 },
  Glo { glob: u64, misc: u64 },
  Dup { eras: (bool, bool), glob: u64, expr: Box<Core>, body: Box<Core> },
  Sup { val0: Box<Core>, val1: Box<Core> },
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

pub struct ReduceCtx<'a> {
  pub heap  : &'a Heap,
  pub prog  : &'a Program,
  pub tid   : usize,
  pub term  : Ptr,
  pub visit : &'a VisitQueue,
  pub redex : &'a RedexBag,
  pub cont  : &'a mut u64,
  pub host  : &'a mut u64,
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
  Interpreted {
    arity: u64,
    visit: VisitObj,
    apply: ApplyObj,
  },
  Compiled {
    arity: u64,
    visit: VisitFun,
    apply: ApplyFun,
  }
}

pub type Funs = U64Map<Function>;
pub type Arit = U64Map<u64>;
pub type Nams = U64Map<String>;

pub struct Program {
  pub funs: Funs,
  pub arit: Arit,
  pub nams: Nams,
}

pub fn new_program() -> Program {
  let mut funs = U64Map::new();
  let mut arit = U64Map::new();
  let mut nams = U64Map::new();
  // Adds the built-in functions
  for fid in 0 .. crate::runtime::precomp::PRECOMP_COUNT as usize {
    if let Some(precomp) = PRECOMP.get(fid) {
      if let Some(funcs) = &precomp.funcs {
        funs.insert(fid as u64, Function::Compiled {
          arity: precomp.arity as u64,
          visit: funcs.visit,
          apply: funcs.apply,
        });
      }
      nams.insert(fid as u64, precomp.name.to_string());
      arit.insert(fid as u64, precomp.arity as u64);
    }
  }
  return Program { funs, arit, nams };
}

pub fn extend_program(prog: &mut Program, funs: &mut Funs, arit: &mut Arit, nams: &Nams) {
  for (fid, fun) in funs.data.drain(0..).enumerate() {
    if let Some(fun) = fun {
      prog.funs.insert(fid as u64, fun);
    }
  }
  for (fid, nam) in nams.data.iter().enumerate() {
    if let Some(nam) = nam {
      prog.nams.insert(fid as u64, nam.clone());
    }
  }
  for (fid, ari) in arit.data.iter().enumerate() {
    if let Some(ari) = ari {
      prog.arit.insert(fid as u64, *ari);
    }
  }
}

pub fn init_program(funs: &mut Funs, arit: &mut Arit, nams: &mut Nams) -> Program {
  let mut prog = new_program();
  extend_program(&mut prog, funs, arit, nams);
  return prog;
}

pub fn get_var(heap: &Heap, term: Ptr, var: &RuleVar) -> Ptr {
  let RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => take_arg(heap, load_arg(heap, term, *param), *i),
    None    => take_arg(heap, term, *param),
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
            val += (*lvar.dups.as_mut_ptr() & 0xFFF_FFFF) * EXT;
          }
          if get_tag(*value) == DP1 {
            val += (*lvar.dups.as_mut_ptr() & 0xFFF_FFFF) * EXT;
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
    if *lvar.dups.as_mut_ptr() + dupk >= (1 << 28) {
      *lvar.dups.as_mut_ptr() = 0;
    }
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

pub fn get_global_name_misc(name: &str) -> Option<u64> {
  if !name.is_empty() && name.starts_with(&"$") {
    if name.starts_with(&"$0") {
      return Some(DP0);
    } else if name.starts_with(&"$1") {
      return Some(DP1);
    } else {
      return Some(VAR);
    }
  }
  return None;
}

// todo: "dups" still needs to be moved out on `alloc_body` etc.
pub fn build_function(book: &language::rulebook::RuleBook, fn_name: &str, rules: &[language::syntax::Rule]) -> Function {
  let mut is_strict = if let language::syntax::Term::Ctr { name: _, ref args } = *rules[0].lhs {
    vec![false; args.len()]
  } else {
    panic!("invalid left-hand side: {}", rules[0].lhs);
  };
  let hoas = fn_name.starts_with("f$");
  let dynrules = rules.iter().filter_map(|rule| {
    if let language::syntax::Term::Ctr { ref name, ref args } = *rule.lhs {
      let mut cond = Vec::new();
      let mut vars = Vec::new();
      let mut inps = Vec::new();
      let mut free = Vec::new();
      if args.len() != is_strict.len() {
        panic!("inconsistent length of left-hand side on equation for '{}'.", name);
      }
      for ((i, arg), is_strict) in args.iter().enumerate().zip(is_strict.iter_mut()) {
        match &**arg {
          language::syntax::Term::Ctr { name, args } => {
            *is_strict = true;
            cond.push(Ctr(*book.name_to_id.get(&*name).unwrap_or(&0), 0));
            free.push((i as u64, args.len() as u64));
            for (j, arg) in args.iter().enumerate() {
              if let language::syntax::Term::Var { ref name } = **arg {
                vars.push(RuleVar { param: i as u64, field: Some(j as u64), erase: name == "*" });
                inps.push(name.clone());
              } else {
                panic!("sorry, left-hand sides can't have nested constructors yet.");
              }
            }
          }
          language::syntax::Term::Num { numb } => {
            *is_strict = true;
            cond.push(Num(*numb as u64));
          }
          language::syntax::Term::Var { name } => {
            cond.push(Var(0));
            vars.push(RuleVar { param: i as u64, field: None, erase: name == "*" });
            inps.push(name.clone());
          }
          _ => {
            panic!("invalid left-hand side.");
          }
        }
      }

      let core = term_to_core(book, &rule.rhs, &inps);
      let body = build_body(&core, vars.len() as u64);

      Some(Rule { hoas, cond, vars, core, body, free })
    } else {
      None
    }
  }).collect();

  let arity = is_strict.len() as u64;
  let mut stricts = Vec::new();
  for (i, is_strict) in is_strict.iter().enumerate() {
    if *is_strict {
      stricts.push(i as u64);
    }
  }

  Function::Interpreted {
    arity,
    visit: VisitObj { strict_map: is_strict, strict_idx: stricts },
    apply: ApplyObj { rules: dynrules },
  }
}

pub fn hash<T: std::hash::Hash>(t: &T) -> u64 {
  use std::hash::Hasher;
  let mut s = std::collections::hash_map::DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

pub fn gen_arities(rb: &language::rulebook::RuleBook) -> U64Map<u64> {
  // todo: checking arity consistency would be nice, but where?
  fn find_arities(rb: &language::rulebook::RuleBook, arit: &mut U64Map<u64>, term: &language::syntax::Term) {
    match term {
      language::syntax::Term::Var { .. } => {}
      language::syntax::Term::Dup { expr, body, .. } => {
        find_arities(rb, arit, expr);
        find_arities(rb, arit, body);
      }
      language::syntax::Term::Sup { val0, val1 } => {
        find_arities(rb, arit, val0);
        find_arities(rb, arit, val1);
      }
      language::syntax::Term::Lam { body, .. } => {
        find_arities(rb, arit, body);
      }
      language::syntax::Term::Let { expr, body, .. } => {
        find_arities(rb, arit, expr);
        find_arities(rb, arit, body);
      }
      language::syntax::Term::App { func, argm } => {
        find_arities(rb, arit, func);
        find_arities(rb, arit, argm);
      }
      language::syntax::Term::Ctr { name, args } => {
        if let Some(id) = rb.name_to_id.get(name) {
          arit.insert(*id, args.len() as u64);
          for arg in args {
            find_arities(rb, arit, arg);
          }
        }
      }
      language::syntax::Term::Num { .. } => {}
      language::syntax::Term::Op2 { val0, val1, .. } => {
        find_arities(rb, arit, val0);
        find_arities(rb, arit, val1);
      }
    }
  }
  let mut arit: U64Map<u64> = U64Map::new();
  for rules_info in rb.rule_group.values() {
    for rule in &rules_info.1 {
      find_arities(rb, &mut arit, &rule.lhs);
      find_arities(rb, &mut arit, &rule.rhs);
    }
  }
  return arit;
}

pub fn gen_functions(book: &language::rulebook::RuleBook) -> U64Map<Function> {
  let mut funs: U64Map<Function> = U64Map::new();
  for (name, rules_info) in &book.rule_group {
    let fnid = book.name_to_id.get(name).unwrap_or(&0);
    let func = build_function(book, &name, &rules_info.1);
    funs.insert(*fnid, func);
  }
  funs
}

pub fn gen_names(book: &language::rulebook::RuleBook) -> U64Map<String> {
  return U64Map::from_hashmap(&mut book.id_to_name.clone());
}

/// converts a language term to a runtime term
pub fn term_to_core(book: &language::rulebook::RuleBook, term: &language::syntax::Term, inps: &[String]) -> Core {
  fn convert_oper(oper: &language::syntax::Oper) -> u64 {
    match oper {
      language::syntax::Oper::Add => ADD,
      language::syntax::Oper::Sub => SUB,
      language::syntax::Oper::Mul => MUL,
      language::syntax::Oper::Div => DIV,
      language::syntax::Oper::Mod => MOD,
      language::syntax::Oper::And => AND,
      language::syntax::Oper::Or  => OR,
      language::syntax::Oper::Xor => XOR,
      language::syntax::Oper::Shl => SHL,
      language::syntax::Oper::Shr => SHR,
      language::syntax::Oper::Ltn => LTN,
      language::syntax::Oper::Lte => LTE,
      language::syntax::Oper::Eql => EQL,
      language::syntax::Oper::Gte => GTE,
      language::syntax::Oper::Gtn => GTN,
      language::syntax::Oper::Neq => NEQ,
    }
  }

  #[allow(clippy::identity_op)]
  fn convert_term(
    term: &language::syntax::Term,
    book: &language::rulebook::RuleBook,
    depth: u64,
    vars: &mut Vec<String>,
  ) -> Core {
    match term {
      language::syntax::Term::Var { name } => {
        if let Some((idx, _)) = vars.iter().enumerate().rev().find(|(_, var)| var == &name) {
          Core::Var { bidx: idx as u64 }
        } else {
          match get_global_name_misc(name) {
            Some(VAR) => Core::Glo { glob: hash(name), misc: VAR },
            Some(DP0) => Core::Glo { glob: hash(&name[2..].to_string()), misc: DP0 },
            Some(DP1) => Core::Glo { glob: hash(&name[2..].to_string()), misc: DP1 },
            _ => panic!("Unexpected error."),
          }
        }
      }
      language::syntax::Term::Dup { nam0, nam1, expr, body } => {
        let eras = (nam0 == "*", nam1 == "*");
        let glob = if get_global_name_misc(nam0).is_some() { hash(&nam0[2..].to_string()) } else { 0 };
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(nam0.clone());
        vars.push(nam1.clone());
        let body = Box::new(convert_term(body, book, depth + 2, vars));
        vars.pop();
        vars.pop();
        Core::Dup { eras, glob, expr, body }
      }
      language::syntax::Term::Sup { val0, val1 } => {
        let val0 = Box::new(convert_term(val0, book, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, book, depth + 0, vars));
        Core::Sup { val0, val1 }
      }
      language::syntax::Term::Lam { name, body } => {
        let glob = if get_global_name_misc(name).is_some() { hash(name) } else { 0 };
        let eras = name == "*";
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        Core::Lam { eras, glob, body }
      }
      language::syntax::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        Core::Let { expr, body }
      }
      language::syntax::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, book, depth + 0, vars));
        let argm = Box::new(convert_term(argm, book, depth + 0, vars));
        Core::App { func, argm }
      }
      language::syntax::Term::Ctr { name, args } => {
        let term_func = *book.name_to_id.get(name).unwrap_or_else(|| panic!("unbound symbol: {}", name));
        let term_args = args.iter().map(|arg| convert_term(arg, book, depth + 0, vars)).collect();
        if *book.ctr_is_cal.get(name).unwrap_or(&false) {
          Core::Fun { func: term_func, args: term_args }
        } else {
          Core::Ctr { func: term_func, args: term_args }
        }
      }
      language::syntax::Term::Num { numb } => Core::Num { numb: *numb },
      language::syntax::Term::Op2 { oper, val0, val1 } => {
        let oper = convert_oper(oper);
        let val0 = Box::new(convert_term(val0, book, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, book, depth + 1, vars));
        Core::Op2 { oper, val0, val1 }
      }
    }
  }

  let mut vars = inps.to_vec();
  convert_term(term, book, 0, &mut vars)
}

pub fn build_body(term: &Core, free_vars: u64) -> RuleBody {
  fn link(nodes: &mut [RuleBodyNode], targ: u64, slot: u64, elem: RuleBodyCell) {
    nodes[targ as usize][slot as usize] = elem;
    if let RuleBodyCell::Ptr { value, targ: var_targ, slot: var_slot } = elem {
      let tag = get_tag(value);
      if tag <= VAR {
        nodes[var_targ as usize][(var_slot + (tag & 0x01)) as usize] = RuleBodyCell::Ptr { value: Arg(0), targ, slot };
      }
    }
  }
  fn alloc_lam(lams: &mut std::collections::HashMap<u64, u64>, nodes: &mut Vec<RuleBodyNode>, glob: u64) -> u64 {
    if let Some(targ) = lams.get(&glob) {
      *targ
    } else {
      let targ = nodes.len() as u64;
      nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
      link(nodes, targ, 0, RuleBodyCell::Val { value: Era() });
      if glob != 0 {
        lams.insert(glob, targ);
      }
      return targ;
    }
  }
  fn alloc_dup(dups: &mut HashMap<u64, (u64,u64)>, nodes: &mut Vec<RuleBodyNode>, links: &mut Vec<(u64, u64, RuleBodyCell)>, dupk: &mut u64, glob: u64) -> (u64, u64) {
    if let Some(got) = dups.get(&glob) {
      return got.clone();
    } else {
      let dupc = *dupk;
      let targ = nodes.len() as u64;
      *dupk += 1;
      nodes.push(vec![RuleBodyCell::Val { value: 0 }; 3]);
      if glob != 0 {
        links.push((targ, 0, RuleBodyCell::Val { value: Era() }));
        links.push((targ, 1, RuleBodyCell::Val { value: Era() }));
        dups.insert(glob, (targ, dupc));
      }
      return (targ, dupc);
    }
  }
  fn gen_elems(
    term: &Core,
    dupk: &mut u64,
    vars: &mut Vec<RuleBodyCell>,
    lams: &mut HashMap<u64, u64>,
    dups: &mut HashMap<u64, (u64,u64)>,
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
      Core::Glo { glob, misc } => {
        match *misc {
          VAR => {
            let targ = alloc_lam(lams, nodes, *glob);
            return RuleBodyCell::Ptr { value: Var(0), targ, slot: 0 };
          }
          DP0 => {
            let (targ, dupc) = alloc_dup(dups, nodes, links, dupk, *glob);
            return RuleBodyCell::Ptr { value: Dp0(dupc, 0), targ, slot: 0 };
          }
          DP1 => {
            let (targ, dupc) = alloc_dup(dups, nodes, links, dupk, *glob);
            return RuleBodyCell::Ptr { value: Dp1(dupc, 0), targ, slot: 0 };
          }
          _ => {
            panic!("Unexpected error.");
          }
        }
      }
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
          nodes.push(vec![RuleBodyCell::Val { value: 0 }; args.len() as usize]);
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
          nodes.push(vec![RuleBodyCell::Val { value: 0 }; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, lams, dups, nodes, links);
            links.push((targ, i as u64, arg));
          }
          RuleBodyCell::Ptr { value: Ctr(*func, 0), targ, slot: 0 }
        } else {
          RuleBodyCell::Val { value: Ctr(*func, 0) }
        }
      }
      Core::Num { numb } => RuleBodyCell::Val { value: Num(*numb as u64) },
      Core::Op2 { oper, val0, val1 } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![RuleBodyCell::Val { value: 0 }; 2]);
        let val0 = gen_elems(val0, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 0, val0));
        let val1 = gen_elems(val1, dupk, vars, lams, dups, nodes, links);
        links.push((targ, 1, val1));
        RuleBodyCell::Ptr { value: Op2(*oper, 0), targ, slot: 0 }
      }
    }
  }

  let mut links: Vec<(u64, u64, RuleBodyCell)> = Vec::new();
  let mut nodes: Vec<RuleBodyNode> = Vec::new();
  let mut lams: HashMap<u64, u64> = HashMap::new();
  let mut dups: HashMap<u64, (u64,u64)> = HashMap::new();
  let mut vars: Vec<RuleBodyCell> = (0..free_vars).map(|i| RuleBodyCell::Var { index: i }).collect();
  let mut dupk: u64 = 0;

  let elem = gen_elems(term, &mut dupk, &mut vars, &mut lams, &mut dups, &mut nodes, &mut links);
  for (targ, slot, elem) in links {
    link(&mut nodes, targ, slot, elem);
  }

  (elem, nodes, dupk)
}

pub fn alloc_closed_core(heap: &Heap, tid: usize, term: &Core) -> u64 {
  let host = alloc(heap, tid, 1);
  let body = build_body(term, 0);
  let term = alloc_body(heap, tid, 0, &[], &body);
  link(heap, host, term);
  host
}

pub fn alloc_term(heap: &Heap, tid: usize, book: &language::rulebook::RuleBook, term: &language::syntax::Term) -> u64 {
  alloc_closed_core(heap, tid, &term_to_core(book, term, &vec![]))
}

// Evaluates a HVM term to normal form
pub fn eval_code(
  call: &language::syntax::Term,
  code: &str,
  debug: bool,
  size: usize,
) -> Result<(String, u64, i64, u64), String> {
  // Parses and reads the input file
  let file = language::syntax::read_file(code)?;

  // Converts the HVM "file" to a Rulebook
  let book = language::rulebook::gen_rulebook(&file);

  // Creates the runtime program
  let prog = init_program(&mut gen_functions(&book), &mut gen_arities(&book), &mut gen_names(&book));

  let mut heap = new_heap(size, available_parallelism());
  let     tids = new_tids(available_parallelism());

  // Allocates the main term
  let host = alloc_term(&mut heap, tids[0], &book, call); // FIXME tid?

  // Normalizes it
  let init = instant::Instant::now();
  #[cfg(not(target_arch = "wasm32"))]
  normalize(&heap, &prog, &tids, host, true);
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a string
  let code = format!("{}", language::readback::as_term(&heap, &prog, host));

  // Collects the result expression
  collect(&heap, &prog.arit, tids[0], load_ptr(&heap, host));

  // Frees the memory used by the main node
  // TODO: must get the arity

  // Returns the result, the rewrite cost, used memory and time elapsed
  Ok((code, get_cost(&heap), get_used(&heap), time))

  //println!("before collect:\n{}", runtime::show_heap(&heap));
  //println!("{}", runtime::show_term(&heap, &prog, runtime::load_ptr(&heap, host), runtime::load_ptr(&heap, host)));
  //runtime::collect(&heap, &prog, tids[0], runtime::load_ptr(&heap, host));
  //runtime::free(&heap, 0, 0, 2); // FIXME: remove this (just clearing main)

  //println!("after collect:\n{}", runtime::show_heap(&heap));
  //println!("{}", runtime::show_term(&heap, &prog, runtime::load_ptr(&heap, host), runtime::load_ptr(&heap, host)));

}
