//! Moves HVM Terms to runtime, and building dynamic functions.

// TODO: "dups" still needs to be moved out on `alloc_body` etc.

use crate::language as lang;
use crate::readback as rd;
use crate::rulebook as rb;
use crate::runtime as rt;
use std::collections::HashMap;
use std::iter;
use std::time::Instant;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub enum DynTerm {
  Var { bidx: u64 },
  Glo { glob: u64 },
  Dup { eras: (bool, bool), expr: Box<DynTerm>, body: Box<DynTerm> },
  Let { expr: Box<DynTerm>, body: Box<DynTerm> },
  Lam { eras: bool, glob: u64, body: Box<DynTerm> },
  App { func: Box<DynTerm>, argm: Box<DynTerm> },
  Cal { func: u64, args: Vec<DynTerm> },
  Ctr { func: u64, args: Vec<DynTerm> },
  Num { numb: u64 },
  Op2 { oper: u64, val0: Box<DynTerm>, val1: Box<DynTerm> },
}

// The right-hand side of a rule, "digested" or "pre-filled" in a way that is almost ready to be
// pasted into the runtime's memory, except for some minor adjustments: internal links need to be
// adjusted taking in account the index where each node is actually allocated, and external
// variables must be linked. This structure helps moving as much computation from the interpreter
// to the static time (i.e., when generating the dynfun closure) as possible.
pub type Body = (Elem, Vec<Node>, u64); // The pre-filled body (TODO: can the Node Vec be unboxed?)
pub type Node = Vec<Elem>; // A node on the pre-filled body

#[derive(Copy, Clone, Debug)]
pub enum Elem {
  // An element of a node
  Fix { value: u64 }, // Fixed value, doesn't require adjustment
  Loc { value: u64, targ: u64, slot: u64 }, // Local link, requires adjustment
  Ext { index: u64 }, // Link to an external variable
}

#[derive(Debug)]
pub struct DynVar {
  pub param: u64,
  pub field: Option<u64>,
  pub erase: bool,
}

#[derive(Debug)]
pub struct DynRule {
  pub cond: Vec<rt::Ptr>,
  pub vars: Vec<DynVar>,
  pub term: DynTerm,
  pub body: Body,
  pub free: Vec<(u64, u64)>,
}

#[derive(Debug)]
pub struct DynFun {
  pub redex: Vec<bool>,
  pub rules: Vec<DynRule>,
}

pub fn build_dynfun(book: &rb::RuleBook, rules: &[lang::Rule]) -> DynFun {
  let mut redex = if let lang::Term::Ctr { name: _, ref args } = *rules[0].lhs {
    vec![false; args.len()]
  } else {
    panic!("Invalid left-hand side: {}", rules[0].lhs);
  };
  let dynrules = rules
    .iter()
    .filter_map(|rule| {
      if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
        let mut cond = Vec::new();
        let mut vars = Vec::new();
        let mut inps = Vec::new();
        let mut free = Vec::new();
        if args.len() != redex.len() {
          panic!("Inconsistent length of left-hand side on equation for '{}'.", name);
        }
        for ((i, arg), redex) in args.iter().enumerate().zip(redex.iter_mut()) {
          match &**arg {
            lang::Term::Ctr { name, args } => {
              *redex = true;
              cond.push(rt::Ctr(args.len() as u64, *book.name_to_id.get(&*name).unwrap_or(&0), 0));
              free.push((i as u64, args.len() as u64));
              for (j, arg) in args.iter().enumerate() {
                if let lang::Term::Var { ref name } = **arg {
                  vars.push(DynVar { param: i as u64, field: Some(j as u64), erase: name == "*" || name == "*$" });
                  inps.push(name.clone());
                } else {
                  panic!("Sorry, left-hand sides can't have nested constructors yet.");
                }
              }
            }
            lang::Term::Num { numb } => {
              *redex = true;
              cond.push(rt::Num(*numb as u64));
            }
            lang::Term::Var { name } => {
              // HOAS_VAR_OPT: this is an internal optimization that allows us to create less cases
              // when generating Kind2's HOAS type checker. It will cause a left-hand side variable
              // name ending with "$" NOT to match when the argument is a "(Var ...)" constructor.
              // For example, the pattern `(Foo x$) = ...` will match on anything, *except* a
              // constructor named `Var` specifically. This is a non-standard, internal feature,
              // and shouldn't be used by end-users. A more general way to achieve this would be to
              // implement guard patterns on HVM, but, until this isn't done, this optimization
              // will allow Kind2's HOAS to be faster.
              if name.ends_with('$') {
                cond.push(rt::Var(*book.name_to_id.get("Var").unwrap_or(&0)));
              } else {
                cond.push(rt::Var(0));
              }
              vars.push(DynVar { param: i as u64, field: None, erase: name == "*" || name == "*$" });
              inps.push(name.clone());
            }
            _ => {
              panic!("Invalid left-hand side.");
            }
          }
        }

        let term = term_to_dynterm(book, &rule.rhs, &inps);
        let body = build_body(&term, vars.len() as u64);

        Some(DynRule { cond, vars, term, body, free })
      } else {
        None
      }
    })
    .collect();
  DynFun { redex, rules: dynrules }
}

pub fn get_var(mem: &rt::Worker, term: rt::Ptr, var: &DynVar) -> rt::Ptr {
  let DynVar { param, field, erase: _ } = var;
  match field {
    Some(i) => rt::ask_arg(mem, rt::ask_arg(mem, term, *param), *i),
    None => rt::ask_arg(mem, term, *param),
  }
}

pub fn hash<T: Hash>(t: &T) -> u64 {
  let mut s = DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

pub fn build_runtime_arities(rb: &rb::RuleBook) -> Vec<rt::Arity> {
  // TODO: checking arity consistency would be nice, but where?
  fn find_arities(rb: &rb::RuleBook, aris: &mut Vec<rt::Arity>, term: &lang::Term) {
    match term {
      lang::Term::Var { .. } => {}
      lang::Term::Dup { expr, body, .. } => {
        find_arities(rb, aris, expr);
        find_arities(rb, aris, body);
      }
      lang::Term::Lam { body, .. } => {
        find_arities(rb, aris, body);
      }
      lang::Term::Let { expr, body, .. } => {
        find_arities(rb, aris, expr);
        find_arities(rb, aris, body);
      }
      lang::Term::App { func, argm } => {
        find_arities(rb, aris, func);
        find_arities(rb, aris, argm);
      }
      lang::Term::Ctr { name, args } => {
        if let Some(id) = rb.name_to_id.get(name) {
          aris[*id as usize] = rt::Arity(args.len() as u64);
          for arg in args {
            find_arities(rb, aris, arg);
          }
        }
      }
      lang::Term::Num { .. } => {}
      lang::Term::Op2 { val0, val1, .. } => {
        find_arities(rb, aris, val0);
        find_arities(rb, aris, val1);
      }
    }
  }
  let mut aris: Vec<rt::Arity> = iter::repeat_with(|| rt::Arity(0)).take(65535).collect(); // FIXME: hardcoded limit
  for rules_info in rb.rule_group.values() {
    for rule in &rules_info.1 {
      find_arities(rb, &mut aris, &rule.lhs);
      find_arities(rb, &mut aris, &rule.rhs);
    }
  }
  return aris;
}

pub fn build_runtime_functions(book: &rb::RuleBook) -> Vec<Option<rt::Function>> {
  let mut funs: Vec<Option<rt::Function>> = iter::repeat_with(|| None).take(65535).collect(); // FIXME: hardcoded limit
  let i2n = book.id_to_name.clone();
  // The logger function. It allows logging ANY term, not just strings :)
  funs[0] = Some(rt::Function {
    arity: 2,
    stricts: vec![],
    rewriter: Box::new(move |rt, funs, _dups, host, term| {
      let msge = rt::get_loc(term,0);
      rt::normal(rt, funs, msge, Some(&i2n), false);
      println!("{}", rd::as_code(rt, Some(&i2n), msge));
      rt::link(rt, host, rt::ask_arg(rt, term, 1));
      rt::clear(rt, rt::get_loc(term, 0), 2);
      rt::collect(rt, rt::ask_lnk(rt, msge));
      return true;
    }),
  });
  // Creates all the other functions
  for (name, rules_info) in &book.rule_group {
    let fnid = book.name_to_id.get(name).unwrap_or(&0);
    let func = build_runtime_function(book, &rules_info.1);
    funs[*fnid as usize] = Some(func);
  }
  funs
}

pub fn build_runtime_function(book: &rb::RuleBook, rules: &[lang::Rule]) -> rt::Function {
  let dynfun = build_dynfun(book, rules);

  let arity = dynfun.redex.len() as u64;
  let mut stricts = Vec::new();
  for (i, is_redex) in dynfun.redex.iter().enumerate() {
    if *is_redex {
      stricts.push(i as u64);
    }
  }

  let rewriter: rt::Rewriter = Box::new(move |mem, _funs, dups, host, term| {
    // For each argument, if it is a redex and a PAR, apply the cal_par rule
    for (i, redex) in dynfun.redex.iter().enumerate() {
      let i = i as u64;
      if *redex && rt::get_tag(rt::ask_arg(mem, term, i)) == rt::PAR {
        rt::cal_par(mem, host, term, rt::ask_arg(mem, term, i), i);
        return true;
      }
    }

    // For each rule condition vector
    for dynrule in &dynfun.rules {
      // Check if the rule matches
      let mut matched = true;

      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      for (i, cond) in dynrule.cond.iter().enumerate() {
        let i = i as u64;
        match rt::get_tag(*cond) {
          rt::NUM => {
            //println!("Didn't match because of NUM. i={} {} {}", i, rt::get_num(rt::ask_arg(mem, term, i)), rt::get_num(*cond));
            let same_tag = rt::get_tag(rt::ask_arg(mem, term, i)) == rt::NUM;
            let same_val = rt::get_num(rt::ask_arg(mem, term, i)) == rt::get_num(*cond);
            matched = matched && same_tag && same_val;
          }
          rt::CTR => {
            //println!("Didn't match because of CTR. i={} {} {}", i, rt::get_tag(rt::ask_arg(mem, term, i)), rt::get_ext(*cond));
            let same_tag = rt::get_tag(rt::ask_arg(mem, term, i)) == rt::CTR;
            let same_ext = rt::get_ext(rt::ask_arg(mem, term, i)) == rt::get_ext(*cond);
            matched = matched && same_tag && same_ext;
          }
          rt::VAR => {
            if dynfun.redex[i as usize] {
              let not_var = rt::get_tag(rt::ask_arg(mem, term, i)) > rt::VAR;
              let not_hoas_var = rt::get_val(*cond) == 0 || rt::get_ext(rt::ask_arg(mem, term, i)) != rt::get_val(*cond); // See "HOAS_VAR_OPT"
              matched = matched && not_var && not_hoas_var;
            }
          }
          _ => {}
        }
      }

      // If all conditions are satisfied, the rule matched, so we must apply it
      if matched {
        // Increments the gas count
        rt::inc_cost(mem);

        // Builds the right-hand side term (ex: `(Succ (Add a b))`)
        let done = alloc_body(mem, term, &dynrule.vars, dups, &dynrule.body);

        // Links the host location to it
        rt::link(mem, host, done);

        // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
        rt::clear(mem, rt::get_loc(term, 0), dynfun.redex.len() as u64);
        for (i, arity) in &dynrule.free {
          let i = *i as u64;
          rt::clear(mem, rt::get_loc(rt::ask_arg(mem, term, i), 0), *arity);
        }

        // Collects unused variables (none in this example)
        for dynvar @ DynVar { param: _, field: _, erase } in dynrule.vars.iter() {
          if *erase {
            rt::collect(mem, get_var(mem, term, dynvar));
          }
        }

        return true;
      }
    }
    false
  });

  rt::Function { arity, stricts, rewriter }
}

/// Converts a language Term to a runtime Term
pub fn term_to_dynterm(book: &rb::RuleBook, term: &lang::Term, inps: &[String]) -> DynTerm {
  fn convert_oper(oper: &lang::Oper) -> u64 {
    match oper {
      lang::Oper::Add => rt::ADD,
      lang::Oper::Sub => rt::SUB,
      lang::Oper::Mul => rt::MUL,
      lang::Oper::Div => rt::DIV,
      lang::Oper::Mod => rt::MOD,
      lang::Oper::And => rt::AND,
      lang::Oper::Or  => rt::OR,
      lang::Oper::Xor => rt::XOR,
      lang::Oper::Shl => rt::SHL,
      lang::Oper::Shr => rt::SHR,
      lang::Oper::Ltn => rt::LTN,
      lang::Oper::Lte => rt::LTE,
      lang::Oper::Eql => rt::EQL,
      lang::Oper::Gte => rt::GTE,
      lang::Oper::Gtn => rt::GTN,
      lang::Oper::Neq => rt::NEQ,
    }
  }

  #[allow(clippy::identity_op)]
  fn convert_term(
    term: &lang::Term,
    book: &rb::RuleBook,
    depth: u64,
    vars: &mut Vec<String>,
  ) -> DynTerm {
    match term {
      lang::Term::Var { name } => {
        if let Some((idx, _)) = vars.iter().enumerate().rev().find(|(_, var)| var == &name) {
          DynTerm::Var { bidx: idx as u64 }
        } else {
          DynTerm::Glo { glob: hash(name) }
        }
      }
      lang::Term::Dup { nam0, nam1, expr, body } => {
        let eras = (nam0 == "*" || nam0 == "*$", nam1 == "*" || nam1 == "*$");
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(nam0.clone());
        vars.push(nam1.clone());
        let body = Box::new(convert_term(body, book, depth + 2, vars));
        vars.pop();
        vars.pop();
        DynTerm::Dup { eras, expr, body }
      }
      lang::Term::Lam { name, body } => {
        let glob = if rb::is_global_name(name) { hash(name) } else { 0 };
        let eras = name == "*" || name == "*$";
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        DynTerm::Lam { eras, glob, body }
      }
      lang::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        DynTerm::Let { expr, body }
      }
      lang::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, book, depth + 0, vars));
        let argm = Box::new(convert_term(argm, book, depth + 0, vars));
        DynTerm::App { func, argm }
      }
      lang::Term::Ctr { name, args } => {
        let term_func = *book.name_to_id.get(name).unwrap_or_else(|| panic!("Unbound symbol: {}", name));
        let term_args = args.iter().map(|arg| convert_term(arg, book, depth + 0, vars)).collect();
        if *book.ctr_is_cal.get(name).unwrap_or(&false) {
          DynTerm::Cal { func: term_func, args: term_args }
        } else {
          DynTerm::Ctr { func: term_func, args: term_args }
        }
      }
      lang::Term::Num { numb } => DynTerm::Num { numb: *numb },
      lang::Term::Op2 { oper, val0, val1 } => {
        let oper = convert_oper(oper);
        let val0 = Box::new(convert_term(val0, book, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, book, depth + 1, vars));
        DynTerm::Op2 { oper, val0, val1 }
      }
    }
  }

  let mut vars = inps.to_vec();
  convert_term(term, book, 0, &mut vars)
}

pub fn build_body(term: &DynTerm, free_vars: u64) -> Body {
  fn link(nodes: &mut [Node], targ: u64, slot: u64, elem: Elem) {
    nodes[targ as usize][slot as usize] = elem;
    if let Elem::Loc { value, targ: var_targ, slot: var_slot } = elem {
      let tag = rt::get_tag(value);
      if tag <= rt::VAR {
        nodes[var_targ as usize][(var_slot + (tag & 0x01)) as usize] =
          Elem::Loc { value: rt::Arg(0), targ, slot };
      }
    }
  }
  fn alloc_lam(globs: &mut HashMap<u64, u64>, nodes: &mut Vec<Node>, glob: u64) -> u64 {
    if let Some(targ) = globs.get(&glob) {
      *targ
    } else {
      let targ = nodes.len() as u64;
      nodes.push(vec![Elem::Fix { value: 0 }; 2]);
      link(nodes, targ, 0, Elem::Fix { value: rt::Era() });
      if glob != 0 {
        globs.insert(glob, targ);
      }
      targ
    }
  }
  fn gen_elems(
    term: &DynTerm,
    dupk: &mut u64,
    vars: &mut Vec<Elem>,
    globs: &mut HashMap<u64, u64>,
    nodes: &mut Vec<Node>,
    links: &mut Vec<(u64, u64, Elem)>,
  ) -> Elem {
    match term {
      DynTerm::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize]
        } else {
          panic!("Unbound variable.");
        }
      }
      DynTerm::Glo { glob } => {
        let targ = alloc_lam(globs, nodes, *glob);
        Elem::Loc { value: rt::Var(0), targ, slot: 0 }
      }
      DynTerm::Dup { eras: _, expr, body } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix { value: 0 }; 3]);
        //let dupk = dups_count.next();
        links.push((targ, 0, Elem::Fix { value: rt::Era() }));
        links.push((targ, 1, Elem::Fix { value: rt::Era() }));
        let expr = gen_elems(expr, dupk, vars, globs, nodes, links);
        links.push((targ, 2, expr));
        vars.push(Elem::Loc { value: rt::Dp0(*dupk, 0), targ, slot: 0 });
        vars.push(Elem::Loc { value: rt::Dp1(*dupk, 0), targ, slot: 0 });
        *dupk = *dupk + 1;
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        vars.pop();
        vars.pop();
        body
      }
      DynTerm::Let { expr, body } => {
        let expr = gen_elems(expr, dupk, vars, globs, nodes, links);
        vars.push(expr);
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        vars.pop();
        body
      }
      DynTerm::Lam { eras: _, glob, body } => {
        let targ = alloc_lam(globs, nodes, *glob);
        let var = Elem::Loc { value: rt::Var(0), targ, slot: 0 };
        vars.push(var);
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        links.push((targ, 1, body));
        vars.pop();
        Elem::Loc { value: rt::Lam(0), targ, slot: 0 }
      }
      DynTerm::App { func, argm } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix { value: 0 }; 2]);
        let func = gen_elems(func, dupk, vars, globs, nodes, links);
        links.push((targ, 0, func));
        let argm = gen_elems(argm, dupk, vars, globs, nodes, links);
        links.push((targ, 1, argm));
        Elem::Loc { value: rt::App(0), targ, slot: 0 }
      }
      DynTerm::Cal { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![Elem::Fix { value: 0 }; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, globs, nodes, links);
            links.push((targ, i as u64, arg));
          }
          Elem::Loc { value: rt::Cal(args.len() as u64, *func, 0), targ, slot: 0 }
        } else {
          Elem::Fix { value: rt::Cal(0, *func, 0) }
        }
      }
      DynTerm::Ctr { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![Elem::Fix { value: 0 }; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, globs, nodes, links);
            links.push((targ, i as u64, arg));
          }
          Elem::Loc { value: rt::Ctr(args.len() as u64, *func, 0), targ, slot: 0 }
        } else {
          Elem::Fix { value: rt::Ctr(0, *func, 0) }
        }
      }
      DynTerm::Num { numb } => Elem::Fix { value: rt::Num(*numb as u64) },
      DynTerm::Op2 { oper, val0, val1 } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix { value: 0 }; 2]);
        let val0 = gen_elems(val0, dupk, vars, globs, nodes, links);
        links.push((targ, 0, val0));
        let val1 = gen_elems(val1, dupk, vars, globs, nodes, links);
        links.push((targ, 1, val1));
        Elem::Loc { value: rt::Op2(*oper, 0), targ, slot: 0 }
      }
    }
  }

  let mut links: Vec<(u64, u64, Elem)> = Vec::new();
  let mut nodes: Vec<Node> = Vec::new();
  let mut globs: HashMap<u64, u64> = HashMap::new();
  let mut vars: Vec<Elem> = (0..free_vars).map(|i| Elem::Ext { index: i }).collect();
  let mut dupk: u64 = 0;

  let elem = gen_elems(term, &mut dupk, &mut vars, &mut globs, &mut nodes, &mut links);
  for (targ, slot, elem) in links {
    link(&mut nodes, targ, slot, elem);
  }

  (elem, nodes, dupk)
}

static mut ALLOC_BODY_WORKSPACE: &mut [u64] = &mut [0; 256 * 256 * 256]; // to avoid dynamic allocations
pub fn alloc_body(
  mem: &mut rt::Worker,
  term: rt::Ptr,
  vars: &[DynVar],
  dups: &mut u64,
  body: &Body,
) -> rt::Ptr {
  unsafe {
    let hosts = &mut ALLOC_BODY_WORKSPACE;
    let (elem, nodes, dupk) = body;
    fn elem_to_lnk(
      mem: &mut rt::Worker,
      term: rt::Ptr,
      vars: &[DynVar],
      dups: &mut u64,
      elem: &Elem,
    ) -> rt::Ptr {
      unsafe {
        let hosts = &mut ALLOC_BODY_WORKSPACE;
        match elem {
          Elem::Fix { value } => *value,
          Elem::Ext { index } => get_var(mem, term, &vars[*index as usize]),
          Elem::Loc { value, targ, slot } => {
            let mut val = value + hosts[*targ as usize] + slot;
            // should be changed if the pointer format changes
            if rt::get_tag(*value) == rt::DP0 {
              val += (*dups & 0xFFFFFF) * rt::EXT;
            }
            if rt::get_tag(*value) == rt::DP1 {
              val += (*dups & 0xFFFFFF) * rt::EXT;
            }
            val
          }
        }
      }
    }
    nodes.iter().enumerate().for_each(|(i, node)| {
      hosts[i] = rt::alloc(mem, node.len() as u64);
    });
    nodes.iter().enumerate().for_each(|(i, node)| {
      let host = hosts[i] as usize;
      node.iter().enumerate().for_each(|(j, elem)| {
        let lnk = elem_to_lnk(mem, term, vars, dups, elem);
        if let Elem::Ext { .. } = elem {
          rt::link(mem, (host + j) as u64, lnk);
        } else {
          mem.node[host + j] = lnk;
        }
      });
    });
    let done = elem_to_lnk(mem, term, vars, dups, elem);
    *dups += dupk;
    return done;
  }
}

pub fn alloc_closed_dynterm(mem: &mut rt::Worker, term: &DynTerm) -> u64 {
  let mut dups = 0;
  let host = rt::alloc(mem, 1);
  let body = build_body(term, 0);
  let term = alloc_body(mem, 0, &[], &mut dups, &body);
  rt::link(mem, host, term);
  host
}

pub fn alloc_term(mem: &mut rt::Worker, book: &rb::RuleBook, term: &lang::Term) -> u64 {
  alloc_closed_dynterm(mem, &term_to_dynterm(book, term, &vec![]))
}

// Evaluates a HVM term to normal form
pub fn eval_code(
  call: &lang::Term,
  code: &str,
  debug: bool,
) -> Result<(String, u64, u64, u64), String> {
  let mut worker = rt::new_worker();

  // Parses and reads the input file
  let file = lang::read_file(code)?;

  // Converts the HVM "file" to a Rulebook
  let book = rb::gen_rulebook(&file);

  // Builds functions
  let funs = build_runtime_functions(&book);

  // Builds arities
  worker.aris = build_runtime_arities(&book);

  // Allocates the main term
  let host = alloc_term(&mut worker, &book, call);

  // Normalizes it
  let init = Instant::now();
  rt::run_io(&mut worker, &funs, host, Some(&book.id_to_name), debug);
  rt::normal(&mut worker, &funs, host, Some(&book.id_to_name), debug);
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a string
  let code = format!("{}", rd::as_term(&worker, Some(&book.id_to_name), host));

  // Returns the normal form and the gas cost
  Ok((code, worker.cost, worker.size, time))
}
