// Moving Lambolt Terms to/from runtime, and building dynamic functions.
// TODO: "dups" still needs to be moved out on alloc_body etc.

#![allow(clippy::identity_op)]

use crate::rulebook as rb;
use crate::lambolt as lb;
use crate::runtime as rt;
use crate::runtime::{Lnk, Worker};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug)]
pub enum DynTerm {
  Var {
    bidx: u64,
  },
  Dup {
    expr: Box<DynTerm>,
    body: Box<DynTerm>,
  },
  Let {
    expr: Box<DynTerm>,
    body: Box<DynTerm>,
  },
  Lam {
    body: Box<DynTerm>,
  },
  App {
    func: Box<DynTerm>,
    argm: Box<DynTerm>,
  },
  Cal {
    func: u64,
    args: Vec<Box<DynTerm>>,
  },
  Ctr {
    func: u64,
    args: Vec<Box<DynTerm>>,
  },
  U32 {
    numb: u32,
  },
  Op2 {
    oper: u64,
    val0: Box<DynTerm>,
    val1: Box<DynTerm>,
  },
}

// The right-hand side of a rule, "digested" or "pre-filled" in a way that is almost ready to be
// pasted into the runtime's memory, except for some minor adjustments: internal links need to be
// adjusted taking in account the index where each node is actually allocated, and external
// variables must be linked. This structure helps moving as much computation from the interpreter
// to the static time (i.e., when generating the dynfun closure) as possible.
pub type Body = (Elem,Vec<Node>);           // The pre-filled body (TODO: can the Node Vec be unboxed?)
pub type Node = Vec<Elem>;                  // A node on the pre-filled body
#[derive(Copy,Clone,Debug)] pub enum Elem { // An element of a node
  Fix{value: u64},                          // Fixed value, doesn't require adjuemtn
  Loc{value: u64, targ: u64, slot: u64},    // Local link, requires adjustment
  Ext{index: u64},                          // Link to an external variable
}


#[derive(Debug)]
pub struct DynVar {
  pub param: u64,
  pub field: Option<u64>,
  pub erase: bool,
}

#[derive(Debug)]
pub struct DynRule {
  pub cond: Vec<rt::Lnk>,
  pub vars: Vec<DynVar>,
  pub body: Body,
  pub free: Vec<(u64, u64)>,
}

#[derive(Debug)]
pub struct DynFun {
  pub redex: Vec<bool>,
  pub rules: Vec<DynRule>,
}

// Given a RuleBook file and a function name, builds a dynamic Rust closure that applies that
// function to the runtime memory buffer directly. This process is complex, so I'll write a lot of
// comments here. All comments will be based on the following Lambolt example:
//   (Add (Succ a) b) = (Succ (Add a b))
//   (Add (Zero)   b) = b
pub fn build_dynfun(comp: &rb::RuleBook, rules: &Vec<lb::Rule>) -> DynFun {

  let mut redex;
  if let lb::Term::Ctr { ref name, ref args } = *rules[0].lhs {
    redex = vec![false; args.len()];
  } else {
    panic!("Invalid left-hand side: {}", rules[0].lhs);
  }

  let mut cond_vec = Vec::new();
  let mut vars_vec = Vec::new();
  let mut free_vec = Vec::new();
  let mut body_vec = Vec::new();

  for rule in rules {
    if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
      let mut cond = Vec::new();
      let mut vars = Vec::new();
      let mut free = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        match &**arg {
          lb::Term::Ctr { name, args } => {
            redex[i] = true;
            cond.push(rt::Ctr(args.len() as u64, *comp.name_to_id.get(&*name).unwrap_or(&0), 0));
            free.push((i as u64, args.len() as u64));
            for j in 0..args.len() {
              if let lb::Term::Var { ref name } = *args[j] {
                vars.push(DynVar { param: i as u64, field: Some(j as u64), erase: name == "*", });
              }
            }
          }
          lb::Term::U32 { numb } => {
            redex[i] = true;
            cond.push(rt::U_32(*numb as u64));
          }
          lb::Term::Var { name } => {
            vars.push(DynVar { param: i as u64, field: None, erase: name == "*" });
          }
          _ => {}
        }
      }

      let mut dups = 0;
      let term = term_to_dynterm(comp, &rule.rhs, vars.len() as u64);
      let body = build_body(&term, vars.len() as u64, &mut dups);
      
      body_vec.push(body);
      cond_vec.push(cond);
      vars_vec.push(vars);
      free_vec.push(free);
    }
  }

  let mut dynrules = Vec::new();
  for (((cond, vars), body), free) in cond_vec.into_iter().zip(vars_vec).zip(body_vec).zip(free_vec) {
    dynrules.push(DynRule { cond, vars, body, free });
  }
  return DynFun { redex, rules: dynrules };
}

pub fn build_runtime_function(comp: &rb::RuleBook, rules: &Vec<lb::Rule>) -> rt::Function {
  let dynfun = build_dynfun(comp, rules);

  let stricts = dynfun.redex.clone();

  let rewriter: rt::Rewriter = Box::new(move |mem, host, term| {

    // Gets the left-hand side arguments (ex: `(Succ a)` and `b`)
    //let mut args = Vec::new();
    //for i in 0 .. dynfun.redex.len() {
      //args.push(rt::ask_arg(mem, term, i as u64));
    //}

    // For each argument, if it is redexand a PAR, apply the cal_par rule
    for i in 0 .. dynfun.redex.len() {
      let i = i as u64;
      if dynfun.redex[i as usize] && rt::get_tag(rt::ask_arg(mem,term,i)) == rt::PAR {
        rt::cal_par(mem, host, term, rt::ask_arg(mem,term,i), i as u64);
        return true;
      }
    }

    // For each rule condition vector
    for dynrule in &dynfun.rules {
      // Check if the rule matches
      let mut matched = true;

      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      //println!(">> testing conditions... total: {} conds", dynrule.cond.len());
      for (i, cond) in dynrule.cond.iter().enumerate() {
        let i = i as u64;
        match rt::get_tag(*cond) {
          rt::U32 => {
            //println!(">>> cond demands U32 {} at {}", rt::get_val(*cond), i);
            let same_tag = rt::get_tag(rt::ask_arg(mem,term,i)) == rt::U32;
            let same_val = rt::get_val(rt::ask_arg(mem,term,i)) == rt::get_val(*cond);
            matched = matched && same_tag && same_val;
          }
          rt::CTR => {
            //println!(">>> cond demands CTR {} at {}", rt::get_ext(*cond), i);
            let same_tag = rt::get_tag(rt::ask_arg(mem,term,i)) == rt::CTR;
            let same_ext = rt::get_ext(rt::ask_arg(mem,term,i)) == rt::get_ext(*cond);
            matched = matched && same_tag && same_ext;
          }
          _ => {}
        }
      }

      //println!(">> matched? {}", matched);

      // If all conditions are satisfied, the rule matched, so we must apply it
      if matched {
        // Increments the gas count
        rt::inc_cost(mem);
        
        // Gets all the left-hand side vars (ex: `a` and `b`).
        let mut vars = Vec::new();
        for DynVar {param, field, erase} in &dynrule.vars {
          match field {
            Some(i) => { vars.push(rt::ask_arg(mem, rt::ask_arg(mem,term,*param), *i)) }
            None    => { vars.push(rt::ask_arg(mem,term,*param)) }
          }
        }

        // FIXME: `dups` must be global to properly color the fan nodes, but Rust complains about
        // mutating borrowed variables. Until this is fixed, the language will be very restrict.
        let mut dups = 0;

        // Builds the right-hand side term (ex: `(Succ (Add a b))`)
        //println!("building {:?}", &dynrule.body);
        //let done = alloc_dynterm(mem, &dynrule.body, &mut vars, &mut dups);
        let done = alloc_body(mem, &dynrule.body, &mut vars);

        // Links the host location to it
        rt::link(mem, host, done);

        // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
        rt::clear(mem, rt::get_loc(term, 0), dynfun.redex.len() as u64);
        for (i, arity) in &dynrule.free {
          let i = *i as u64;
          rt::clear(mem, rt::get_loc(rt::ask_arg(mem,term,i), 0), *arity);
        }

        // Collects unused variables (none in this example)
        for (i, DynVar {param, field, erase}) in dynrule.vars.iter().enumerate() {
          if *erase {
            rt::collect(mem, vars[i]);
          }
        }

        return true;
      }
    }
    return false;
  });

  return rt::Function { stricts, rewriter };
}

pub fn build_runtime_functions(comp: &rb::RuleBook) -> HashMap<u64, rt::Function> {
  let mut fns: HashMap<u64, rt::Function> = HashMap::new();
  for (name, rules_info) in &comp.func_rules {
    let id = comp.name_to_id.get(name).unwrap_or(&0);
    let rules = &rules_info.1;
    let ff = build_runtime_function(comp, rules);
    fns.insert(*id, ff);
  }
  return fns;
}

/// Converts a Lambolt Term to a Runtime Term
pub fn term_to_dynterm(comp: &rb::RuleBook, term: &lb::Term, free_vars: u64) -> DynTerm {
  fn convert_oper(oper: &lb::Oper) -> u64 {
    match oper {
      lb::Oper::Add => rt::ADD,
      lb::Oper::Sub => rt::SUB,
      lb::Oper::Mul => rt::MUL,
      lb::Oper::Div => rt::DIV,
      lb::Oper::Mod => rt::MOD,
      lb::Oper::And => rt::AND,
      lb::Oper::Or  => rt::OR,
      lb::Oper::Xor => rt::XOR,
      lb::Oper::Shl => rt::SHL,
      lb::Oper::Shr => rt::SHR,
      lb::Oper::Ltn => rt::LTN,
      lb::Oper::Lte => rt::LTE,
      lb::Oper::Eql => rt::EQL,
      lb::Oper::Gte => rt::GTE,
      lb::Oper::Gtn => rt::GTN,
      lb::Oper::Neq => rt::NEQ,
    }
  }
  fn convert_term(
    term: &lb::Term,
    comp: &rb::RuleBook,
    depth: u64,
    vars: &mut Vec<String>,
  ) -> DynTerm {
    match term {
      lb::Term::Var { name } => {
        for i in 0 .. vars.len() {
          let j = vars.len() - i - 1;
          if vars[j] == *name {
            return DynTerm::Var { bidx: j as u64 }
          }
        }
        panic!("Unbound variable: '{}'.", name);
      }
      lb::Term::Dup {
        nam0,
        nam1,
        expr,
        body,
      } => {
        let expr = Box::new(convert_term(expr, comp, depth + 0, vars));
        vars.push(nam0.clone());
        vars.push(nam1.clone());
        let body = Box::new(convert_term(body, comp, depth + 2, vars));
        vars.pop();
        vars.pop();
        DynTerm::Dup { expr, body }
      }
      lb::Term::Lam { name, body } => {
        vars.push(name.clone());
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.pop();
        DynTerm::Lam { body }
      }
      lb::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, comp, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.pop();
        DynTerm::Let { expr, body }
      }
      lb::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, comp, depth + 0, vars));
        let argm = Box::new(convert_term(argm, comp, depth + 0, vars));
        DynTerm::App { func, argm }
      }
      lb::Term::Ctr { name, args } => {
        let term_func = comp.name_to_id[name];
        let mut term_args: Vec<Box<DynTerm>> = Vec::new();
        for arg in args {
          term_args.push(Box::new(convert_term(arg, comp, depth + 0, vars)));
        }
        if *comp.ctr_is_cal.get(name).unwrap_or(&false) {
          DynTerm::Cal {
            func: term_func,
            args: term_args,
          }
        } else {
          DynTerm::Ctr {
            func: term_func,
            args: term_args,
          }
        }
      }
      lb::Term::U32 { numb } => DynTerm::U32 { numb: *numb },
      lb::Term::Op2 { oper, val0, val1 } => {
        let oper = convert_oper(oper);
        let val0 = Box::new(convert_term(val0, comp, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, comp, depth + 1, vars));
        DynTerm::Op2 { oper, val0, val1 }
      }
    }
  }

  let mut vars = Vec::new();
  for i in 0 .. free_vars {
    vars.push(format!("x{}", i).to_string());
  }
  convert_term(term, comp, 0, &mut vars)
}

/// Reads back a Lambolt term from Runtime's memory
// TODO: we should readback as a lambolt::Term, not as a string
pub fn readback_as_code(mem: &Worker, comp: &rb::RuleBook, host: u64) -> String {
  struct CtxName<'a> {
    mem: &'a Worker,
    names: &'a mut HashMap<Lnk, String>,
    seen: &'a mut HashSet<Lnk>,
    count: &'a mut u32,
  }

  fn name(ctx: &mut CtxName, term: Lnk, depth: u32) {
    if ctx.seen.contains(&term) {
      return;
    };

    ctx.seen.insert(term);

    match rt::get_tag(term) {
      rt::LAM => {
        let param = rt::ask_arg(ctx.mem, term, 0);
        let body = rt::ask_arg(ctx.mem, term, 1);
        if rt::get_tag(param) != rt::ERA {
          let var = rt::Var(rt::get_loc(term, 0));
          *ctx.count += 1;
          ctx.names.insert(var, format!("x{}", *ctx.count));
        };
        name(ctx, body, depth + 1);
      }
      rt::APP => {
        let lam = rt::ask_arg(ctx.mem, term, 0);
        let arg = rt::ask_arg(ctx.mem, term, 1);
        name(ctx, lam, depth + 1);
        name(ctx, arg, depth + 1);
      }
      rt::PAR => {
        let arg0 = rt::ask_arg(ctx.mem, term, 0);
        let arg1 = rt::ask_arg(ctx.mem, term, 1);
        name(ctx, arg0, depth + 1);
        name(ctx, arg1, depth + 1);
      }
      rt::DP0 => {
        let arg = rt::ask_arg(ctx.mem, term, 2);
        name(ctx, arg, depth + 1);
      }
      rt::DP1 => {
        let arg = rt::ask_arg(ctx.mem, term, 2);
        name(ctx, arg, depth + 1);
      }
      rt::OP2 => {
        let arg0 = rt::ask_arg(ctx.mem, term, 0);
        let arg1 = rt::ask_arg(ctx.mem, term, 1);
        name(ctx, arg0, depth + 1);
        name(ctx, arg1, depth + 1);
      }
      rt::U32 => {}
      rt::CTR | rt::CAL => {
        let arity = rt::get_ari(term);
        for i in 0..arity {
          let arg = rt::ask_arg(ctx.mem, term, i);
          name(ctx, arg, depth + 1);
        }
      }
      default => {}
    }
  }

  struct CtxGo<'a> {
    mem: &'a Worker,
    comp: &'a rb::RuleBook,
    names: &'a HashMap<Lnk, String>,
    seen: &'a HashSet<Lnk>,
    // count: &'a mut u32,
  }

  // TODO: more efficient, immutable data structure
  // Note: Because of clone? Use a bit-string:
  // struct BitString { O{...}, I{...}, E }
  #[derive(Clone)]
  struct Stacks {
    stacks: HashMap<Lnk, Vec<bool>>,
  }

  impl Stacks {
    fn new() -> Stacks {
      Stacks {
        stacks: HashMap::new(),
      }
    }
    fn get(&self, col: Lnk) -> Option<&Vec<bool>> {
      self.stacks.get(&col)
    }
    fn pop(&self, col: Lnk) -> Stacks {
      let mut stacks = self.stacks.clone();
      let stack = stacks.entry(col).or_insert_with(Vec::new);
      stack.pop();
      Stacks { stacks }
    }
    fn push(&self, col: Lnk, val: bool) -> Stacks {
      let mut stacks = self.stacks.clone();
      let stack = stacks.entry(col).or_insert_with(Vec::new);
      stack.push(val);
      Stacks { stacks }
    }
  }

  fn go(ctx: &mut CtxGo, stacks: Stacks, term: Lnk, depth: u32) -> String {
    // TODO: seems like the "seen" map isn't used anymore here?
    // Should investigate if it is needed or not.

    //if ctx.seen.contains(&term) {
    //"@".to_string()
    //} else {
    match rt::get_tag(term) {
      rt::LAM => {
        let body = rt::ask_arg(ctx.mem, term, 1);
        let body_txt = go(ctx, stacks, body, depth + 1);
        let arg = rt::ask_arg(ctx.mem, term, 0);
        let name_txt = if rt::get_tag(arg) == rt::ERA {
          "~"
        } else {
          let var = rt::Var(rt::get_loc(term, 0));
          ctx.names.get(&var).map(|s| s as &str).unwrap_or("?")
        };
        format!("λ{} {}", name_txt, body_txt)
      }
      rt::APP => {
        let func = rt::ask_arg(ctx.mem, term, 0);
        let argm = rt::ask_arg(ctx.mem, term, 1);
        let func_txt = go(ctx, stacks.clone(), func, depth + 1);
        let argm_txt = go(ctx, stacks, argm, depth + 1);
        format!("({} {})", func_txt, argm_txt)
      }
      rt::PAR => {
        let col = rt::get_ext(term);
        let empty = &Vec::new();
        let stack = stacks.get(col).unwrap_or(empty);
        if let Some(val) = stack.last() {
          let arg_idx = *val as u64;
          let val = rt::ask_arg(ctx.mem, term, arg_idx);
          go(ctx, stacks.pop(col), val, depth + 1)
        } else {
          let val0 = rt::ask_arg(ctx.mem, term, 0);
          let val1 = rt::ask_arg(ctx.mem, term, 1);
          let val0_txt = go(ctx, stacks.clone(), val0, depth + 1);
          let val1_txt = go(ctx, stacks, val1, depth + 1);
          format!("<{} {}>", val0_txt, val1_txt)
        }
      }
      rt::DP0 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        go(ctx, stacks.push(col, false), val, depth + 1)
      }
      rt::DP1 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        go(ctx, stacks.push(col, true), val, depth + 1)
      }
      rt::OP2 => {
        let op = rt::get_ext(term);
        let op_txt = match op {
          rt::ADD => "+",
          rt::SUB => "-",
          rt::MUL => "*",
          rt::DIV => "/",
          rt::MOD => "%",
          rt::AND => "&",
          rt::OR  => "|",
          rt::XOR => "^",
          rt::SHL => "<<",
          rt::SHR => ">>",
          rt::LTN => "<",
          rt::LTE => "<=",
          rt::EQL => "==",
          rt::GTE => ">=",
          rt::GTN => ">",
          rt::NEQ => "!=",
          default => panic!("unknown operation"),
        };
        let val0 = rt::ask_arg(ctx.mem, term, 0);
        let val1 = rt::ask_arg(ctx.mem, term, 1);
        let val0_txt = go(ctx, stacks.clone(), val0, depth + 1);
        let val1_txt = go(ctx, stacks, val1, depth + 1);
        format!("({} {} {})", op_txt, val0_txt, val1_txt)
      }
      rt::U32 => {
        format!("{}", rt::get_val(term))
      }
      rt::CTR | rt::CAL => {
        let func = rt::get_ext(term);
        let arit = rt::get_ari(term);
        let args_txt = (0..arit)
          .map(|i| {
            let arg = rt::ask_arg(ctx.mem, term, i);
            go(ctx, stacks.clone(), arg, depth + 1)
          })
          .map(|x| format!(" {}", x))
          .collect::<String>();
        let name = ctx
          .comp
          .id_to_name
          .get(&func)
          .map(String::to_string)
          .unwrap_or_else(|| format!("${}", func));
        format!("({}{})", name, args_txt)
      }
      rt::VAR => ctx
        .names
        .get(&term)
        .map(|x| x.to_string())
        .unwrap_or_else(|| format!("^{}", rt::get_loc(term, 0))),
      rt::ARG => "!".to_string(),
      rt::ERA => "~".to_string(),
      default => {
        format!("?({})", rt::get_tag(term))
      }
    }
    //}
  }

  let term = rt::ask_lnk(mem, host);

  let mut names = HashMap::<Lnk, String>::new();
  let mut seen = HashSet::<Lnk>::new();
  let mut count: u32 = 0;

  let ctx = &mut CtxName {
    mem,
    names: &mut names,
    seen: &mut seen,
    count: &mut count,
  };
  name(ctx, term, 0);

  let ctx = &mut CtxGo {
    mem,
    comp,
    names: &names,
    seen: &seen,
  };
  let stacks = Stacks::new();

  go(ctx, stacks, term, 0)
}

pub fn build_body(term: &DynTerm, free_vars: u64, dups: &mut u64) -> Body {
  fn link(nodes: &mut Vec<Node>, targ: u64, slot: u64, elem: Elem) {
    nodes[targ as usize][slot as usize] = elem;
    if let Elem::Loc{value, targ: var_targ, slot: var_slot} = elem {
      let tag = rt::get_tag(value);
      if tag <= rt::VAR {
        nodes[var_targ as usize][(var_slot + (tag & 0x01)) as usize] = Elem::Loc{value: rt::Arg(0), targ, slot};
      }
    }
  }
  fn go(term: &DynTerm, vars: &mut Vec<Elem>, dups: &mut u64, nodes: &mut Vec<Node>) -> Elem {
    match term {
      DynTerm::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize].clone()
        } else {
          panic!("Unbound variable.");
        }
      }
      DynTerm::Dup { expr, body } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix{value: 0}; 3]);
        let dupk = *dups;
        *dups += 1;
        link(nodes, targ, 0, Elem::Fix{value: rt::Era()});
        link(nodes, targ, 1, Elem::Fix{value: rt::Era()});
        let expr = go(expr, vars, dups, nodes);
        link(nodes, targ, 2, expr);
        vars.push(Elem::Loc{value: rt::Dp0(dupk, 0), targ, slot: 0});
        vars.push(Elem::Loc{value: rt::Dp1(dupk, 0), targ, slot: 0});
        let body = go(body, vars, dups, nodes);
        vars.pop();
        vars.pop();
        body
      }
      DynTerm::Let { expr, body } => {
        let expr = go(expr, vars, dups, nodes);
        vars.push(expr);
        let body = go(body, vars, dups, nodes);
        vars.pop();
        body
      }
      DynTerm::Lam { body } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix{value: 0}; 2]);
        link(nodes, targ, 0, Elem::Fix{value: rt::Era()});
        vars.push(Elem::Loc{value: rt::Var(0), targ, slot: 0});
        let body = go(body, vars, dups, nodes);
        link(nodes, targ, 1, body);
        vars.pop();
        Elem::Loc{value: rt::Lam(0), targ, slot: 0}
      }
      DynTerm::App { func, argm } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix{value: 0}; 2]);
        let func = go(func, vars, dups, nodes);
        link(nodes, targ, 0, func);
        let argm = go(argm, vars, dups, nodes);
        link(nodes, targ, 1, argm);
        Elem::Loc{value: rt::App(0), targ, slot: 0}
      }
      DynTerm::Cal { func, args } => {
        if args.len() > 0 {
          let targ = nodes.len() as u64;
          nodes.push(vec![Elem::Fix{value: 0}; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = go(arg, vars, dups, nodes);
            link(nodes, targ, i as u64, arg);
          }
          Elem::Loc{value: rt::Cal(args.len() as u64, *func, 0), targ, slot: 0}
        } else {
          Elem::Fix{value: rt::Cal(0, *func, 0)}
        }
      }
      DynTerm::Ctr { func, args } => {
        if args.len() > 0 {
          let targ = nodes.len() as u64;
          nodes.push(vec![Elem::Fix{value: 0}; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = go(arg, vars, dups, nodes);
            link(nodes, targ, i as u64, arg);
          }
          Elem::Loc{value: rt::Ctr(args.len() as u64, *func, 0), targ, slot: 0}
        } else {
          Elem::Fix{value: rt::Ctr(0, *func, 0)}
        }
      }
      DynTerm::U32 { numb } => {
        Elem::Fix{value: rt::U_32(*numb as u64)}
      }
      DynTerm::Op2 { oper, val0, val1 } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![Elem::Fix{value: 0}; 2]);
        let val0 = go(val0, vars, dups, nodes);
        link(nodes, targ, 0, val0);
        let val1 = go(val1, vars, dups, nodes);
        link(nodes, targ, 1, val1);
        Elem::Loc{value: rt::Op2(*oper, 0), targ, slot: 0}
      }
    }
  }
  let mut nodes : Vec<Node> = Vec::new();
  let mut vars : Vec<Elem> = Vec::new();
  for i in 0 .. free_vars {
    vars.push(Elem::Ext{index: i});
  }
  let elem = go(term, &mut vars, dups, &mut nodes);
  return (elem, nodes);
}

pub fn alloc_body(mem: &mut rt::Worker, body: &Body, vars: &[rt::Lnk]) -> rt::Lnk {
  let (elem, nodes) = body;
  let mut hosts = vec![0; nodes.len()];
  for i in 0 .. nodes.len() {
    hosts[i] = rt::alloc(mem, nodes[i].len() as u64);
  }
  for i in 0 .. nodes.len() as u64 {
    let node = &nodes[i as usize];
    let host = hosts[i as usize];
    for j in 0 .. node.len() as u64 {
      match &node[j as usize] {
        Elem::Fix{value} => {
          mem.node[(host + j) as usize] = *value;
        }
        Elem::Ext{index} => {
          let lnk = vars[*index as usize];
          mem.node[(host + j) as usize] = lnk;
          if rt::get_tag(lnk) <= rt::VAR {
            mem.node[rt::get_loc(lnk, rt::get_tag(lnk) & 0x01) as usize] = rt::Arg(host + j);
          }
          //rt::link(mem, host + j, value);
        }
        Elem::Loc{value, targ, slot} => {
          mem.node[(host + j) as usize] = value + hosts[*targ as usize] + slot;
        }
      }
    }
  }
  match elem {
    Elem::Fix{value} => *value,
    Elem::Ext{index} => *index,
    Elem::Loc{value, targ, slot} => value + hosts[*targ as usize] + slot,
  }
}

pub fn alloc_dynterm(mem: &mut rt::Worker, term: &DynTerm, vars: &mut Vec<u64>, dups: &mut u64) -> rt::Lnk {
  let body = build_body(term, vars.len() as u64, dups);
  return alloc_body(mem, &body, vars);
}

pub fn alloc_closed_dynterm(mem: &mut rt::Worker, term: &DynTerm) -> u64 {
  let mut dups = 0;
  let host = rt::alloc(mem, 1);
  let term = alloc_dynterm(mem, term, &mut Vec::new(), &mut dups);
  rt::link(mem, host, term);
  host
}

pub fn alloc_term(mem: &mut rt::Worker, comp: &rb::RuleBook, term: &lb::Term) -> u64 {
  return alloc_closed_dynterm(mem, &term_to_dynterm(comp, term, 0));
}

// This function isn't needed anymore; I'll leave it as a comment for a few commits
//pub fn old_alloc_dynterm(mem: &mut rt::Worker, term: &DynTerm, vars: &mut Vec<u64>, dups: &mut u64) -> rt::Lnk {
  //match term {
    //DynTerm::Var { bidx } => {
      //if *bidx < vars.len() as u64 {
        //vars[*bidx as usize]
      //} else {
        //panic!("Unbound variable.");
      //}
    //}
    //DynTerm::Dup { expr, body } => {
      //let node = rt::alloc(mem, 3);
      //let dupk = *dups;
      //*dups += 1;
      //rt::link(mem, node + 0, rt::Era());
      //rt::link(mem, node + 1, rt::Era());
      //let expr = alloc_dynterm(mem, expr, vars, dups);
      //rt::link(mem, node + 2, expr);
      //vars.push(rt::Dp0(dupk, node));
      //vars.push(rt::Dp1(dupk, node));
      //let body = alloc_dynterm(mem, body, vars, dups);
      //vars.pop();
      //vars.pop();
      //body
    //}
    //DynTerm::Let { expr, body } => {
      //let expr = alloc_dynterm(mem, expr, vars, dups);
      //vars.push(expr);
      //let body = alloc_dynterm(mem, body, vars, dups);
      //vars.pop();
      //body
    //}
    //DynTerm::Lam { body } => {
      //let node = rt::alloc(mem, 2);
      //rt::link(mem, node + 0, rt::Era());
      //vars.push(rt::Var(node));
      //let body = alloc_dynterm(mem, body, vars, dups);
      //rt::link(mem, node + 1, body);
      //vars.pop();
      //rt::Lam(node)
    //}
    //DynTerm::App { func, argm } => {
      //let node = rt::alloc(mem, 2);
      //let func = alloc_dynterm(mem, func, vars, dups);
      //rt::link(mem, node + 0, func);
      //let argm = alloc_dynterm(mem, argm, vars, dups);
      //rt::link(mem, node + 1, argm);
      //rt::App(node)
    //}
    //DynTerm::Cal { func, args } => {
      //let size = args.len() as u64;
      //let node = rt::alloc(mem, size);
      //for (i, arg) in args.iter().enumerate() {
        //let arg_lnk = alloc_dynterm(mem, arg, vars, dups);
        //rt::link(mem, node + i as u64, arg_lnk);
      //}
      //rt::Cal(size, *func, node)
    //}
    //DynTerm::Ctr { func, args } => {
      //let size = args.len() as u64;
      //let node = rt::alloc(mem, size);
      //for (i, arg) in args.iter().enumerate() {
        //let arg_lnk = alloc_dynterm(mem, arg, vars, dups);
        //rt::link(mem, node + i as u64, arg_lnk);
      //}
      //rt::Ctr(size, *func, node)
    //}
    //DynTerm::U32 { numb } => {
      //rt::U_32(*numb as u64)
    //}
    //DynTerm::Op2 { oper, val0, val1 } => {
      //let node = rt::alloc(mem, 2);
      //let val0 = alloc_dynterm(mem, val0, vars, dups);
      //rt::link(mem, node + 0, val0);
      //let val1 = alloc_dynterm(mem, val1, vars, dups);
      //rt::link(mem, node + 1, val1);
      //rt::Op2(*oper, node)
    //}
  //}
//}

