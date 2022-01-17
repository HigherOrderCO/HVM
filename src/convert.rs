// TODO: we should readback as a lambolt::Term, not as a string

#![allow(clippy::identity_op)]

use crate::compilable as cm;
use crate::lambolt as lb;
use crate::runtime as rt;
use crate::runtime::{Lnk, Worker};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Converts a Lambolt term to a Runtime-ready term
pub fn lambolt_to_runtime(term: &lb::Term, comp: &cm::Compilable) -> rt::Term {
  fn convert_oper(oper: &lb::Oper) -> u64 {
    match oper {
      lb::Oper::Add => rt::ADD,
      lb::Oper::Sub => rt::SUB,
      lb::Oper::Mul => rt::MUL,
      lb::Oper::Div => rt::DIV,
      lb::Oper::Mod => rt::MOD,
      lb::Oper::And => rt::AND,
      lb::Oper::Or => rt::OR,
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
    comp: &cm::Compilable,
    depth: u64,
    vars: &mut HashMap<String, u64>,
  ) -> rt::Term {
    match term {
      lb::Term::Var { name } => {
        if let Some(var_depth) = vars.get(name) {
          rt::Term::Var { bidx: *var_depth }
        } else {
          panic!("Unbound variable.");
        }
      }
      lb::Term::Dup {
        nam0,
        nam1,
        expr,
        body,
      } => {
        let expr = Box::new(convert_term(expr, comp, depth + 0, vars));
        vars.insert(nam0.clone(), depth + 0);
        vars.insert(nam1.clone(), depth + 1);
        let body = Box::new(convert_term(body, comp, depth + 2, vars));
        vars.remove(nam0);
        vars.remove(nam1);
        rt::Term::Dup { expr, body }
      }
      lb::Term::Lam { name, body } => {
        vars.insert(name.clone(), depth);
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.remove(name);
        rt::Term::Lam { body }
      }
      lb::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, comp, depth + 0, vars));
        vars.insert(name.clone(), depth);
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.remove(name);
        rt::Term::Let { expr, body }
      }
      lb::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, comp, depth + 0, vars));
        let argm = Box::new(convert_term(argm, comp, depth + 0, vars));
        rt::Term::App { func, argm }
      }
      lb::Term::Ctr { name, args } => {
        let mut new_args: Vec<rt::Term> = Vec::new();
        for arg in args {
          new_args.push(convert_term(arg, comp, depth + 0, vars));
        }
        rt::Term::Ctr {
          func: comp.name_to_id[name],
          args: new_args,
        }
      }
      lb::Term::U32 { numb } => rt::Term::U32 { numb: *numb },
      lb::Term::Op2 { oper, val0, val1 } => {
        let oper = convert_oper(oper);
        let val0 = Box::new(convert_term(val0, comp, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, comp, depth + 1, vars));
        rt::Term::Op2 { oper, val0, val1 }
      }
    }
  }

  convert_term(term, comp, 0, &mut HashMap::new())
}

//pub func_rules: HashMap<String, Vec<lb::Rule>>,
//pub id_to_name: HashMap<u64, String>,
//pub name_to_id: HashMap<String, u64>,
//pub ctr_is_cal: HashMap<String, bool>,
pub fn make_rewriter(comp: &cm::Compilable, name: String) -> Option<rt::Function> {
  // Makes the test vector, which is used to determine if certain clause matched
  fn make_cond(comp: &cm::Compilable, lhs: &lb::Term) -> Vec<rt::Lnk> {
    if let lb::Term::Ctr { name, args } = lhs {
      let mut lnks: Vec<rt::Lnk> = Vec::new();
      for arg in args {
        lnks.push(match &**arg {
          lb::Term::Ctr { name, args } => {
            let ari = args.len() as u64;
            let fun = comp.name_to_id.get(&*name).unwrap_or(&0);
            let pos = 0;
            rt::Ctr(ari, *fun, pos)
          }
          lb::Term::U32 { numb } => rt::U_32(*numb as u64),
          _ => 0,
        })
      }
      return lnks;
    }
    panic!("Left-hand side not a function.");
  }

  // Makes the vars vector, which is used to access lhs variables
  fn make_vars(comp: &cm::Compilable, lhs: &lb::Term) -> Vec<(u64, Option<u64>)> {
    if let lb::Term::Ctr { name, args } = lhs {
      let mut vars: Vec<(u64, Option<u64>)> = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        match &**arg {
          lb::Term::Ctr { name, args } => {
            for j in 0..args.len() {
              vars.push((i as u64, Some(j as u64)));
            }
          }
          lb::Term::Var { .. } => {
            vars.push((i as u64, None));
          }
          _ => {}
        }
      }
      return vars;
    }
    panic!("Left-hand side not a function.");
  }

  if let Some(rules) = comp.func_rules.get(&name) {
    // Builds the "stricts" vector
    let mut stricts = Vec::new();
    for (i, rule) in rules.iter().enumerate() {
      while stricts.len() < i {
        stricts.push(false);
      }
      match *rule.lhs {
        lb::Term::Ctr { .. } => {
          stricts[i] = true;
        }
        lb::Term::U32 { .. } => {
          stricts[i] = true;
        }
        _ => {}
      }
    }

    // Builds the "rewrite" function
    let mut conds = Vec::new();
    let mut varss = Vec::new();
    for rule in rules {
      conds.push(make_cond(comp, &rule.lhs));
      varss.push(make_vars(comp, &rule.lhs));
    }
  }

  return None;
}

/// Reads back a Lambolt term from Runtime's memory
pub fn runtime_to_lambolt(mem: &Worker, comp: &cm::Compilable, host: u64) -> String {
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
      rt::CTR | rt::FUN => {
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
    comp: &'a cm::Compilable,
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
          rt::OR => "|",
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
      rt::CTR | rt::FUN => {
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
