// Moving Lambolt Terms to/from runtime, and building dynamic functions.

#![allow(clippy::identity_op)]

use crate::compilable as cm;
use crate::lambolt as lb;
use crate::runtime as rt;
use crate::runtime::{Lnk, Worker};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub fn build_dynamic_functions(comp: &cm::Compilable) -> HashMap<u64, rt::Function> {
  let mut fns: HashMap<u64, rt::Function> = HashMap::new();
  for (name, rules) in &comp.func_rules {
    let id = comp.name_to_id.get(name).unwrap_or(&0);
    let ff = build_dynamic_function(comp, rules);
    fns.insert(*id, ff);
  }
  return fns;
}

// Given a Compilable file and a function name, builds a dynamic Rust closure that applies that
// function to the runtime memory buffer directly. This process is complex, so I'll write a lot of
// comments here. All comments will be based on the following Lambolt example:
//   (Add (Succ a) b) = (Succ (Add a b))
//   (Add (Zero)   b) = b
pub fn build_dynamic_function(comp: &cm::Compilable, rules: &Vec<lb::Rule>) -> rt::Function {
  type VarInfo = (u64, Option<u64>, bool); // Argument index, field index, is it used?

  // This is an aux function that makes the stricts vector. It specifies which arguments need
  // reduction. For example, on `(Add (Succ a) b) = ...`, only the first argument must be
  // reduced. The stricts vector will be: `[true, false]`.
  fn make_stricts(comp: &cm::Compilable, rules: &Vec<lb::Rule>) -> Vec<bool> {
    let mut stricts = Vec::new();
    for rule in rules {
      if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
        while stricts.len() < args.len() {
          stricts.push(false);
        }
        for (i, arg) in args.iter().enumerate() {
          match **arg {
            lb::Term::Ctr { .. } => {
              stricts[i] = true;
            }
            lb::Term::U32 { .. } => {
              stricts[i] = true;
            }
            _ => {}
          }
        }
      } else {
        panic!("Invalid left-hand side: {}", rule.lhs);
      }
    }
    return stricts;
  }

  // This is an aux function that makes the vectors used to determine if certain rule matched.
  // That vector will contain Lnks with the proper constructor tag, for each strict argument, and
  // 0, for each variable argument. For example, on `(Add (Succ a) b) = ...`, we only need to
  // match one constructor, `Succ`. The resulting vector will be: `[rt::Ctr(SUCC,1,0), 0]`.
  fn make_conds(comp: &cm::Compilable, rules: &Vec<lb::Rule>) -> Vec<Vec<rt::Lnk>> {
    let mut conds = Vec::new();
    for rule in rules {
      if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
        let mut lnks: Vec<rt::Lnk> = Vec::new();
        for arg in args {
          lnks.push(match **arg {
            lb::Term::Ctr { ref name, ref args } => {
              let ari = args.len() as u64;
              let fun = comp.name_to_id.get(&*name).unwrap_or(&0);
              let pos = 0;
              rt::Ctr(ari, *fun, pos)
            }
            lb::Term::U32 { ref numb } => rt::U_32(*numb as u64),
            _ => 0,
          })
        }
        conds.push(lnks);
      } else {
        panic!("Invalid left-hand side: {}", rule.lhs);
      }
    }
    return conds;
  }

  // This is an aux function that makes the vars vectors, which is used to locate left-hand side
  // variables. For example, on `(Add (Succ a) b) = ...`, we have two variables, one on the first
  // field of the first argument, and one is the second argument. Both variables are used. The vars
  // vector for it is: `[(0,Some(0),true), (1,None,true)]`
  fn make_varss(comp: &cm::Compilable, rules: &Vec<lb::Rule>) -> Vec<Vec<VarInfo>> {
    let mut varss = Vec::new();
    for rule in rules {
      if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
        let mut vars: Vec<VarInfo> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
          match &**arg {
            lb::Term::Ctr { name, args } => {
              for j in 0..args.len() {
                match *args[j] {
                  lb::Term::Var { ref name } => {
                    vars.push((i as u64, Some(j as u64), name != "*"));
                  }
                  _ => {
                    panic!(
                      "Argument {}, constructor {}, field {}, is not a variable.",
                      i, name, j
                    );
                  }
                }
              }
            }
            lb::Term::Var { name } => {
              vars.push((i as u64, None, name != "*"));
            }
            _ => {}
          }
        }
        varss.push(vars);
      } else {
        panic!("Invalid left-hand side: {}", rule.lhs);
      }
    }
    return varss;
  }

  // This is an aux function that makes the clears vector. It specifies which arguments need to
  // be freed after reduction. For example, on `(Add (Succ a) b) = ...`, only the first argument
  // is a constructor that can be freed. The clears vector will be: `[(0,1)]`. The first value is
  // the argument index, the second value is the ctor arity.
  fn make_clears(comp: &cm::Compilable, rules: &Vec<lb::Rule>) -> Vec<(u64, u64)> {
    let mut clears = Vec::new();
    for rule in rules {
      if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
        for (i, arg) in args.iter().enumerate() {
          match **arg {
            lb::Term::Ctr { ref args, .. } => {
              clears.push((i as u64, args.len() as u64));
            }
            _ => {}
          }
        }
      } else {
        panic!("Invalid left-hand side: {}", rule.lhs);
      }
    }
    return clears;
  }

  // Makes the bodies vector.
  fn make_bodies(comp: &cm::Compilable, rules: &Vec<lb::Rule>, varss: &Vec<Vec<VarInfo>>) -> Vec<rt::Term> {
    let mut bodies = Vec::new();
    for i in 0 .. rules.len() {
      bodies.push(to_runtime_term(comp, &rules[i].rhs, varss[i].len() as u64));
    }
    return bodies;
  }

  // Builds the static objects
  let stricts = make_stricts(comp, &rules);
  let conds = make_conds(comp, &rules);
  let varss = make_varss(comp, &rules);
  let bodies = make_bodies(comp, &rules, &varss);
  let clears = make_clears(comp, &rules);
  let count = rules.len() as u64;
  let arity = stricts.len() as u64;

  // Builds the returned stricts vector.
  let stricts_ret = stricts.clone();

  // Builds the returned rewriter function.
  let rewriter: rt::Rewriter = Box::new(move |mem, host, term| {
    //println!("> rewriter");

    // Gets the left-hand side arguments (ex: `(Succ a)` and `b`)
    let mut args = Vec::new();
    for i in 0..arity {
      args.push(rt::ask_arg(mem, term, i));
    }

    // For each argument, if it is strict and a PAR, apply the cal_par rule
    for i in 0..arity {
      if stricts[i as usize] && rt::get_tag(args[i as usize]) == rt::PAR {
        rt::cal_par(mem, host, term, args[i as usize], i);
        return true;
      }
    }

    // For each rule condition vector
    for rule_index in 0..count {
      let rule_cond = &conds[rule_index as usize];
      let rule_vars = &varss[rule_index as usize];
      let rule_body = &bodies[rule_index as usize];

      // Check if the rule matches
      let mut matched = true;

      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      //println!(">> testing conditions... total: {} conds", rule_cond.len());
      for (i, cond) in rule_cond.iter().enumerate() {
        match rt::get_tag(*cond) {
          rt::U32 => {
            //println!(">>> cond demands U32 {} at {}", rt::get_val(*cond), i);
            let same_tag = rt::get_tag(args[i]) == rt::U32;
            let same_val = rt::get_val(args[i]) == rt::get_val(*cond);
            matched = matched && same_tag && same_val;
          }
          rt::CTR => {
            //println!(">>> cond demands CTR {} at {}", rt::get_ext(*cond), i);
            //println!(">>> got: {} {}", rt::get_tag(args[i]), rt::get_ext(args[i]));
            let same_tag = rt::get_tag(args[i]) == rt::CTR;
            let same_ext = rt::get_ext(args[i]) == rt::get_ext(*cond);
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
        for (i, may_j, used) in rule_vars {
          match *may_j {
            Some(j) => vars.push(rt::ask_arg(mem, args[*i as usize], j)),
            None => vars.push(args[*i as usize]),
          }
        }

        // FIXME: `dups` must be global to properly color the fan nodes, but Rust complains about
        // mutating borrowed variables. Until this is fixed, the language will be very restrict.
        let mut dups = 0;

        // Builds the right-hand side term (ex: `(Succ (Add a b))`)
        let done = rt::make_term(mem, &bodies[rule_index as usize], &mut vars, &mut dups);

        // Links the host location to it
        rt::link(mem, host, done);

        // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
        rt::clear(mem, rt::get_loc(term, 0), arity);
        for (i, arity) in &clears {
          rt::clear(mem, rt::get_loc(args[*i as usize], 0), *arity);
        }

        // Collects unused variables (none in this example)
        for (i, (_, _, used)) in rule_vars.iter().enumerate() {
          if !used {
            rt::collect(mem, vars[i]);
          }
        }

        return true;
      }
    }
    return false;
  });

  return rt::Function {
    stricts: stricts_ret,
    rewriter,
  };
}

/// Converts a Lambolt Term to a Runtime Term
pub fn to_runtime_term(comp: &cm::Compilable, term: &lb::Term, free_vars: u64) -> rt::Term {
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
    comp: &cm::Compilable,
    depth: u64,
    vars: &mut Vec<String>,
  ) -> rt::Term {
    match term {
      lb::Term::Var { name } => {
        for i in 0 .. vars.len() {
          let j = vars.len() - i - 1;
          if vars[j] == *name {
            return rt::Term::Var { bidx: j as u64 }
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
        rt::Term::Dup { expr, body }
      }
      lb::Term::Lam { name, body } => {
        vars.push(name.clone());
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.pop();
        rt::Term::Lam { body }
      }
      lb::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, comp, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(convert_term(body, comp, depth + 1, vars));
        vars.pop();
        rt::Term::Let { expr, body }
      }
      lb::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, comp, depth + 0, vars));
        let argm = Box::new(convert_term(argm, comp, depth + 0, vars));
        rt::Term::App { func, argm }
      }
      lb::Term::Ctr { name, args } => {
        let term_func = comp.name_to_id[name];
        let mut term_args: Vec<rt::Term> = Vec::new();
        for arg in args {
          term_args.push(convert_term(arg, comp, depth + 0, vars));
        }
        if *comp.ctr_is_cal.get(name).unwrap_or(&false) {
          rt::Term::Cal {
            func: term_func,
            args: term_args,
          }
        } else {
          rt::Term::Ctr {
            func: term_func,
            args: term_args,
          }
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

  let mut vars = Vec::new();
  for i in 0 .. free_vars {
    vars.push(format!("x{}", i).to_string());
  }
  convert_term(term, comp, 0, &mut vars)
}

/// Reads back a Lambolt term from Runtime's memory
// TODO: we should readback as a lambolt::Term, not as a string
pub fn readback_as_code(mem: &Worker, comp: &cm::Compilable, host: u64) -> String {
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
