//! Moves HVM Terms from runtime, and building dynamic functions.

// FIXME: `as_code` and `as_term` should just call `readback`, but before doing so, we must test
// the new readback properly to ensure it is correct

use crate::language as lang;
use crate::runtime as rt;
use crate::runtime::{Ptr, Worker};
use std::collections::{hash_map, HashMap, HashSet};

/// Reads back a term from Runtime's memory
pub fn as_code(mem: &Worker, i2n: Option<&HashMap<u64, String>>, host: u64) -> String {
  return format!("{}", as_term(mem, i2n, host));
}

/// Reads back a term from Runtime's memory
pub fn as_term(mem: &Worker, i2n: Option<&HashMap<u64, String>>, host: u64) -> Box<lang::Term> {
  struct CtxName<'a> {
    mem: &'a Worker,
    names: &'a mut HashMap<Ptr, String>,
    seen: &'a mut HashSet<Ptr>,
  }

  fn gen_var_names(mem: &Worker, ctx: &mut CtxName, term: Ptr, depth: u32) {
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
          ctx.names.insert(var, format!("x{}", ctx.names.len()));
        };
        gen_var_names(mem, ctx, body, depth + 1);
      }
      rt::APP => {
        let lam = rt::ask_arg(ctx.mem, term, 0);
        let arg = rt::ask_arg(ctx.mem, term, 1);
        gen_var_names(mem, ctx, lam, depth + 1);
        gen_var_names(mem, ctx, arg, depth + 1);
      }
      rt::SUP => {
        let arg0 = rt::ask_arg(ctx.mem, term, 0);
        let arg1 = rt::ask_arg(ctx.mem, term, 1);
        gen_var_names(mem, ctx, arg0, depth + 1);
        gen_var_names(mem, ctx, arg1, depth + 1);
      }
      rt::DP0 => {
        let arg = rt::ask_arg(ctx.mem, term, 2);
        gen_var_names(mem, ctx, arg, depth + 1);
      }
      rt::DP1 => {
        let arg = rt::ask_arg(ctx.mem, term, 2);
        gen_var_names(mem, ctx, arg, depth + 1);
      }
      rt::OP2 => {
        let arg0 = rt::ask_arg(ctx.mem, term, 0);
        let arg1 = rt::ask_arg(ctx.mem, term, 1);
        gen_var_names(mem, ctx, arg0, depth + 1);
        gen_var_names(mem, ctx, arg1, depth + 1);
      }
      rt::NUM => {}
      rt::CTR | rt::FUN => {
        let arity = rt::ask_ari(mem, term);
        for i in 0..arity {
          let arg = rt::ask_arg(ctx.mem, term, i);
          gen_var_names(mem, ctx, arg, depth + 1);
        }
      }
      _ => {}
    }
  }

  #[allow(dead_code)]
  struct CtxGo<'a> {
    mem: &'a Worker,
    i2n: Option<&'a HashMap<u64, String>>,
    names: &'a HashMap<Ptr, String>,
    seen: &'a HashSet<Ptr>,
  }

  struct Stacks {
    stacks: HashMap<Ptr, Vec<bool>>,
  }

  impl Stacks {
    fn new() -> Stacks {
      Stacks { stacks: HashMap::new() }
    }
    fn get(&self, col: Ptr) -> Option<&Vec<bool>> {
      self.stacks.get(&col)
    }
    fn pop(&mut self, col: Ptr) -> bool {
      let stack = self.stacks.entry(col).or_insert_with(Vec::new);
      stack.pop().unwrap_or(false)
    }
    fn push(&mut self, col: Ptr, val: bool) {
      let stack = self.stacks.entry(col).or_insert_with(Vec::new);
      stack.push(val);
    }
  }

  fn readback(mem: &Worker, ctx: &mut CtxGo, stacks: &mut Stacks, term: Ptr, depth: u32) -> Box<lang::Term> {
    match rt::get_tag(term) {
      rt::LAM => {
        let body = rt::ask_arg(ctx.mem, term, 1);
        let body = readback(mem, ctx, stacks, body, depth + 1);
        let bind = rt::ask_arg(ctx.mem, term, 0);
        let name = if rt::get_tag(bind) == rt::ERA {
          "*".to_string()
        } else {
          let var = rt::Var(rt::get_loc(term, 0));
          ctx.names.get(&var).map(|s| s.clone()).unwrap_or("?".to_string())
        };
        return Box::new(lang::Term::Lam { name, body });
      }
      rt::APP => {
        let func = rt::ask_arg(ctx.mem, term, 0);
        let argm = rt::ask_arg(ctx.mem, term, 1);
        let func = readback(mem, ctx, stacks, func, depth + 1);
        let argm = readback(mem, ctx, stacks, argm, depth + 1);
        return Box::new(lang::Term::App { func, argm });
      }
      rt::SUP => {
        let col = rt::get_ext(term);
        let empty = &Vec::new();
        let stack = stacks.get(col).unwrap_or(empty);
        if let Some(val) = stack.last() {
          let arg_idx = *val as u64;
          let val = rt::ask_arg(ctx.mem, term, arg_idx);
          let old = stacks.pop(col);
          let got = readback(mem, ctx, stacks, val, depth + 1);
          stacks.push(col, old);
          got
        } else {
          let name = "HVM.sup".to_string(); // lang::Term doesn't have a Sup variant
          let val0 = rt::ask_arg(ctx.mem, term, 0);
          let val1 = rt::ask_arg(ctx.mem, term, 1);
          let val0 = readback(mem, ctx, stacks, val0, depth + 1);
          let val1 = readback(mem, ctx, stacks, val1, depth + 1);
          let args = vec![val0, val1];
          return Box::new(lang::Term::Ctr { name, args });
        }
      }
      rt::DP0 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        stacks.push(col, false);
        let result = readback(mem, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      rt::DP1 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        stacks.push(col, true);
        let result = readback(mem, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      rt::OP2 => {
        let oper = match rt::get_ext(term) {
          rt::ADD => lang::Oper::Add,
          rt::SUB => lang::Oper::Sub,
          rt::MUL => lang::Oper::Mul,
          rt::DIV => lang::Oper::Div,
          rt::MOD => lang::Oper::Mod,
          rt::AND => lang::Oper::And,
          rt::OR  => lang::Oper::Or,
          rt::XOR => lang::Oper::Xor,
          rt::SHL => lang::Oper::Shl,
          rt::SHR => lang::Oper::Shr,
          rt::LTN => lang::Oper::Ltn,
          rt::LTE => lang::Oper::Lte,
          rt::EQL => lang::Oper::Eql,
          rt::GTE => lang::Oper::Gte,
          rt::GTN => lang::Oper::Gtn,
          rt::NEQ => lang::Oper::Neq,
          _       => panic!("unknown operation"),
        };
        let val0 = rt::ask_arg(ctx.mem, term, 0);
        let val1 = rt::ask_arg(ctx.mem, term, 1);
        let val0 = readback(mem, ctx, stacks, val0, depth + 1);
        let val1 = readback(mem, ctx, stacks, val1, depth + 1);
        return Box::new(lang::Term::Op2 { oper, val0, val1 });
      }
      rt::NUM => {
        let numb = rt::get_num(term);
        return Box::new(lang::Term::Num { numb });
      }
      rt::CTR | rt::FUN => {
        let func = rt::get_ext(term);
        let arit = rt::ask_ari(mem, term);
        let mut args = Vec::new();
        for i in 0 .. arit {
          let arg = rt::ask_arg(ctx.mem, term, i);
          args.push(readback(mem, ctx, stacks, arg, depth + 1));
        }
        let name = match ctx.i2n {
          None => format!("${}", func),
          Some(i2n) => i2n.get(&func).map(String::to_string).unwrap_or_else(|| format!("${}", func))
        };
        return Box::new(lang::Term::Ctr { name, args });
      }
      rt::VAR => {
        let name = ctx.names.get(&term).map(String::to_string).unwrap_or_else(|| format!("^{}", rt::get_loc(term, 0)));
        return Box::new(lang::Term::Var { name }); // ............... /\ why this sounds so threatening?
      }
      rt::ARG => {
        return Box::new(lang::Term::Var { name: "<arg>".to_string() });
      }
      rt::ERA => {
        return Box::new(lang::Term::Var { name: "<era>".to_string() });
      }
      _ => {
        return Box::new(lang::Term::Var { name: format!("<unknown_tag_{}>", rt::get_tag(term)) });
      }
    }
  }

  let term = rt::ask_lnk(mem, host);

  let mut names = HashMap::<Ptr, String>::new();
  let mut seen = HashSet::<Ptr>::new();

  let ctx = &mut CtxName { mem, names: &mut names, seen: &mut seen };
  gen_var_names(mem, ctx, term, 0);

  let ctx = &mut CtxGo { mem, i2n, names: &names, seen: &seen };
  let mut stacks = Stacks::new();
  readback(mem, ctx, &mut stacks, term, 0)
}

// Reads a term linearly, i.e., preserving dups
pub fn as_linear_term(rt: &Worker, i2n: Option<&HashMap<u64, String>>, host: u64) -> Box<lang::Term> {
  enum StackItem {
    Term(Ptr),
    Resolver(Ptr),
  }

  fn ctr_name(i2n: Option<&HashMap<u64, String>>, id: u64) -> String {
    if let Some(i2n) = i2n {
      if let Some(name) = i2n.get(&id) {
        return name.clone();
      }
    }
    return format!("${}", id);
  }

  fn dups(rt: &Worker, i2n: Option<&HashMap<u64, String>>, term: Ptr, names: &mut HashMap<u64, String>) -> lang::Term {
    let mut lets: HashMap<u64, u64> = HashMap::new();
    let mut kinds: HashMap<u64, u64> = HashMap::new();
    let mut stack = vec![term];
    while !stack.is_empty() {
      let term = stack.pop().unwrap();
      match rt::get_tag(term) {
        rt::LAM => {
          names.insert(rt::get_loc(term, 0), format!("{}", names.len()));
          stack.push(rt::ask_arg(rt, term, 1));
        }
        rt::APP => {
          stack.push(rt::ask_arg(rt, term, 1));
          stack.push(rt::ask_arg(rt, term, 0));
        }
        rt::SUP => {
          stack.push(rt::ask_arg(rt, term, 1));
          stack.push(rt::ask_arg(rt, term, 0));
        }
        rt::DP0 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(rt::get_loc(term, 0)) {
            names.insert(rt::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(rt::get_loc(term, 0), rt::get_ext(term));
            e.insert(rt::get_loc(term, 0));
            stack.push(rt::ask_arg(rt, term, 2));
          }
        }
        rt::DP1 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(rt::get_loc(term, 0)) {
            names.insert(rt::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(rt::get_loc(term, 0), rt::get_ext(term));
            e.insert(rt::get_loc(term, 0));
            stack.push(rt::ask_arg(rt, term, 2));
          }
        }
        rt::OP2 => {
          stack.push(rt::ask_arg(rt, term, 1));
          stack.push(rt::ask_arg(rt, term, 0));
        }
        rt::CTR | rt::FUN => {
          let arity = rt::ask_ari(rt, term);
          for i in (0..arity).rev() {
            stack.push(rt::ask_arg(rt, term, i));
          }
        }
        _ => {}
      }
    }

    let cont = expr(rt, i2n, term, &names);
    if lets.is_empty() {
      cont
    } else {
      let mut output = lang::Term::Var { name: "?".to_string() };
      for (i, (_key, pos)) in lets.iter().enumerate() {
        // todo: reverse
        let what = String::from("?h");
        let name = names.get(&pos).unwrap_or(&what);
        let nam0 = if rt::ask_lnk(rt, pos + 0) == rt::Era() { String::from("*") } else { format!("a{}", name) };
        let nam1 = if rt::ask_lnk(rt, pos + 1) == rt::Era() { String::from("*") } else { format!("b{}", name) };
        let expr = expr(rt, i2n, rt::ask_lnk(rt, pos + 2), &names);
        if i == 0 {
          output = lang::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(cont.clone()) };
        } else {
          output = lang::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(output) };
        }
      }
      output
    }
  }

  fn expr(rt: &Worker, i2n: Option<&HashMap<u64, String>>, term: Ptr, names: &HashMap<u64, String>) -> lang::Term {
    let mut stack = vec![StackItem::Term(term)];
    let mut output : Vec<lang::Term> = Vec::new();
    while !stack.is_empty() {
      let item = stack.pop().unwrap();
      match item {
        StackItem::Resolver(term) => {
          match rt::get_tag(term) {
            rt::CTR => {
              let func = rt::get_ext(term);
              let arit = rt::ask_ari(rt, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(i2n, func);
              output.push(lang::Term::Ctr { name, args });
            },
            rt::FUN => {
              let func = rt::get_ext(term);
              let arit = rt::ask_ari(rt, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(i2n, func);
              output.push(lang::Term::Ctr { name, args });
            }
            rt::LAM => {
              let name = format!("x{}", names.get(&rt::get_loc(term, 0)).unwrap_or(&String::from("?")));
              let body = Box::new(output.pop().unwrap());
              output.push(lang::Term::Lam { name, body });
            }
            rt::APP => {
              let argm = Box::new(output.pop().unwrap());
              let func = Box::new(output.pop().unwrap());
              output.push(lang::Term::App { func , argm });
            }
            rt::OP2 => {
              let oper = rt::get_ext(term);
              let oper = match oper {
                rt::ADD => lang::Oper::Add,
                rt::SUB => lang::Oper::Sub,
                rt::MUL => lang::Oper::Mul,
                rt::DIV => lang::Oper::Div,
                rt::MOD => lang::Oper::Mod,
                rt::AND => lang::Oper::And,
                rt::OR  => lang::Oper::Or,
                rt::XOR => lang::Oper::Xor,
                rt::SHL => lang::Oper::Shl,
                rt::SHR => lang::Oper::Shr,
                rt::LTN => lang::Oper::Ltn,
                rt::LTE => lang::Oper::Lte,
                rt::EQL => lang::Oper::Eql,
                rt::GTE => lang::Oper::Gte,
                rt::GTN => lang::Oper::Gtn,
                rt::NEQ => lang::Oper::Neq,
                _       => panic!("Invalid operator."),
              };
              let val1 = Box::new(output.pop().unwrap());
              let val0 = Box::new(output.pop().unwrap());
              output.push(lang::Term::Op2 { oper, val0, val1 })
            }
            _ => panic!("Term not valid in readback"),
          }
        },
        StackItem::Term(term) => {
          match rt::get_tag(term) {
            rt::DP0 => {
              let name = format!("a{}", names.get(&rt::get_loc(term, 0)).unwrap_or(&String::from("?a")));
              output.push(lang::Term::Var { name });
            }
            rt::DP1 => {
              let name = format!("b{}", names.get(&rt::get_loc(term, 0)).unwrap_or(&String::from("?b")));
              output.push(lang::Term::Var { name });
            }
            rt::VAR => {
              let name = format!("x{}", names.get(&rt::get_loc(term, 0)).unwrap_or(&String::from("?x")));
              output.push(lang::Term::Var { name });
            }
            rt::LAM => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(rt::ask_arg(rt, term, 1)));
            }
            rt::APP => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(rt::ask_arg(rt, term, 1)));
              stack.push(StackItem::Term(rt::ask_arg(rt, term, 0)));
            }
            rt::SUP => {}
            rt::OP2 => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(rt::ask_arg(rt, term, 1)));
              stack.push(StackItem::Term(rt::ask_arg(rt, term, 0)));
            }
            rt::NUM => {
              let numb = rt::get_num(term);
              output.push(lang::Term::Num { numb });
            }
            rt::CTR => {
              let arit = rt::ask_ari(rt, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(rt::ask_arg(rt, term, i)));
              }
            }
            rt::FUN => {
              let arit = rt::ask_ari(rt, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(rt::ask_arg(rt, term, i)));
              }
            }
            rt::ERA => {}
            _ => {}
          }
        }
      }
    }
    output.pop().unwrap()
  }

  let mut names: HashMap<u64, String> = HashMap::new();
  Box::new(dups(rt, i2n, rt::ask_lnk(rt, host), &mut names))
}

/// Reads back a term from Runtime's memory
pub fn as_linear_code(mem: &Worker, i2n: Option<&HashMap<u64, String>>, host: u64) -> String {
  return format!("{}", as_linear_term(mem, i2n, host));
}
