//! Moves HVM Terms from runtime, and building dynamic functions.

// FIXME: `as_code` and `as_term` should just call `readback`, but before doing so, we must test
// the new readback properly to ensure it is correct

use crate::language as lang;
use crate::runtime as runtime;
use crate::runtime::{Ptr, Heap, Stats, Info};
use std::collections::{hash_map, HashMap, HashSet};

/// Reads back a term from Runtime's memory
pub fn as_code(heap: &Heap, stats: &Stats, info: &Info, host: u64) -> String {
  return format!("{}", as_term(heap, stats, info, host));
}

/// Reads back a term from Runtime's memory
pub fn as_term(heap: &Heap, stats: &Stats, info: &Info, host: u64) -> Box<lang::Term> {
  struct CtxName<'a> {
    heap: &'a Heap,
    stats: &'a Stats,
    info: &'a Info,
    names: &'a mut HashMap<Ptr, String>,
    seen: &'a mut HashSet<Ptr>,
  }

  fn gen_var_names(heap: &Heap, stats: &Stats, info: &Info, ctx: &mut CtxName, term: Ptr, depth: u32) {
    if ctx.seen.contains(&term) {
      return;
    };

    ctx.seen.insert(term);

    match runtime::get_tag(term) {
      runtime::LAM => {
        let param = runtime::ask_arg(&ctx.heap, term, 0);
        let body = runtime::ask_arg(&ctx.heap, term, 1);
        if runtime::get_tag(param) != runtime::ERA {
          let var = runtime::Var(runtime::get_loc(term, 0));
          ctx.names.insert(var, format!("x{}", ctx.names.len()));
        };
        gen_var_names(heap, stats, info, ctx, body, depth + 1);
      }
      runtime::APP => {
        let lam = runtime::ask_arg(&ctx.heap, term, 0);
        let arg = runtime::ask_arg(&ctx.heap, term, 1);
        gen_var_names(heap, stats, info, ctx, lam, depth + 1);
        gen_var_names(heap, stats, info, ctx, arg, depth + 1);
      }
      runtime::SUP => {
        let arg0 = runtime::ask_arg(&ctx.heap, term, 0);
        let arg1 = runtime::ask_arg(&ctx.heap, term, 1);
        gen_var_names(heap, stats, info, ctx, arg0, depth + 1);
        gen_var_names(heap, stats, info, ctx, arg1, depth + 1);
      }
      runtime::DP0 => {
        let arg = runtime::ask_arg(&ctx.heap, term, 2);
        gen_var_names(heap, stats, info, ctx, arg, depth + 1);
      }
      runtime::DP1 => {
        let arg = runtime::ask_arg(&ctx.heap, term, 2);
        gen_var_names(heap, stats, info, ctx, arg, depth + 1);
      }
      runtime::OP2 => {
        let arg0 = runtime::ask_arg(&ctx.heap, term, 0);
        let arg1 = runtime::ask_arg(&ctx.heap, term, 1);
        gen_var_names(heap, stats, info, ctx, arg0, depth + 1);
        gen_var_names(heap, stats, info, ctx, arg1, depth + 1);
      }
      runtime::NUM => {}
      runtime::CTR | runtime::FUN => {
        let arity = runtime::ask_ari(&ctx.info, term);
        for i in 0..arity {
          let arg = runtime::ask_arg(&ctx.heap, term, i);
          gen_var_names(heap, stats, info, ctx, arg, depth + 1);
        }
      }
      _ => {}
    }
  }

  #[allow(dead_code)]
  struct CtxGo<'a> {
    heap: &'a Heap,
    stats: &'a Stats,
    info: &'a Info,
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

  fn readback(heap: &Heap, stats: &Stats, info: &Info, ctx: &mut CtxGo, stacks: &mut Stacks, term: Ptr, depth: u32) -> Box<lang::Term> {
    match runtime::get_tag(term) {
      runtime::LAM => {
        let body = runtime::ask_arg(&ctx.heap, term, 1);
        let body = readback(heap, stats, info, ctx, stacks, body, depth + 1);
        let bind = runtime::ask_arg(&ctx.heap, term, 0);
        let name = if runtime::get_tag(bind) == runtime::ERA {
          "*".to_string()
        } else {
          let var = runtime::Var(runtime::get_loc(term, 0));
          ctx.names.get(&var).map(|s| s.clone()).unwrap_or("?".to_string())
        };
        return Box::new(lang::Term::Lam { name, body });
      }
      runtime::APP => {
        let func = runtime::ask_arg(&ctx.heap, term, 0);
        let argm = runtime::ask_arg(&ctx.heap, term, 1);
        let func = readback(heap, stats, info, ctx, stacks, func, depth + 1);
        let argm = readback(heap, stats, info, ctx, stacks, argm, depth + 1);
        return Box::new(lang::Term::App { func, argm });
      }
      runtime::SUP => {
        let col = runtime::get_ext(term);
        let empty = &Vec::new();
        let stack = stacks.get(col).unwrap_or(empty);
        if let Some(val) = stack.last() {
          let arg_idx = *val as u64;
          let val = runtime::ask_arg(&ctx.heap, term, arg_idx);
          let old = stacks.pop(col);
          let got = readback(heap, stats, info, ctx, stacks, val, depth + 1);
          stacks.push(col, old);
          got
        } else {
          let name = "HVM.sup".to_string(); // lang::Term doesn't have a Sup variant
          let val0 = runtime::ask_arg(&ctx.heap, term, 0);
          let val1 = runtime::ask_arg(&ctx.heap, term, 1);
          let val0 = readback(heap, stats, info, ctx, stacks, val0, depth + 1);
          let val1 = readback(heap, stats, info, ctx, stacks, val1, depth + 1);
          let args = vec![val0, val1];
          return Box::new(lang::Term::Ctr { name, args });
        }
      }
      runtime::DP0 => {
        let col = runtime::get_ext(term);
        let val = runtime::ask_arg(&ctx.heap, term, 2);
        stacks.push(col, false);
        let result = readback(heap, stats, info, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      runtime::DP1 => {
        let col = runtime::get_ext(term);
        let val = runtime::ask_arg(&ctx.heap, term, 2);
        stacks.push(col, true);
        let result = readback(heap, stats, info, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      runtime::OP2 => {
        let oper = match runtime::get_ext(term) {
          runtime::ADD => lang::Oper::Add,
          runtime::SUB => lang::Oper::Sub,
          runtime::MUL => lang::Oper::Mul,
          runtime::DIV => lang::Oper::Div,
          runtime::MOD => lang::Oper::Mod,
          runtime::AND => lang::Oper::And,
          runtime::OR  => lang::Oper::Or,
          runtime::XOR => lang::Oper::Xor,
          runtime::SHL => lang::Oper::Shl,
          runtime::SHR => lang::Oper::Shr,
          runtime::LTN => lang::Oper::Ltn,
          runtime::LTE => lang::Oper::Lte,
          runtime::EQL => lang::Oper::Eql,
          runtime::GTE => lang::Oper::Gte,
          runtime::GTN => lang::Oper::Gtn,
          runtime::NEQ => lang::Oper::Neq,
          _       => panic!("unknown operation"),
        };
        let val0 = runtime::ask_arg(&ctx.heap, term, 0);
        let val1 = runtime::ask_arg(&ctx.heap, term, 1);
        let val0 = readback(heap, stats, info, ctx, stacks, val0, depth + 1);
        let val1 = readback(heap, stats, info, ctx, stacks, val1, depth + 1);
        return Box::new(lang::Term::Op2 { oper, val0, val1 });
      }
      runtime::NUM => {
        let numb = runtime::get_num(term);
        return Box::new(lang::Term::Num { numb });
      }
      runtime::CTR | runtime::FUN => {
        let func = runtime::get_ext(term);
        let arit = runtime::ask_ari(&ctx.info, term);
        let mut args = Vec::new();
        for i in 0 .. arit {
          let arg = runtime::ask_arg(&ctx.heap, term, i);
          args.push(readback(heap, stats, info, ctx, stacks, arg, depth + 1));
        }
        let name = ctx.info.nams.get(&func).map(String::to_string).unwrap_or_else(|| format!("${}", func));
        return Box::new(lang::Term::Ctr { name, args });
      }
      runtime::VAR => {
        let name = ctx.names.get(&term).map(String::to_string).unwrap_or_else(|| format!("^{}", runtime::get_loc(term, 0)));
        return Box::new(lang::Term::Var { name }); // ............... /\ why this sounds so threatening?
      }
      runtime::ARG => {
        return Box::new(lang::Term::Var { name: "<arg>".to_string() });
      }
      runtime::ERA => {
        return Box::new(lang::Term::Var { name: "<era>".to_string() });
      }
      _ => {
        return Box::new(lang::Term::Var { name: format!("<unknown_tag_{}>", runtime::get_tag(term)) });
      }
    }
  }

  let term = runtime::ask_lnk(heap, host);

  let mut names = HashMap::<Ptr, String>::new();
  let mut seen = HashSet::<Ptr>::new();

  let ctx = &mut CtxName { heap, stats, info, names: &mut names, seen: &mut seen };
  gen_var_names(heap, stats, info, ctx, term, 0);

  let ctx = &mut CtxGo { heap, stats, info, names: &names, seen: &seen };
  let mut stacks = Stacks::new();
  readback(heap, stats, info, ctx, &mut stacks, term, 0)
}

// Reads a term linearly, i.e., preserving dups
pub fn as_linear_term(heap: &Heap, stats: &Stats, info: &Info, host: u64) -> Box<lang::Term> {
  enum StackItem {
    Term(Ptr),
    Resolver(Ptr),
  }

  fn ctr_name(info: &Info, id: u64) -> String {
    if let Some(name) = info.nams.get(&id) {
      return name.clone();
    } else {
      return format!("${}", id);
    }
  }

  fn dups(heap: &Heap, stats: &Stats, info: &Info, term: Ptr, names: &mut HashMap<u64, String>) -> lang::Term {
    let mut lets: HashMap<u64, u64> = HashMap::new();
    let mut kinds: HashMap<u64, u64> = HashMap::new();
    let mut stack = vec![term];
    while !stack.is_empty() {
      let term = stack.pop().unwrap();
      match runtime::get_tag(term) {
        runtime::LAM => {
          names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
          stack.push(runtime::ask_arg(heap, term, 1));
        }
        runtime::APP => {
          stack.push(runtime::ask_arg(heap, term, 1));
          stack.push(runtime::ask_arg(heap, term, 0));
        }
        runtime::SUP => {
          stack.push(runtime::ask_arg(heap, term, 1));
          stack.push(runtime::ask_arg(heap, term, 0));
        }
        runtime::DP0 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(runtime::get_loc(term, 0)) {
            names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(runtime::get_loc(term, 0), runtime::get_ext(term));
            e.insert(runtime::get_loc(term, 0));
            stack.push(runtime::ask_arg(heap, term, 2));
          }
        }
        runtime::DP1 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(runtime::get_loc(term, 0)) {
            names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(runtime::get_loc(term, 0), runtime::get_ext(term));
            e.insert(runtime::get_loc(term, 0));
            stack.push(runtime::ask_arg(heap, term, 2));
          }
        }
        runtime::OP2 => {
          stack.push(runtime::ask_arg(heap, term, 1));
          stack.push(runtime::ask_arg(heap, term, 0));
        }
        runtime::CTR | runtime::FUN => {
          let arity = runtime::ask_ari(info, term);
          for i in (0..arity).rev() {
            stack.push(runtime::ask_arg(heap, term, i));
          }
        }
        _ => {}
      }
    }

    let cont = expr(heap, stats, info, term, &names);
    if lets.is_empty() {
      cont
    } else {
      let mut output = lang::Term::Var { name: "?".to_string() };
      for (i, (_key, pos)) in lets.iter().enumerate() {
        // todo: reverse
        let what = String::from("?h");
        let name = names.get(&pos).unwrap_or(&what);
        let nam0 = if runtime::ask_lnk(heap, pos + 0) == runtime::Era() { String::from("*") } else { format!("a{}", name) };
        let nam1 = if runtime::ask_lnk(heap, pos + 1) == runtime::Era() { String::from("*") } else { format!("b{}", name) };
        let expr = expr(heap, stats, info, runtime::ask_lnk(heap, pos + 2), &names);
        if i == 0 {
          output = lang::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(cont.clone()) };
        } else {
          output = lang::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(output) };
        }
      }
      output
    }
  }

  fn expr(heap: &Heap, stats: &Stats, info: &Info, term: Ptr, names: &HashMap<u64, String>) -> lang::Term {
    let mut stack = vec![StackItem::Term(term)];
    let mut output : Vec<lang::Term> = Vec::new();
    while !stack.is_empty() {
      let item = stack.pop().unwrap();
      match item {
        StackItem::Resolver(term) => {
          match runtime::get_tag(term) {
            runtime::CTR => {
              let func = runtime::get_ext(term);
              let arit = runtime::ask_ari(info, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(info, func);
              output.push(lang::Term::Ctr { name, args });
            },
            runtime::FUN => {
              let func = runtime::get_ext(term);
              let arit = runtime::ask_ari(info, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(info, func);
              output.push(lang::Term::Ctr { name, args });
            }
            runtime::LAM => {
              let name = format!("x{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?")));
              let body = Box::new(output.pop().unwrap());
              output.push(lang::Term::Lam { name, body });
            }
            runtime::APP => {
              let argm = Box::new(output.pop().unwrap());
              let func = Box::new(output.pop().unwrap());
              output.push(lang::Term::App { func , argm });
            }
            runtime::OP2 => {
              let oper = runtime::get_ext(term);
              let oper = match oper {
                runtime::ADD => lang::Oper::Add,
                runtime::SUB => lang::Oper::Sub,
                runtime::MUL => lang::Oper::Mul,
                runtime::DIV => lang::Oper::Div,
                runtime::MOD => lang::Oper::Mod,
                runtime::AND => lang::Oper::And,
                runtime::OR  => lang::Oper::Or,
                runtime::XOR => lang::Oper::Xor,
                runtime::SHL => lang::Oper::Shl,
                runtime::SHR => lang::Oper::Shr,
                runtime::LTN => lang::Oper::Ltn,
                runtime::LTE => lang::Oper::Lte,
                runtime::EQL => lang::Oper::Eql,
                runtime::GTE => lang::Oper::Gte,
                runtime::GTN => lang::Oper::Gtn,
                runtime::NEQ => lang::Oper::Neq,
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
          match runtime::get_tag(term) {
            runtime::DP0 => {
              let name = format!("a{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?a")));
              output.push(lang::Term::Var { name });
            }
            runtime::DP1 => {
              let name = format!("b{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?b")));
              output.push(lang::Term::Var { name });
            }
            runtime::VAR => {
              let name = format!("x{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?x")));
              output.push(lang::Term::Var { name });
            }
            runtime::LAM => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::ask_arg(heap, term, 1)));
            }
            runtime::APP => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::ask_arg(heap, term, 1)));
              stack.push(StackItem::Term(runtime::ask_arg(heap, term, 0)));
            }
            runtime::SUP => {}
            runtime::OP2 => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::ask_arg(heap, term, 1)));
              stack.push(StackItem::Term(runtime::ask_arg(heap, term, 0)));
            }
            runtime::NUM => {
              let numb = runtime::get_num(term);
              output.push(lang::Term::Num { numb });
            }
            runtime::CTR => {
              let arit = runtime::ask_ari(info, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(runtime::ask_arg(heap, term, i)));
              }
            }
            runtime::FUN => {
              let arit = runtime::ask_ari(info, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(runtime::ask_arg(heap, term, i)));
              }
            }
            runtime::ERA => {}
            _ => {}
          }
        }
      }
    }
    output.pop().unwrap()
  }

  let mut names: HashMap<u64, String> = HashMap::new();
  Box::new(dups(heap, stats, info, runtime::ask_lnk(heap, host), &mut names))
}

/// Reads back a term from Runtime's memory
pub fn as_linear_code(heap: &Heap, stats: &Stats, info: &Info, host: u64) -> String {
  return format!("{}", as_linear_term(heap, stats, info, host));
}
