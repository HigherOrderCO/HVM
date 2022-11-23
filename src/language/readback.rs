//! Moves HVM Terms from runtime, and building dynamic functions.

// FIXME: `as_code` and `as_term` should just call `readback`, but before doing so, we must test
// the new readback properly to ensure it is correct

use crate::language as language;
use crate::runtime as runtime;
use crate::runtime::{Ptr, Heap, Program};
use std::collections::{hash_map, HashMap, HashSet};

/// Reads back a term from Runtime's memory
pub fn as_code(heap: &Heap, prog: &Program, host: u64) -> String {
  return format!("{}", as_term(heap, prog, host));
}

/// Reads back a term from Runtime's memory
pub fn as_term(heap: &Heap, prog: &Program, host: u64) -> Box<language::syntax::Term> {
  struct CtxName<'a> {
    heap: &'a Heap,
    prog: &'a Program,
    names: &'a mut HashMap<Ptr, String>,
    seen: &'a mut HashSet<Ptr>,
  }

  fn gen_var_names(heap: &Heap, prog: &Program, ctx: &mut CtxName, term: Ptr, depth: u32) {
    if ctx.seen.contains(&term) {
      return;
    };

    ctx.seen.insert(term);

    match runtime::get_tag(term) {
      runtime::LAM => {
        let param = runtime::load_arg(&ctx.heap, term, 0);
        let body = runtime::load_arg(&ctx.heap, term, 1);
        if runtime::get_tag(param) != runtime::ERA {
          let var = runtime::Var(runtime::get_loc(term, 0));
          ctx.names.insert(var, format!("x{}", ctx.names.len()));
        };
        gen_var_names(heap, prog, ctx, body, depth + 1);
      }
      runtime::APP => {
        let lam = runtime::load_arg(&ctx.heap, term, 0);
        let arg = runtime::load_arg(&ctx.heap, term, 1);
        gen_var_names(heap, prog, ctx, lam, depth + 1);
        gen_var_names(heap, prog, ctx, arg, depth + 1);
      }
      runtime::SUP => {
        let arg0 = runtime::load_arg(&ctx.heap, term, 0);
        let arg1 = runtime::load_arg(&ctx.heap, term, 1);
        gen_var_names(heap, prog, ctx, arg0, depth + 1);
        gen_var_names(heap, prog, ctx, arg1, depth + 1);
      }
      runtime::DP0 => {
        let arg = runtime::load_arg(&ctx.heap, term, 2);
        gen_var_names(heap, prog, ctx, arg, depth + 1);
      }
      runtime::DP1 => {
        let arg = runtime::load_arg(&ctx.heap, term, 2);
        gen_var_names(heap, prog, ctx, arg, depth + 1);
      }
      runtime::OP2 => {
        let arg0 = runtime::load_arg(&ctx.heap, term, 0);
        let arg1 = runtime::load_arg(&ctx.heap, term, 1);
        gen_var_names(heap, prog, ctx, arg0, depth + 1);
        gen_var_names(heap, prog, ctx, arg1, depth + 1);
      }
      runtime::U60 => {}
      runtime::F60 => {}
      runtime::CTR | runtime::FUN => {
        let arity = runtime::arity_of(&ctx.prog.aris, term);
        for i in 0..arity {
          let arg = runtime::load_arg(&ctx.heap, term, i);
          gen_var_names(heap, prog, ctx, arg, depth + 1);
        }
      }
      _ => {}
    }
  }

  #[allow(dead_code)]
  struct CtxGo<'a> {
    heap: &'a Heap,
    prog: &'a Program,
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

  fn readback(heap: &Heap, prog: &Program, ctx: &mut CtxGo, stacks: &mut Stacks, term: Ptr, depth: u32) -> Box<language::syntax::Term> {
    match runtime::get_tag(term) {
      runtime::LAM => {
        let body = runtime::load_arg(&ctx.heap, term, 1);
        let body = readback(heap, prog, ctx, stacks, body, depth + 1);
        let bind = runtime::load_arg(&ctx.heap, term, 0);
        let name = if runtime::get_tag(bind) == runtime::ERA {
          "*".to_string()
        } else {
          let var = runtime::Var(runtime::get_loc(term, 0));
          ctx.names.get(&var).map(|s| s.clone()).unwrap_or("?".to_string())
        };
        return Box::new(language::syntax::Term::Lam { name, body });
      }
      runtime::APP => {
        let func = runtime::load_arg(&ctx.heap, term, 0);
        let argm = runtime::load_arg(&ctx.heap, term, 1);
        let func = readback(heap, prog, ctx, stacks, func, depth + 1);
        let argm = readback(heap, prog, ctx, stacks, argm, depth + 1);
        return Box::new(language::syntax::Term::App { func, argm });
      }
      runtime::SUP => {
        let col = runtime::get_ext(term);
        let empty = &Vec::new();
        let stack = stacks.get(col).unwrap_or(empty);
        if let Some(val) = stack.last() {
          let arg_idx = *val as u64;
          let val = runtime::load_arg(&ctx.heap, term, arg_idx);
          let old = stacks.pop(col);
          let got = readback(heap, prog, ctx, stacks, val, depth + 1);
          stacks.push(col, old);
          got
        } else {
          let val0 = runtime::load_arg(&ctx.heap, term, 0);
          let val1 = runtime::load_arg(&ctx.heap, term, 1);
          let val0 = readback(heap, prog, ctx, stacks, val0, depth + 1);
          let val1 = readback(heap, prog, ctx, stacks, val1, depth + 1);
          return Box::new(language::syntax::Term::Sup { val0, val1 });
        }
      }
      runtime::DP0 => {
        let col = runtime::get_ext(term);
        let val = runtime::load_arg(&ctx.heap, term, 2);
        stacks.push(col, false);
        let result = readback(heap, prog, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      runtime::DP1 => {
        let col = runtime::get_ext(term);
        let val = runtime::load_arg(&ctx.heap, term, 2);
        stacks.push(col, true);
        let result = readback(heap, prog, ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      runtime::OP2 => {
        let oper = match runtime::get_ext(term) {
          runtime::ADD => language::syntax::Oper::Add,
          runtime::SUB => language::syntax::Oper::Sub,
          runtime::MUL => language::syntax::Oper::Mul,
          runtime::DIV => language::syntax::Oper::Div,
          runtime::MOD => language::syntax::Oper::Mod,
          runtime::AND => language::syntax::Oper::And,
          runtime::OR  => language::syntax::Oper::Or,
          runtime::XOR => language::syntax::Oper::Xor,
          runtime::SHL => language::syntax::Oper::Shl,
          runtime::SHR => language::syntax::Oper::Shr,
          runtime::LTN => language::syntax::Oper::Ltn,
          runtime::LTE => language::syntax::Oper::Lte,
          runtime::EQL => language::syntax::Oper::Eql,
          runtime::GTE => language::syntax::Oper::Gte,
          runtime::GTN => language::syntax::Oper::Gtn,
          runtime::NEQ => language::syntax::Oper::Neq,
          _       => panic!("unknown operation"),
        };
        let val0 = runtime::load_arg(&ctx.heap, term, 0);
        let val1 = runtime::load_arg(&ctx.heap, term, 1);
        let val0 = readback(heap, prog, ctx, stacks, val0, depth + 1);
        let val1 = readback(heap, prog, ctx, stacks, val1, depth + 1);
        return Box::new(language::syntax::Term::Op2 { oper, val0, val1 });
      }
      runtime::U60 => {
        let numb = runtime::get_num(term);
        return Box::new(language::syntax::Term::U6O { numb });
      }
      runtime::F60 => {
        let numb = runtime::get_num(term);
        return Box::new(language::syntax::Term::F6O { numb });
      }
      runtime::CTR | runtime::FUN => {
        let func = runtime::get_ext(term);
        let arit = runtime::arity_of(&ctx.prog.aris, term);
        let mut args = Vec::new();
        for i in 0 .. arit {
          let arg = runtime::load_arg(&ctx.heap, term, i);
          args.push(readback(heap, prog, ctx, stacks, arg, depth + 1));
        }
        let name = ctx.prog.nams.get(&func).map(String::to_string).unwrap_or_else(|| format!("${}", func));
        return Box::new(language::syntax::Term::Ctr { name, args });
      }
      runtime::VAR => {
        let name = ctx.names.get(&term).map(String::to_string).unwrap_or_else(|| format!("^{}", runtime::get_loc(term, 0)));
        return Box::new(language::syntax::Term::Var { name }); // ............... /\ why this sounds so threatening?
      }
      runtime::ARG => {
        return Box::new(language::syntax::Term::Var { name: "<arg>".to_string() });
      }
      runtime::ERA => {
        return Box::new(language::syntax::Term::Var { name: "<era>".to_string() });
      }
      _ => {
        return Box::new(language::syntax::Term::Var { name: format!("<unknown_tag_{}>", runtime::get_tag(term)) });
      }
    }
  }

  let term = runtime::load_ptr(heap, host);

  let mut names = HashMap::<Ptr, String>::new();
  let mut seen = HashSet::<Ptr>::new();

  let ctx = &mut CtxName { heap, prog, names: &mut names, seen: &mut seen };
  gen_var_names(heap, prog, ctx, term, 0);

  let ctx = &mut CtxGo { heap, prog, names: &names, seen: &seen };
  let mut stacks = Stacks::new();
  readback(heap, prog, ctx, &mut stacks, term, 0)
}

// Reads a term linearly, i.e., preserving dups
pub fn as_linear_term(heap: &Heap, prog: &Program, host: u64) -> Box<language::syntax::Term> {
  enum StackItem {
    Term(Ptr),
    Resolver(Ptr),
  }

  fn ctr_name(prog: &Program, id: u64) -> String {
    if let Some(name) = prog.nams.get(&id) {
      return name.clone();
    } else {
      return format!("${}", id);
    }
  }

  fn dups(heap: &Heap, prog: &Program, term: Ptr, names: &mut HashMap<u64, String>) -> language::syntax::Term {
    let mut lets: HashMap<u64, u64> = HashMap::new();
    let mut kinds: HashMap<u64, u64> = HashMap::new();
    let mut stack = vec![term];
    while !stack.is_empty() {
      let term = stack.pop().unwrap();
      match runtime::get_tag(term) {
        runtime::LAM => {
          names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
          stack.push(runtime::load_arg(heap, term, 1));
        }
        runtime::APP => {
          stack.push(runtime::load_arg(heap, term, 1));
          stack.push(runtime::load_arg(heap, term, 0));
        }
        runtime::SUP => {
          stack.push(runtime::load_arg(heap, term, 1));
          stack.push(runtime::load_arg(heap, term, 0));
        }
        runtime::DP0 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(runtime::get_loc(term, 0)) {
            names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(runtime::get_loc(term, 0), runtime::get_ext(term));
            e.insert(runtime::get_loc(term, 0));
            stack.push(runtime::load_arg(heap, term, 2));
          }
        }
        runtime::DP1 => {
          if let hash_map::Entry::Vacant(e) = lets.entry(runtime::get_loc(term, 0)) {
            names.insert(runtime::get_loc(term, 0), format!("{}", names.len()));
            kinds.insert(runtime::get_loc(term, 0), runtime::get_ext(term));
            e.insert(runtime::get_loc(term, 0));
            stack.push(runtime::load_arg(heap, term, 2));
          }
        }
        runtime::OP2 => {
          stack.push(runtime::load_arg(heap, term, 1));
          stack.push(runtime::load_arg(heap, term, 0));
        }
        runtime::CTR | runtime::FUN => {
          let arity = runtime::arity_of(&prog.aris, term);
          for i in (0..arity).rev() {
            stack.push(runtime::load_arg(heap, term, i));
          }
        }
        _ => {}
      }
    }

    let cont = expr(heap, prog, term, &names);
    if lets.is_empty() {
      cont
    } else {
      let mut output = language::syntax::Term::Var { name: "?".to_string() };
      for (i, (_key, pos)) in lets.iter().enumerate() {
        // todo: reverse
        let what = String::from("?h");
        let name = names.get(&pos).unwrap_or(&what);
        let nam0 = if runtime::load_ptr(heap, pos + 0) == runtime::Era() { String::from("*") } else { format!("a{}", name) };
        let nam1 = if runtime::load_ptr(heap, pos + 1) == runtime::Era() { String::from("*") } else { format!("b{}", name) };
        let expr = expr(heap, prog, runtime::load_ptr(heap, pos + 2), &names);
        if i == 0 {
          output = language::syntax::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(cont.clone()) };
        } else {
          output = language::syntax::Term::Dup { nam0, nam1, expr: Box::new(expr), body: Box::new(output) };
        }
      }
      output
    }
  }

  fn expr(heap: &Heap, prog: &Program, term: Ptr, names: &HashMap<u64, String>) -> language::syntax::Term {
    let mut stack = vec![StackItem::Term(term)];
    let mut output : Vec<language::syntax::Term> = Vec::new();
    while !stack.is_empty() {
      let item = stack.pop().unwrap();
      match item {
        StackItem::Resolver(term) => {
          match runtime::get_tag(term) {
            runtime::CTR => {
              let func = runtime::get_ext(term);
              let arit = runtime::arity_of(&prog.aris, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(prog, func);
              output.push(language::syntax::Term::Ctr { name, args });
            },
            runtime::FUN => {
              let func = runtime::get_ext(term);
              let arit = runtime::arity_of(&prog.aris, term);
              let mut args = Vec::new();
              for _ in 0..arit {
                args.push(Box::new(output.pop().unwrap()));
              }
              let name = ctr_name(prog, func);
              output.push(language::syntax::Term::Ctr { name, args });
            }
            runtime::LAM => {
              let name = format!("x{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?")));
              let body = Box::new(output.pop().unwrap());
              output.push(language::syntax::Term::Lam { name, body });
            }
            runtime::APP => {
              let argm = Box::new(output.pop().unwrap());
              let func = Box::new(output.pop().unwrap());
              output.push(language::syntax::Term::App { func , argm });
            }
            runtime::OP2 => {
              let oper = runtime::get_ext(term);
              let oper = match oper {
                runtime::ADD => language::syntax::Oper::Add,
                runtime::SUB => language::syntax::Oper::Sub,
                runtime::MUL => language::syntax::Oper::Mul,
                runtime::DIV => language::syntax::Oper::Div,
                runtime::MOD => language::syntax::Oper::Mod,
                runtime::AND => language::syntax::Oper::And,
                runtime::OR  => language::syntax::Oper::Or,
                runtime::XOR => language::syntax::Oper::Xor,
                runtime::SHL => language::syntax::Oper::Shl,
                runtime::SHR => language::syntax::Oper::Shr,
                runtime::LTN => language::syntax::Oper::Ltn,
                runtime::LTE => language::syntax::Oper::Lte,
                runtime::EQL => language::syntax::Oper::Eql,
                runtime::GTE => language::syntax::Oper::Gte,
                runtime::GTN => language::syntax::Oper::Gtn,
                runtime::NEQ => language::syntax::Oper::Neq,
                _       => panic!("Invalid operator."),
              };
              let val1 = Box::new(output.pop().unwrap());
              let val0 = Box::new(output.pop().unwrap());
              output.push(language::syntax::Term::Op2 { oper, val0, val1 })
            }
            _ => panic!("Term not valid in readback"),
          }
        },
        StackItem::Term(term) => {
          match runtime::get_tag(term) {
            runtime::DP0 => {
              let name = format!("a{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?a")));
              output.push(language::syntax::Term::Var { name });
            }
            runtime::DP1 => {
              let name = format!("b{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?b")));
              output.push(language::syntax::Term::Var { name });
            }
            runtime::VAR => {
              let name = format!("x{}", names.get(&runtime::get_loc(term, 0)).unwrap_or(&String::from("?x")));
              output.push(language::syntax::Term::Var { name });
            }
            runtime::LAM => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::load_arg(heap, term, 1)));
            }
            runtime::APP => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::load_arg(heap, term, 1)));
              stack.push(StackItem::Term(runtime::load_arg(heap, term, 0)));
            }
            runtime::SUP => {}
            runtime::OP2 => {
              stack.push(StackItem::Resolver(term));
              stack.push(StackItem::Term(runtime::load_arg(heap, term, 1)));
              stack.push(StackItem::Term(runtime::load_arg(heap, term, 0)));
            }
            runtime::U60 => {
              let numb = runtime::get_num(term);
              output.push(language::syntax::Term::U6O { numb });
            }
            runtime::F60 => {
              let numb = runtime::get_num(term);
              output.push(language::syntax::Term::F6O { numb });
            }
            runtime::CTR => {
              let arit = runtime::arity_of(&prog.aris, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(runtime::load_arg(heap, term, i)));
              }
            }
            runtime::FUN => {
              let arit = runtime::arity_of(&prog.aris, term);
              stack.push(StackItem::Resolver(term));
              for i in 0..arit {
                stack.push(StackItem::Term(runtime::load_arg(heap, term, i)));
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
  Box::new(dups(heap, prog, runtime::load_ptr(heap, host), &mut names))
}

/// Reads back a term from Runtime's memory
pub fn as_linear_code(heap: &Heap, prog: &Program, host: u64) -> String {
  return format!("{}", as_linear_term(heap, prog, host));
}


// This reads a term in the `(String.cons ... String.nil)` shape directly into a string.
pub fn as_string(heap: &Heap, prog: &Program, tids: &[usize], host: u64) -> Option<String> {
  let mut host = host;
  let mut text = String::new();
  loop {
    let term = runtime::reduce(heap, prog, tids, host, false);
    if runtime::get_tag(term) == runtime::CTR {
      let fid = runtime::get_ext(term);
      if fid == runtime::STRING_NIL {
        break;
      }
      if fid == runtime::STRING_CONS {
        let chr = runtime::reduce(heap, prog, tids, runtime::get_loc(term, 0), false);
        if runtime::get_tag(chr) == runtime::U60 {
          text.push(std::char::from_u32(runtime::get_num(chr) as u32).unwrap_or('?'));
          host = runtime::get_loc(term, 1);
          continue;
        } else {
          return None;
        }
      }
      return None;
    } else {
      return None;
    }
  }
  return Some(text);
}
