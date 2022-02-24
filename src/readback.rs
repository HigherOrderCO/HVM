// Moving Lambolt Terms to/from runtime, and building dynamic functions.

use crate::language as lang;
use crate::rulebook as rb;
use crate::runtime as rt;
use crate::runtime::{Lnk, Worker};
use std::collections::{HashMap, HashSet};

/// Reads back a term from Runtime's memory
// TODO: we should readback as a language::Term, not as a string
pub fn as_code(mem: &Worker, comp: &Option<rb::RuleBook>, host: u64) -> String {
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
      _ => {}
    }
  }

  #[allow(dead_code)]
  struct CtxGo<'a> {
    mem: &'a Worker,
    comp: &'a Option<rb::RuleBook>,
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
      Stacks { stacks: HashMap::new() }
    }
    fn get(&self, col: Lnk) -> Option<&Vec<bool>> {
      self.stacks.get(&col)
    }
    fn pop(&mut self, col: Lnk) -> bool {
      let stack = self.stacks.entry(col).or_insert_with(Vec::new);
      stack.pop().unwrap_or(false)
    }
    fn push(&mut self, col: Lnk, val: bool) {
      let stack = self.stacks.entry(col).or_insert_with(Vec::new);
      stack.push(val);
    }
  }

  fn go(ctx: &mut CtxGo, stacks: &mut Stacks, term: Lnk, depth: u32) -> String {
    //println!("readback term {}", rt::show_lnk(term));

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
          "_"
        } else {
          let var = rt::Var(rt::get_loc(term, 0));
          ctx.names.get(&var).map(|s| s as &str).unwrap_or("?")
        };
        format!("λ{} {}", name_txt, body_txt)
      }
      rt::APP => {
        let func = rt::ask_arg(ctx.mem, term, 0);
        let argm = rt::ask_arg(ctx.mem, term, 1);
        let func_txt = go(ctx, stacks, func, depth + 1);
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
          let old = stacks.pop(col);
          let got = go(ctx, stacks, val, depth + 1);
          stacks.push(col, old);
          got
        } else {
          let val0 = rt::ask_arg(ctx.mem, term, 0);
          let val1 = rt::ask_arg(ctx.mem, term, 1);
          let val0_txt = go(ctx, stacks, val0, depth + 1);
          let val1_txt = go(ctx, stacks, val1, depth + 1);
          format!("<{} {}>", val0_txt, val1_txt)
        }
      }
      rt::DP0 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        stacks.push(col, false);
        let result = go(ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
      }
      rt::DP1 => {
        let col = rt::get_ext(term);
        let val = rt::ask_arg(ctx.mem, term, 2);
        stacks.push(col, true);
        let result = go(ctx, stacks, val, depth + 1);
        stacks.pop(col);
        result
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
          _ => panic!("unknown operation"),
        };
        let val0 = rt::ask_arg(ctx.mem, term, 0);
        let val1 = rt::ask_arg(ctx.mem, term, 1);
        let val0_txt = go(ctx, stacks, val0, depth + 1);
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
            format!(" {}", go(ctx, stacks, arg, depth + 1))
          })
          .collect::<String>();
        let name = match ctx.comp {
          None => format!("${}", func),
          Some(x) => {
            x.id_to_name.get(&func).map(String::to_string).unwrap_or_else(|| format!("${}", func))
          }
        };
        format!("({}{})", name, args_txt)
      }
      rt::VAR => ctx
        .names
        .get(&term)
        .map(String::to_string)
        .unwrap_or_else(|| format!("^{}", rt::get_loc(term, 0))),
      rt::ARG => "!".to_string(),
      rt::ERA => "_".to_string(),
      _ => {
        format!("?({})", rt::get_tag(term))
      }
    }
    //}
  }

  let term = rt::ask_lnk(mem, host);

  let mut names = HashMap::<Lnk, String>::new();
  let mut seen = HashSet::<Lnk>::new();
  let mut count: u32 = 0;

  let ctx = &mut CtxName { mem, names: &mut names, seen: &mut seen, count: &mut count };
  name(ctx, term, 0);

  let ctx = &mut CtxGo { mem, comp, names: &names, seen: &seen };
  let mut stacks = Stacks::new();

  go(ctx, &mut stacks, term, 0)
}

pub fn as_term(
  mem: &Worker,
  comp: &Option<rb::RuleBook>,
  host: u64,
) -> Result<Box<lang::Term>, String> {
  lang::read_term(&as_code(mem, comp, host))
}
