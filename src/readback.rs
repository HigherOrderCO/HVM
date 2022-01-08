use crate::lambolt as lb;
use crate::runtime as rt;
use crate::runtime::{Lnk, Worker};
use std::collections::{HashMap, HashSet};
use std::fmt;

struct State<'a> {
  mem: &'a Worker,
  names: &'a mut HashMap<Lnk, String>,
  seen: &'a mut HashSet<u64>,
  count: &'a mut u32,
}

fn name(state: &mut State, term: Lnk, depth: u32) {
  // let &mut State{mem, ref mut seen, ref mut names, ref mut count} = state; // TODO: ???
  if state.seen.contains(&term) {
    return;
  };
  match rt::get_tag(term) {
    rt::LAM => {
      let param = rt::ask_arg(state.mem, term, 0);
      let body = rt::ask_arg(state.mem, term, 1);
      if rt::get_tag(param) != rt::ERA {
        let var = rt::Var(rt::get_loc(term, 0));
        state.names.insert(var, format!("x{}", *state.count));
        *state.count += 1;
      };
      name(state, body, depth + 1);
    }
    rt::APP => {
      let lam = rt::ask_arg(state.mem, term, 0);
      let arg = rt::ask_arg(state.mem, term, 1);
      name(state, lam, depth + 1);
      name(state, arg, depth + 1);
    }
    rt::PAR => {
      let arg0 = rt::ask_arg(state.mem, term, 0);
      let arg1 = rt::ask_arg(state.mem, term, 1);
      name(state, arg0, depth + 1);
      name(state, arg1, depth + 1);
    }
    rt::DP0 => {
      let arg = rt::ask_arg(state.mem, term, 2);
      name(state, arg, depth + 1);
    }
    rt::DP1 => {
      let arg = rt::ask_arg(state.mem, term, 2);
      name(state, arg, depth + 1);
    }
    rt::OP2 => {
      let arg0 = rt::ask_arg(state.mem, term, 0);
      let arg1 = rt::ask_arg(state.mem, term, 1);
      name(state, arg0, depth + 1);
      name(state, arg1, depth + 1);
    }
    rt::U32 => {}
    rt::CTR | rt::FUN => {
      let arity = rt::get_ari(term);
      for i in 0..arity {
        let arg = rt::ask_arg(state.mem, term, i);
        name(state, arg, depth + 1);
      }
    }
    default => {}
  }
}

fn go(
  // mem: &Worker,
  // names: &mut HashMap<Lnk, String>,
  // seen: &mut HashSet<u64>,
  state: &mut State,
  term: Lnk,
  depth: u32,
) -> String {
  if state.seen.contains(&term) {
    "@".to_string()
  } else {
    match rt::get_tag(term) {
      rt::LAM => {
        let body = rt::ask_arg(state.mem, term, 1);
        let body_txt = go(state, body, depth + 1);
        let arg = rt::ask_arg(state.mem, term, 0);
        let name_txt = if rt::get_tag(arg) == rt::ERA {
          "~"
        } else {
          let var = rt::Var(rt::get_loc(term, 0));
          state.names.get(&var).map(|s| s as &str).unwrap_or("?")
        };
        format!("λ{} {}", name_txt, body_txt)
      }
      rt::APP => {
        let func = rt::ask_arg(state.mem, term, 0);
        let func_txt = go(state, func, depth + 1);
        let argm = rt::ask_arg(state.mem, term, 1);
        let argm_txt = go(state, func, depth + 1);
        format!("({} {})", func_txt, argm_txt)
      }
      rt::PAR => {
        panic!("TODO: not implemented")
      }
      rt::DP0 => {
        panic!("TODO: not implemented")
      }
      rt::DP1 => {
        panic!("TODO: not implemented")
      }
      rt::OP2 => {
        panic!("TODO: not implemented")
      }
      rt::U32 => {
        panic!("TODO: not implemented")
      }
      rt::CTR | rt::FUN => {
        let func = rt::get_fun(term);
        let arit = rt::get_ari(term);
        let args = (0..arit).map(|i| {
          let arg = rt::ask_arg(state.mem, term, i);
          let arg_txt = go(state, arg, depth + 1);
          arg_txt
        });
        // TODO: Handle `table[func] ||`.
        let name = format!("${}", func);
        format!(
          "({}{})",
          name,
          args.map(|x| format!(" {}", x)).collect::<String>()
        )
      }
      default => {
        format!("?({})", rt::get_tag(term))
      }
    }
  }
}

pub fn runtime_to_lambolt(mem: &Worker, input_term: Option<u64>, table: ()) {
  let term: Lnk = input_term.unwrap_or(rt::ask_lnk(mem, 0));
  let names = HashMap::<String, String>::new();
  let count: u32 = 0;
  let seen = HashMap::<u32, bool>::new();
}
