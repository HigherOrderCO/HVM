use crate::rulebook as rb;
use crate::dynamic as dn;
use crate::lambolt as lb;
use crate::runtime as rt;

pub fn compile_book(comp: &rb::RuleBook) -> String {
  let mut code = String::new(); 
  for (name, (arity, rules)) in &comp.func_rules {
    code.push_str(&format!("\n:: {}\n", name));
    code.push_str(&compile_rule(comp, &rules));
  }
  return code;
}

pub fn compile_rule(comp: &rb::RuleBook, rules: &Vec<lb::Rule>) -> String {
  let dynfun = dn::build_dynfun(comp, rules);

  let mut code = String::new();
  let mut tab = 0;

  for i in 0 .. dynfun.redex.len() as u64 {
    if dynfun.redex[i as usize] {
      line(&mut code, tab + 0, &format!("if (get_tag(ask_arg(mem,term,{})) == PAR) {{", i));
      line(&mut code, tab + 1, &format!("cal_par(mem, host, term, ask_arg(mem, term, {}), {});", i, i));
      line(&mut code, tab + 0, &format!("}}"));
    }
  }

  // For each rule condition vector
  for dynrule in &dynfun.rules {

    let mut matched : Vec<String> = Vec::new();

    // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
    for (i, cond) in dynrule.cond.iter().enumerate() {
      let i = i as u64;
      if rt::get_tag(*cond) == rt::U32 {
        let same_tag = format!("get_tag(ask_arg(mem, term, {})) == U32", i);
        let same_val = format!("get_val(ask_arg(mem, term, {})) == {}", i, rt::get_val(*cond));
        matched.push(format!("({} && {})", same_tag, same_val));
      }
      if rt::get_tag(*cond) == rt::CTR {
        let some_tag = format!("get_tag(ask_arg(mem, term, {})) == CTR", i);
        let some_ext = format!("get_ext(ask_arg(mem, term, {})) == {}", i, rt::get_ext(*cond));
        matched.push(format!("({} && {})", some_tag, some_ext));
      }
    }

    line(&mut code, tab + 0, &format!("if ({}) {{", matched.join(" && ")));

      // Increments the gas count
    line(&mut code, tab + 1, &format!("inc_cost(mem);"));
      
    // Builds the right-hand side term (ex: `(Succ (Add a b))`)
    let done = compile_body(&mut code, tab + 1, &dynrule.body, &dynrule.vars);

    // Links the host location to it
    line(&mut code, tab + 1, &format!("link(mem, host, done);"));


    // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
    line(&mut code, tab + 1, &format!("clear(mem, get_loc(term, 0), {});", dynfun.redex.len()));
    for (i, arity) in &dynrule.free {
      let i = *i as u64;
      line(&mut code, tab + 1, &format!("clear(mem, get_loc(ask_arg(mem, term, {}), 0), {});", i, arity));
    }

    // Collects unused variables (none in this example)
    for (i, dn::DynVar {param, field, erase}) in dynrule.vars.iter().enumerate() {
      if *erase {
        line(&mut code, tab + 1, &format!("collect(mem, {});", get_var(&dynrule.vars[i])));
      }
    }

    line(&mut code, tab + 0, &format!("}}"));
  }

  return code;
}

pub fn compile_body(code: &mut String, tab: u64, body: &dn::Body, vars: &[dn::DynVar]) -> String {
  let (elem, nodes) = body;
  for i in 0 .. nodes.len() {
    line(code, tab + 0, &format!("u64 loc_{} = alloc(mem, {});", i, nodes[i].len()));
  }
  for i in 0 .. nodes.len() as u64 {
    let node = &nodes[i as usize];
    for j in 0 .. node.len() as u64 {
      match &node[j as usize] {
        dn::Elem::Fix{value} => {
          //mem.node[(host + j) as usize] = *value;
          line(code, tab + 0, &format!("mem.node[loc_{} + {}] = {:#x};", i, j, value));
        }
        dn::Elem::Ext{index} => {
          //rt::link(mem, host + j, get_var(mem, term, &vars[*index as usize]));
          line(code, tab + 0, &format!("link(mem, loc_{} + {}, {});", i, j, get_var(&vars[*index as usize])));
          //line(code, tab + 0, &format!("u64 lnk = {};", get_var(&vars[*index as usize])));
          //line(code, tab + 0, &format!("u64 tag = get_tag(lnk);"));
          //line(code, tab + 0, &format!("mem.node[loc_{} + {}] = lnk;", i, j));
          //line(code, tab + 0, &format!("if (tag <= VAR) mem.node[get_loc(lnk, tag & 1)] = Arg(loc_{} + {});", i, j));
        }
        dn::Elem::Loc{value, targ, slot} => {
          //mem.node[(host + j) as usize] = value + hosts[*targ as usize] + slot;
          line(code, tab + 0, &format!("mem.node[loc_{} + {}] = {:#x} + loc_{} + {};", i, j, value, targ, slot));
        }
      }
    }
  }
  match elem {
    dn::Elem::Fix{value} => format!("{}", value),
    dn::Elem::Ext{index} => get_var(&vars[*index as usize]),
    dn::Elem::Loc{value, targ, slot} => format!("({} + loc_{} + {})", value, targ, slot),
  }
}

fn get_var(var: &dn::DynVar) -> String {
  let dn::DynVar {param, field, erase} = var;
  match field {
    Some(i) => { format!("ask_arg(mem, ask_arg(mem, term, {}), {})", param, i) }
    None    => { format!("ask_arg(mem, term, {})", param) }
  }
}

fn line(code: &mut String, tab: u64, line: &str) {
  for i in 0 .. tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}

pub fn compile_code(code: &str) -> String {
  let file = lb::read_file(code);
  let book = rb::gen_rulebook(&file);
  let funs = dn::build_runtime_functions(&book);
  return compile_book(&book);
}
