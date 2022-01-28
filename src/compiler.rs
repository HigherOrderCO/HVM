#![allow(clippy::identity_op)]

use crate::builder as bd;
use crate::language as lang;
use crate::rulebook as rb;
use crate::runtime as rt;
use std::io::Write;

pub fn compile_code_and_save(code: &str, file_name: &str) -> std::io::Result<()> {
  let as_clang = compile_code(code);
  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(true)
    .open(file_name)?;
  file.write_all(as_clang.as_bytes())?;
  Ok(())
}

pub fn compile_code(code: &str) -> String {
  let file = lang::read_file(code);
  let book = rb::gen_rulebook(&file);
  let funs = bd::build_runtime_functions(&book);
  compile_book(&book)
}

pub fn compile_name(name: &str) -> String {
  format!("_{}_", name.to_uppercase())
}

pub fn compile_book(comp: &rb::RuleBook) -> String {
  let mut c_ids = String::new();
  let mut inits = String::new();
  let mut codes = String::new();
  let mut id2nm = String::new();
  for (id, name) in &comp.id_to_name {
    line(&mut id2nm, 1, &format!(r#"id_to_name_data[{}] = "{}";"#, id, name));
  }
  for (name, (arity, rules)) in &comp.func_rules {
    let (init, code) = compile_func(comp, rules, 7);

    line(
      &mut c_ids,
      0,
      &format!("const u64 {} = {};", &compile_name(name), comp.name_to_id.get(name).unwrap_or(&0)),
    );

    line(&mut inits, 6, &format!("case {}: {{", &compile_name(name)));
    inits.push_str(&init);
    line(&mut inits, 6, "};");

    line(&mut codes, 6, &format!("case {}: {{", &compile_name(name)));
    codes.push_str(&code);
    line(&mut codes, 7, "break;");
    line(&mut codes, 6, "};");
  }

  clang_runtime_template(&c_ids, &inits, &codes, &id2nm, comp.id_to_name.len() as u64)
}

pub fn compile_func(comp: &rb::RuleBook, rules: &[lang::Rule], tab: u64) -> (String, String) {
  let dynfun = bd::build_dynfun(comp, rules);

  let mut dups = 0;
  let mut init = String::new();
  let mut code = String::new();

  // Converts redex vector to stricts vector
  // TODO: avoid code duplication, this same algo is on builder.rs
  let arity = dynfun.redex.len() as u64;
  let mut stricts = Vec::new();
  for (i, is_redex) in dynfun.redex.iter().enumerate() {
    if *is_redex {
      stricts.push(i as u64);
    }
  }

  // Computes the initializer, which calls reduce recursivelly
  line(&mut init, tab + 0, &format!("if (get_ari(term) == {}) {{", dynfun.redex.len()));
  if stricts.is_empty() {
    line(&mut init, tab + 1, "init = 0;");
  } else {
    line(&mut init, tab + 1, "stk_push(&stack, host);");
    for i in &stricts {
      if *i < stricts.len() as u64 - 1 {
        line(&mut init, tab + 1, &format!("stk_push(&stack, get_loc(term, {}) | 0x80000000);", i));
      } else {
        line(&mut init, tab + 1, &format!("host = get_loc(term, {});", i));
      }
    }
  }
  line(&mut init, tab + 1, "continue;");
  line(&mut init, tab + 0, "}");

  // Applies the cal_par rule to superposed args
  for (i, is_redex) in dynfun.redex.iter().enumerate() {
    if *is_redex {
      line(&mut code, tab + 0, &format!("if (get_tag(ask_arg(mem,term,{})) == PAR) {{", i));
      line(
        &mut code,
        tab + 1,
        &format!("cal_par(mem, host, term, ask_arg(mem, term, {}), {});", i, i),
      );
      line(&mut code, tab + 0, "}");
    }
  }

  // For each rule condition vector
  for dynrule in &dynfun.rules {
    let mut matched: Vec<String> = Vec::new();

    // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
    for (i, cond) in dynrule.cond.iter().enumerate() {
      let i = i as u64;
      if rt::get_tag(*cond) == rt::U32 {
        let same_tag = format!("get_tag(ask_arg(mem, term, {})) == U32", i);
        let same_val = format!("get_val(ask_arg(mem, term, {})) == {}u", i, rt::get_val(*cond));
        matched.push(format!("({} && {})", same_tag, same_val));
      }
      if rt::get_tag(*cond) == rt::CTR {
        let some_tag = format!("get_tag(ask_arg(mem, term, {})) == CTR", i);
        let some_ext = format!("get_ext(ask_arg(mem, term, {})) == {}u", i, rt::get_ext(*cond));
        matched.push(format!("({} && {})", some_tag, some_ext));
      }
    }

    let conds = if matched.is_empty() { String::from("1") } else { matched.join(" && ") };
    line(&mut code, tab + 0, &format!("if ({}) {{", conds));

    // Increments the gas count
    line(&mut code, tab + 1, "inc_cost(mem);");

    // Builds the right-hand side term (ex: `(Succ (Add a b))`)
    //let done = compile_func_rule_body(&mut code, tab + 1, &dynrule.body, &dynrule.vars);
    let done = compile_func_rule_term(&mut code, tab + 1, &dynrule.term, &dynrule.vars, &mut dups);
    line(&mut code, tab + 1, &format!("u64 done = {};", done));

    // Links the host location to it
    line(&mut code, tab + 1, "link(mem, host, done);");

    // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
    line(&mut code, tab + 1, &format!("clear(mem, get_loc(term, 0), {});", dynfun.redex.len()));
    for (i, arity) in &dynrule.free {
      let i = *i as u64;
      line(
        &mut code,
        tab + 1,
        &format!("clear(mem, get_loc(ask_arg(mem, term, {}), 0), {});", i, arity),
      );
    }

    // Collects unused variables (none in this example)
    for dynvar @ bd::DynVar { param, field, erase } in dynrule.vars.iter() {
      if *erase {
        line(&mut code, tab + 1, &format!("collect(mem, {});", get_var(dynvar)));
      }
    }

    line(&mut code, tab + 1, "init = 1;");
    line(&mut code, tab + 1, "continue;");

    line(&mut code, tab + 0, "}");
  }

  (init, code)
}

pub fn compile_func_rule_term(
  code: &mut String,
  tab: u64,
  term: &bd::DynTerm,
  vars: &[bd::DynVar],
  dups: &mut u64,
) -> String {
  fn go(
    code: &mut String,
    tab: u64,
    term: &bd::DynTerm,
    vars: &mut Vec<String>,
    nams: &mut u64,
    dups: &mut u64,
  ) -> String {
    let INLINE_NUMBERS = true;
    //println!("compile {:?}", term);
    //println!("- vars: {:?}", vars);
    match term {
      bd::DynTerm::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize].clone()
        } else {
          panic!("Unbound variable.");
        }
      }
      bd::DynTerm::Dup { eras, expr, body } => {
        //if INLINE_NUMBERS {
        //line(code, tab + 0, &format!("if (get_tag({}) == U32 && get_tag({}) == U32) {{", val0, val1));
        //}

        let copy = fresh(nams, "cpy");
        let dup0 = fresh(nams, "dp0");
        let dup1 = fresh(nams, "dp1");
        let expr = go(code, tab, expr, vars, nams, dups);
        line(code, tab, &format!("u64 {} = {};", copy, expr));
        line(code, tab, &format!("u64 {};", dup0));
        line(code, tab, &format!("u64 {};", dup1));
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if (get_tag({}) == U32) {{", copy));
          line(code, tab + 1, "inc_cost(mem);");
          line(code, tab + 1, &format!("{} = {};", dup0, copy));
          line(code, tab + 1, &format!("{} = {};", dup1, copy));
          line(code, tab + 0, "} else {");
        }
        let name = fresh(nams, "dup");
        let coln = fresh(nams, "col");
        let colx = *dups;
        *dups += 1;
        line(code, tab + 1, &format!("u64 {} = alloc(mem, 3);", name));
        line(code, tab + 1, &format!("u64 {} = {};", coln, colx));
        if (eras.0) {
          line(code, tab + 1, &format!("link(mem, {} + 0, Era());", name));
        }
        if (eras.1) {
          line(code, tab + 1, &format!("link(mem, {} + 1, Era());", name));
        }
        line(code, tab + 1, &format!("link(mem, {} + 2, {});", name, copy));
        line(code, tab + 1, &format!("{} = Dp0({}, {});", dup0, colx, name));
        line(code, tab + 1, &format!("{} = Dp1({}, {});", dup1, colx, name));
        if INLINE_NUMBERS {
          line(code, tab + 0, "}");
        }
        vars.push(dup0);
        vars.push(dup1);
        let body = go(code, tab + 0, body, vars, nams, dups);
        vars.pop();
        vars.pop();
        body
      }
      bd::DynTerm::Let { expr, body } => {
        let expr = go(code, tab, expr, vars, nams, dups);
        vars.push(expr);
        let body = go(code, tab, body, vars, nams, dups);
        vars.pop();
        body
      }
      bd::DynTerm::Lam { eras, body } => {
        let name = fresh(nams, "lam");
        line(code, tab, &format!("u64 {} = alloc(mem, 2);", name));
        vars.push(format!("Var({})", name));
        let body = go(code, tab, body, vars, nams, dups);
        vars.pop();
        if *eras {
          line(code, tab, &format!("link(mem, {} + 0, Era());", name));
        }
        line(code, tab, &format!("link(mem, {} + 1, {});", name, body));
        format!("Lam({})", name)
      }
      bd::DynTerm::App { func, argm } => {
        let name = fresh(nams, "app");
        let func = go(code, tab, func, vars, nams, dups);
        let argm = go(code, tab, argm, vars, nams, dups);
        line(code, tab, &format!("u64 {} = alloc(mem, 2);", name));
        line(code, tab, &format!("link(mem, {} + 0, {});", name, func));
        line(code, tab, &format!("link(mem, {} + 1, {});", name, argm));
        format!("App({})", name)
      }
      bd::DynTerm::Ctr { func, args } => {
        let ctr_args: Vec<String> =
          args.iter().map(|arg| go(code, tab, arg, vars, nams, dups)).collect();
        let name = fresh(nams, "ctr");
        line(code, tab, &format!("u64 {} = alloc(mem, {});", name, ctr_args.len()));
        for (i, arg) in ctr_args.iter().enumerate() {
          line(code, tab, &format!("link(mem, {} + {}, {});", name, i, arg));
        }
        format!("Ctr({}, {}, {})", ctr_args.len(), func, name)
      }
      bd::DynTerm::Cal { func, args } => {
        let cal_args: Vec<String> =
          args.iter().map(|arg| go(code, tab, arg, vars, nams, dups)).collect();
        let name = fresh(nams, "cal");
        line(code, tab, &format!("u64 {} = alloc(mem, {});", name, cal_args.len()));
        for (i, arg) in cal_args.iter().enumerate() {
          line(code, tab, &format!("link(mem, {} + {}, {});", name, i, arg));
        }
        format!("Cal({}, {}, {})", cal_args.len(), func, name)
      }
      bd::DynTerm::U32 { numb } => {
        format!("U_32({})", numb)
      }
      bd::DynTerm::Op2 { oper, val0, val1 } => {
        let retx = fresh(nams, "ret");
        let name = fresh(nams, "op2");
        let val0 = go(code, tab, val0, vars, nams, dups);
        let val1 = go(code, tab, val1, vars, nams, dups);
        line(code, tab + 0, &format!("u64 {};", retx));
        // Optimization: do inline operation, avoiding Op2 allocation, when operands are already number
        if INLINE_NUMBERS {
          line(
            code,
            tab + 0,
            &format!("if (get_tag({}) == U32 && get_tag({}) == U32) {{", val0, val1),
          );
          let a = format!("get_val({})", val0);
          let b = format!("get_val({})", val1);
          match *oper {
            rt::ADD => line(code, tab + 1, &format!("{} = U_32({} + {});", retx, a, b)),
            rt::SUB => line(code, tab + 1, &format!("{} = U_32({} - {});", retx, a, b)),
            rt::MUL => line(code, tab + 1, &format!("{} = U_32({} * {});", retx, a, b)),
            rt::DIV => line(code, tab + 1, &format!("{} = U_32({} / {});", retx, a, b)),
            rt::MOD => line(code, tab + 1, &format!("{} = U_32({} % {});", retx, a, b)),
            rt::AND => line(code, tab + 1, &format!("{} = U_32({} & {});", retx, a, b)),
            rt::OR => line(code, tab + 1, &format!("{} = U_32({} | {});", retx, a, b)),
            rt::XOR => line(code, tab + 1, &format!("{} = U_32({} ^ {});", retx, a, b)),
            rt::SHL => line(code, tab + 1, &format!("{} = U_32({} << {});", retx, a, b)),
            rt::SHR => line(code, tab + 1, &format!("{} = U_32({} >> {});", retx, a, b)),
            rt::LTN => line(code, tab + 1, &format!("{} = U_32({} <  {} ? 1 : 0);", retx, a, b)),
            rt::LTE => line(code, tab + 1, &format!("{} = U_32({} <= {} ? 1 : 0);", retx, a, b)),
            rt::EQL => line(code, tab + 1, &format!("{} = U_32({} == {} ? 1 : 0);", retx, a, b)),
            rt::GTE => line(code, tab + 1, &format!("{} = U_32({} >= {} ? 1 : 0);", retx, a, b)),
            rt::GTN => line(code, tab + 1, &format!("{} = U_32({} >  {} ? 1 : 0);", retx, a, b)),
            rt::NEQ => line(code, tab + 1, &format!("{} = U_32({} != {} ? 1 : 0);", retx, a, b)),
            _ => line(code, tab + 1, &format!("{} = ?;", retx)),
          }
          line(code, tab + 1, "inc_cost(mem);");
          line(code, tab + 0, "} else {");
        }
        line(code, tab + 1, &format!("u64 {} = alloc(mem, 2);", name));
        line(code, tab + 1, &format!("link(mem, {} + 0, {});", name, val0));
        line(code, tab + 1, &format!("link(mem, {} + 1, {});", name, val1));
        let oper_name = match *oper {
          rt::ADD => "ADD",
          rt::SUB => "SUB",
          rt::MUL => "MUL",
          rt::DIV => "DIV",
          rt::MOD => "MOD",
          rt::AND => "AND",
          rt::OR => "OR",
          rt::XOR => "XOR",
          rt::SHL => "SHL",
          rt::SHR => "SHR",
          rt::LTN => "LTN",
          rt::LTE => "LTE",
          rt::EQL => "EQL",
          rt::GTE => "GTE",
          rt::GTN => "GTN",
          rt::NEQ => "NEQ",
          _ => "?",
        };
        line(code, tab + 1, &format!("{} = Op2({}, {});", retx, oper_name, name));
        if (INLINE_NUMBERS) {
          line(code, tab + 0, "}");
        }
        retx
      }
    }
  }
  fn fresh(nams: &mut u64, name: &str) -> String {
    let name = format!("{}_{}", name, nams);
    *nams += 1;
    name
  }
  let mut nams = 0;
  let mut vars: Vec<String> = vars
    .iter()
    .map(|var @ bd::DynVar { param, field, erase }| {
      match field {
        Some(field) => {
          format!("ask_arg(mem, ask_arg(mem, term, {}), {})", param, field)
        }
        None => {
          format!("ask_arg(mem, term, {})", param)
        }
      }
    })
    .collect();
  go(code, tab, term, &mut vars, &mut nams, dups)
}

// This isn't used, but it is an alternative way to compile right-hand side bodies. It results in
// slightly different code that might be faster since it inlines many memory writes. But it doesn't
// optimize numeric operations to avoid extra rules, so that may make it slower, depending.
pub fn compile_func_rule_body(
  code: &mut String,
  tab: u64,
  body: &bd::Body,
  vars: &[bd::DynVar],
) -> String {
  let (elem, nodes) = body;
  for (i, node) in nodes.iter().enumerate() {
    line(code, tab + 0, &format!("u64 loc_{} = alloc(mem, {});", i, node.len()));
  }
  for (i, node) in nodes.iter().enumerate() {
    for (j, element) in node.iter().enumerate() {
      match element {
        bd::Elem::Fix { value } => {
          //mem.node[(host + j) as usize] = *value;
          line(code, tab + 0, &format!("mem->node[loc_{} + {}] = {:#x}u;", i, j, value));
        }
        bd::Elem::Ext { index } => {
          //rt::link(mem, host + j, get_var(mem, term, &vars[*index as usize]));
          line(
            code,
            tab + 0,
            &format!("link(mem, loc_{} + {}, {});", i, j, get_var(&vars[*index as usize])),
          );
          //line(code, tab + 0, &format!("u64 lnk = {};", get_var(&vars[*index as usize])));
          //line(code, tab + 0, &format!("u64 tag = get_tag(lnk);"));
          //line(code, tab + 0, &format!("mem.node[loc_{} + {}] = lnk;", i, j));
          //line(code, tab + 0, &format!("if (tag <= VAR) mem.node[get_loc(lnk, tag & 1)] = Arg(loc_{} + {});", i, j));
        }
        bd::Elem::Loc { value, targ, slot } => {
          //mem.node[(host + j) as usize] = value + hosts[*targ as usize] + slot;
          line(
            code,
            tab + 0,
            &format!("mem->node[loc_{} + {}] = {:#x}u + loc_{} + {};", i, j, value, targ, slot),
          );
        }
      }
    }
  }
  match elem {
    bd::Elem::Fix { value } => format!("{}u", value),
    bd::Elem::Ext { index } => get_var(&vars[*index as usize]),
    bd::Elem::Loc { value, targ, slot } => format!("({}u + loc_{} + {})", value, targ, slot),
  }
}

fn get_var(var: &bd::DynVar) -> String {
  let bd::DynVar { param, field, erase } = var;
  match field {
    Some(i) => {
      format!("ask_arg(mem, ask_arg(mem, term, {}), {})", param, i)
    }
    None => {
      format!("ask_arg(mem, term, {})", param)
    }
  }
}

fn line(code: &mut String, tab: u64, line: &str) {
  for i in 0..tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}

pub fn clang_runtime_template(
  c_ids: &str,
  inits: &str,
  codes: &str,
  id2nm: &str,
  names_count: u64,
) -> String {
  return format!(include_str!("template.c"), c_ids, inits, codes, names_count, id2nm);
}
