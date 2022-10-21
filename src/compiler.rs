#![allow(clippy::identity_op)]

//use regex::Regex;
use std::collections::HashMap;
//use std::io::Write;

use crate::language as lang;
use crate::rulebook as rb;
use crate::runtime as rt;

pub fn compile(code: &str, file_name: &str, heap_size: usize, parallel: bool) -> std::io::Result<()> {
  let language = include_str!("language.rs");
  let parser   = include_str!("parser.rs");
  let readback = include_str!("readback.rs");
  let (runtime, rulebook) = compile_code(code, heap_size, parallel).unwrap();

  let cargo = r#"
[package]
name = "hvm-app"
version = "0.1.0"
edition = "2021"
description = "An HVM application"
repository = "https://github.com/Kindelia/HVM"
license = "MIT"
keywords = ["functional", "language", "runtime", "compiler", "target"]
categories = ["compilers"]

[[bin]]
name = "hvm-app"
test = false

[dependencies]
crossbeam = "0.8.2"
thread-priority = "0.9.2"
itertools = "0.10"
num_cpus = "1.13"
regex = "1.5.4"
fastrand = "1.8.0"
highlight_error = "0.1.1"
clap = { version = "3.1.8", features = ["derive"] }
wasm-bindgen = "0.2.82"
reqwest = { version = "0.11.11", features = ["blocking"] }
web-sys = { version = "0.3", features = ["console"] }
instant = { version = "0.1", features = [ "wasm-bindgen", "inaccurate" ] }
  "#;

  let main = r#"
#![feature(atomic_from_mut)]
#![feature(atomic_mut_ptr)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_macros)]
#![allow(unused_parens)]

mod language;
mod parser;
mod readback;
mod rulebook;
mod runtime;

fn make_main_call(params: &Vec<String>) -> Result<language::Term, String> {
  let name = "Main".to_string();
  let args = params.iter().map(|x| language::read_term(x).unwrap()).collect();
  return Ok(language::Term::Ctr { name, args });
}

fn run_code(code: &str, debug: bool, params: Vec<String>, size: usize) -> Result<(), String> {
  let call = make_main_call(&params)?;
  let (norm, cost, size, time) = rulebook::eval_code(&call, code, debug, size)?;
  println!("{}", norm);
  eprintln!();
  eprintln!("Rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  eprintln!("Mem.Size: {}", size);
  return Ok(());
}

fn main() -> Result<(), String> {
  let params : Vec<String> = vec![];
  let size = 67108864;
  let debug = false;
  run_code("(Main) = (Fib 5)", debug, params, size)?;
  return Ok(());
}
"#;

  std::fs::create_dir("./_hvm_").ok();
  std::fs::create_dir("./_hvm_/src").ok();
  std::fs::write("./_hvm_/src/language.rs", language)?;
  std::fs::write("./_hvm_/src/parser.rs", parser)?;
  std::fs::write("./_hvm_/src/readback.rs", readback)?;
  std::fs::write("./_hvm_/src/rulebook.rs", rulebook)?;
  std::fs::write("./_hvm_/src/runtime.rs", runtime)?;
  std::fs::write("./_hvm_/src/main.rs", main)?;
  std::fs::write("./_hvm_/Cargo.toml", cargo)?;

  return Ok(());
}

fn compile_name(name: &str) -> String {
  // TODO: this can still cause some name collisions.
  // Note: avoiding the use of `$` because it is not an actually valid
  // identifier character in C.
  //let name = name.replace('_', "__");
  let name = name.replace('.', "_").replace('$', "_S_");
  format!("_{}_", name)
}

fn compile_code(code: &str, heap_size: usize, parallel: bool) -> Result<(String,String), String> {
  let file = lang::read_file(code)?;
  let book = rb::gen_rulebook(&file);
  rb::build_dynamic_functions(&book);
  Ok(compile_book(&book, heap_size, parallel))
}

fn compile_book(comp: &rb::RuleBook, heap_size: usize, parallel: bool) -> (String, String) {

  // Function matches and rewrite rules

  let mut fun_match = String::new();
  let mut fun_rules = String::new();

  for (name, (arity, rules)) in &comp.rule_group {
    let (init, func) = compile_func(comp, &name, rules, 11);

    line(&mut fun_match, 10, &format!("{} => {{", &compile_name(name)));
    fun_match.push_str(&init);
    line(&mut fun_match, 10, &format!("}}"));

    line(&mut fun_rules, 10, &format!("{} => {{", &compile_name(name)));
    fun_rules.push_str(&func);
    line(&mut fun_rules, 10, &format!("}}"));
  }

  // Constructor and function ids and arities

  let mut ctr_ids = String::new();
  let mut ctr_ari = String::new();

  for (id, arity) in &comp.id_to_arit {
    if id > &rt::HVM_MAX_RESERVED_ID {
      let name = comp.id_to_name.get(id).unwrap();
      line(&mut ctr_ari, 2, &format!(r#"{} => {{ return {}; }}"#, &compile_name(&name), arity));
    }
  }

  for (name, id) in &comp.name_to_id {
    if id > &rt::HVM_MAX_RESERVED_ID {
      line(&mut ctr_ids, 0, &format!("pub const {} : u64 = {};", &compile_name(name), id));
    }
  }

  let runtime = include_str!("runtime.rs");
  let runtime = runtime.replace("//GENERATED-FUN-IDS//", &ctr_ids);
  let runtime = runtime.replace("//GENERATED-FUN-ARI//", &ctr_ari);
  let runtime = runtime.replace("//GENERATED-FUN-CTR-MATCH//", &fun_match);
  let runtime = runtime.replace("//GENERATED-FUN-CTR-RULES//", &fun_rules);

  // Constructor and function book entries

  let mut book_entries = String::new();

  for (name, id) in &comp.name_to_id {
    if id > &rt::HVM_MAX_RESERVED_ID {
      let arity = comp.id_to_arit.get(id).unwrap_or(&0);
      let isfun = comp.ctr_is_cal.get(name).unwrap_or(&false);
      line(&mut book_entries, 1, &format!("register(&mut book, \"{}\" , rt::{}, {}, {});", name, &compile_name(name), arity, isfun));
    }
  }

  let rulebook = include_str!("rulebook.rs");
  let rulebook = rulebook.replace("//GENERATED-BOOK-ENTRIES//", &book_entries);

  return (runtime, rulebook);

  //let mut c_ids = String::new();
  //let mut inits = String::new();
  //let mut codes = String::new();
  //let mut id2nm = String::new();
  //let mut id2ar = String::new();

  //for (id, arity) in &comp.id_to_arit {
    //line(&mut id2ar, 1, &format!(r#"id_to_arity_data[{}] = {};"#, id, arity));
  //}

  //for (name, (_arity, rules)) in &comp.rule_group {
    //let (init, code) = compile_func(comp, &name, rules, 7);

    //line(
      //&mut c_ids,
      //0,
      //&format!("#define {} ({})", &compile_name(name), comp.name_to_id.get(name).unwrap_or(&0)),
    //);

    //line(&mut inits, 6, &format!("case {}: {{", &compile_name(name)));
    //inits.push_str(&init);
    //line(&mut inits, 6, "};");

    //line(&mut codes, 6, &format!("case {}: {{", &compile_name(name)));
    //codes.push_str(&code);
    //line(&mut codes, 7, "break;");
    //line(&mut codes, 6, "};");
  //}

  //c_runtime_template(heap_size, &c_ids, &inits, &codes, &id2nm, comp.id_to_name.len() as u64, &id2ar, comp.id_to_name.len() as u64, parallel)
}

fn compile_func(comp: &rb::RuleBook, fn_name: &str, rules: &[lang::Rule], tab: u64) -> (String, String) {
  let function = rb::build_function(comp, fn_name, rules);

  let mut init = String::new();
  let mut code = String::new();

  // Computes the initializer, which calls reduce recursively
  //line(&mut init, tab + 0, &format!("if ask_ari(mem, term) == {} {{", function.is_strict.len()));
  if function.stricts.is_empty() {
    line(&mut init, tab + 0, "init = false;");
  } else {
    line(&mut init, tab + 0, &format!("let goup = redex.insert(stat.tid, new_redex(host, cont, {}));", function.stricts.len()));
    for (i, strict) in function.stricts.iter().enumerate() {
      if i < function.stricts.len() - 1 {
        line(&mut init, tab + 0, &format!("visit.push(new_visit(get_loc(term, {}), goup));", strict));
      } else {
        line(&mut init, tab + 0, &format!("work = true;"));
        line(&mut init, tab + 0, &format!("init = true;"));
        line(&mut init, tab + 0, &format!("cont = goup;"));
        line(&mut init, tab + 0, &format!("host = get_loc(term, {});", strict));
      }
    }
  }
  line(&mut init, tab + 0, "continue;");
  //line(&mut init, tab + 0, "}");

  // Applies the fun_sup rule to superposed args
  for (i, is_strict) in function.is_strict.iter().enumerate() {
    if *is_strict {
      line(&mut code, tab + 0, &format!("if get_tag(ask_arg(heap,term,{})) == SUP {{", i));
      line(
        &mut code,
        tab + 1,
        &format!("fun_sup(heap, stat, info, host, term, ask_arg(heap, term, {}), {});", i, i),
      );
      line(&mut code, tab + 1, "continue;");
      line(&mut code, tab + 0, "}");
    }
  }

  // For each rule condition vector
  for (r, dynrule) in function.rules.iter().enumerate() {
    let mut matched: Vec<String> = Vec::new();

    // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
    for (i, cond) in dynrule.cond.iter().enumerate() {
      let i = i as u64;
      if rt::get_tag(*cond) == rt::NUM {
        let same_tag = format!("get_tag(ask_arg(heap, term, {})) == NUM", i);
        let same_val = format!("get_num(ask_arg(heap, term, {})) == {}", i, rt::get_num(*cond));
        matched.push(format!("({} && {})", same_tag, same_val));
      }
      if rt::get_tag(*cond) == rt::CTR {
        let some_tag = format!("get_tag(ask_arg(heap, term, {})) == CTR", i);
        let some_ext = format!("get_ext(ask_arg(heap, term, {})) == {}", i, rt::get_ext(*cond));
        matched.push(format!("({} && {})", some_tag, some_ext));
      }
        // If this is a strict argument, then we're in a default variable
      if rt::get_tag(*cond) == rt::VAR && function.is_strict[i as usize] {

        // This is a Kind2-specific optimization. Check 'HOAS_OPT'.
        if dynrule.hoas && r != function.rules.len() - 1 {

          // Matches number literals
          let is_num
            = format!("get_tag(ask_arg(heap, term, {})) == NUM", i);

          // Matches constructor labels
          let is_ctr = format!("({} && {})",
            format!("get_tag(ask_arg(heap, term, {})) == CTR", i),
            format!("ask_ari(heap, ask_arg(heap, term, {})) == 0u", i));

          // Matches HOAS numbers and constructors
          let is_hoas_ctr_num = format!("({} && {} && {})",
            format!("get_tag(ask_arg(heap, term, {})) == CTR", i),
            format!("get_ext(ask_arg(heap, term, {})) >= HOAS_CT0", i),
            format!("get_ext(ask_arg(heap, term, {})) <= HOAS_NUM", i));

          matched.push(format!("({} || {} || {})", is_num, is_ctr, is_hoas_ctr_num));

        // Only match default variables on CTRs and NUMs
        } else {
          let is_ctr = format!("get_tag(ask_arg(heap, term, {})) == CTR", i);
          let is_num = format!("get_tag(ask_arg(heap, term, {})) == NUM", i);
          matched.push(format!("({} || {})", is_ctr, is_num));
        }

      }
    }

    let conds = if matched.is_empty() { String::from("true") } else { matched.join(" && ") };
    line(&mut code, tab + 0, &format!("if {} {{", conds));

    // Increments the gas count
    line(&mut code, tab + 1, "inc_cost(stat);");

    // Builds the right-hand side term (ex: `(Succ (Add a b))`)
    //let done = compile_func_rule_body(&mut code, tab + 1, &dynrule.body, &dynrule.vars);
    let done = compile_func_rule_term(&mut code, tab + 1, &dynrule.term, &dynrule.vars);
    line(&mut code, tab + 1, &format!("let done = {};", done));

    // Links the host location to it
    line(&mut code, tab + 1, "link(heap, host, done);");

    // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
    line(&mut code, tab + 1, &format!("free(heap, get_loc(term, 0), {});", function.is_strict.len()));
    for (i, arity) in &dynrule.free {
      let i = *i as u64;
      line(
        &mut code,
        tab + 1,
        &format!("free(heap, get_loc(ask_arg(heap, term, {}), 0), {});", i, arity),
      );
    }

    // Collects unused variables (none in this example)
    for dynvar @ rt::RuleVar { param: _, field: _, erase } in dynrule.vars.iter() {
      if *erase {
        line(&mut code, tab + 1, &format!("collect(heap, stat, info, {});", get_var(dynvar)));
      }
    }

    line(&mut code, tab + 1, "init = true;");
    line(&mut code, tab + 1, "continue;");

    line(&mut code, tab + 0, "}");
  }

  (init, code)
}

fn compile_func_rule_term(
  code: &mut String,
  tab: u64,
  term: &rt::Term,
  vars: &[rt::RuleVar],
) -> String {
  fn alloc_lam(
    code: &mut String,
    tab: u64,
    nams: &mut u64,
    globs: &mut HashMap<u64, String>,
    glob: u64,
  ) -> String {
    if let Some(got) = globs.get(&glob) {
      got.clone()
    } else {
      let name = fresh(nams, "lam");
      line(code, tab, &format!("let {} = alloc(heap, stat, 2);", name));
      if glob != 0 {
        // FIXME: sanitizer still can't detect if a scopeless lambda doesn't use its bound
        // variable, so we must write an Era() here. When it does, we can remove this line.
        line(code, tab, &format!("link(heap, {} + 0, Era());", name));
        globs.insert(glob, name.clone());
      }
      name
    }
  }
  fn compile_term(
    code: &mut String,
    tab: u64,
    vars: &mut Vec<String>,
    nams: &mut u64,
    globs: &mut HashMap<u64, String>,
    term: &rt::Term,
  ) -> String {
    const INLINE_NUMBERS: bool = true;
    //println!("compile {:?}", term);
    //println!("- vars: {:?}", vars);
    match term {
      rt::Term::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize].clone()
        } else {
          panic!("Unbound variable.");
        }
      }
      rt::Term::Glo { glob } => {
        format!("Var({})", alloc_lam(code, tab, nams, globs, *glob))
      }
      rt::Term::Dup { eras, expr, body } => {
        //if INLINE_NUMBERS {
        //line(code, tab + 0, &format!("if (get_tag({}) == NUM && get_tag({}) == NUM) {{", val0, val1));
        //}

        let copy = fresh(nams, "cpy");
        let dup0 = fresh(nams, "dp0");
        let dup1 = fresh(nams, "dp1");
        let expr = compile_term(code, tab, vars, nams, globs, expr);
        line(code, tab, &format!("let {} = {};", copy, expr));
        line(code, tab, &format!("let {};", dup0));
        line(code, tab, &format!("let {};", dup1));
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if get_tag({}) == NUM {{", copy));
          line(code, tab + 1, "inc_cost(stat);");
          line(code, tab + 1, &format!("{} = {};", dup0, copy));
          line(code, tab + 1, &format!("{} = {};", dup1, copy));
          line(code, tab + 0, "} else {");
        }
        let name = fresh(nams, "dup");
        let coln = fresh(nams, "col");
        //let colx = *dups;
        //*dups += 1;
        line(code, tab + 1, &format!("let {} = alloc(heap, stat, 3);", name));
        line(code, tab + 1, &format!("let {} = gen_dup(stat);", coln));
        if eras.0 {
          line(code, tab + 1, &format!("link(heap, {} + 0, Era());", name));
        }
        if eras.1 {
          line(code, tab + 1, &format!("link(heap, {} + 1, Era());", name));
        }
        line(code, tab + 1, &format!("link(heap, {} + 2, {});", name, copy));
        line(code, tab + 1, &format!("{} = Dp0({}, {});", dup0, coln, name));
        line(code, tab + 1, &format!("{} = Dp1({}, {});", dup1, coln, name));
        if INLINE_NUMBERS {
          line(code, tab + 0, "}");
        }
        vars.push(dup0);
        vars.push(dup1);
        let body = compile_term(code, tab + 0, vars, nams, globs, body);
        vars.pop();
        vars.pop();
        body
      }
      rt::Term::Let { expr, body } => {
        let expr = compile_term(code, tab, vars, nams, globs, expr);
        vars.push(expr);
        let body = compile_term(code, tab, vars, nams, globs, body);
        vars.pop();
        body
      }
      rt::Term::Lam { eras, glob, body } => {
        let name = alloc_lam(code, tab, nams, globs, *glob);
        vars.push(format!("Var({})", name));
        let body = compile_term(code, tab, vars, nams, globs, body);
        vars.pop();
        if *eras {
          line(code, tab, &format!("link(heap, {} + 0, Era());", name));
        }
        line(code, tab, &format!("link(heap, {} + 1, {});", name, body));
        format!("Lam({})", name)
      }
      rt::Term::App { func, argm } => {
        let name = fresh(nams, "app");
        let func = compile_term(code, tab, vars, nams, globs, func);
        let argm = compile_term(code, tab, vars, nams, globs, argm);
        line(code, tab, &format!("let {} = alloc(heap, stat, 2);", name));
        line(code, tab, &format!("link(heap, {} + 0, {});", name, func));
        line(code, tab, &format!("link(heap, {} + 1, {});", name, argm));
        format!("App({})", name)
      }
      rt::Term::Ctr { func, args } => {
        let ctr_args: Vec<String> = args.iter().map(|arg| compile_term(code, tab, vars, nams, globs, arg)).collect();
        let name = fresh(nams, "ctr");
        line(code, tab, &format!("let {} = alloc(heap, stat, {});", name, ctr_args.len()));
        for (i, arg) in ctr_args.iter().enumerate() {
          line(code, tab, &format!("link(heap, {} + {}, {});", name, i, arg));
        }
        format!("Ctr({}, {}, {})", ctr_args.len(), func, name)
      }
      rt::Term::Fun { func, args } => {
        let cal_args: Vec<String> =
          args.iter().map(|arg| compile_term(code, tab, vars, nams, globs, arg)).collect();
        let name = fresh(nams, "cal");
        line(code, tab, &format!("let {} = alloc(heap, stat, {});", name, cal_args.len()));
        for (i, arg) in cal_args.iter().enumerate() {
          line(code, tab, &format!("link(heap, {} + {}, {});", name, i, arg));
        }
        format!("Fun({}, {}, {})", cal_args.len(), func, name)
      }
      rt::Term::Num { numb } => {
        format!("Num({})", numb)
      }
      rt::Term::Op2 { oper, val0, val1 } => {
        let retx = fresh(nams, "ret");
        let name = fresh(nams, "op2");
        let val0 = compile_term(code, tab, vars, nams, globs, val0);
        let val1 = compile_term(code, tab, vars, nams, globs, val1);
        line(code, tab + 0, &format!("let {};", retx));
        // Optimization: do inline operation, avoiding Op2 allocation, when operands are already number
        if INLINE_NUMBERS {
          line(
            code,
            tab + 0,
            &format!("if get_tag({}) == NUM && get_tag({}) == NUM {{", val0, val1),
          );
          let a = format!("get_num({})", val0);
          let b = format!("get_num({})", val1);
          match *oper {
            rt::ADD => line(code, tab + 1, &format!("{} = Num({} + {});", retx, a, b)),
            rt::SUB => line(code, tab + 1, &format!("{} = Num({} - {});", retx, a, b)),
            rt::MUL => line(code, tab + 1, &format!("{} = Num({} * {});", retx, a, b)),
            rt::DIV => line(code, tab + 1, &format!("{} = Num({} / {});", retx, a, b)),
            rt::MOD => line(code, tab + 1, &format!("{} = Num({} % {});", retx, a, b)),
            rt::AND => line(code, tab + 1, &format!("{} = Num({} & {});", retx, a, b)),
            rt::OR  => line(code, tab + 1, &format!("{} = Num({} | {});", retx, a, b)),
            rt::XOR => line(code, tab + 1, &format!("{} = Num({} ^ {});", retx, a, b)),
            rt::SHL => line(code, tab + 1, &format!("{} = Num({} << {});", retx, a, b)),
            rt::SHR => line(code, tab + 1, &format!("{} = Num({} >> {});", retx, a, b)),
            rt::LTN => line(code, tab + 1, &format!("{} = Num(if {} < {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            rt::LTE => line(code, tab + 1, &format!("{} = Num(if {} <= {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            rt::EQL => line(code, tab + 1, &format!("{} = Num(if {} == {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            rt::GTE => line(code, tab + 1, &format!("{} = Num(if {} >= {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            rt::GTN => line(code, tab + 1, &format!("{} = Num(if {} >  {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            rt::NEQ => line(code, tab + 1, &format!("{} = Num(if {} != {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            _ => line(code, tab + 1, &format!("{} = ?;", retx)),
          }
          line(code, tab + 1, "inc_cost(stat);");
          line(code, tab + 0, "} else {");
        }
        line(code, tab + 1, &format!("let {} = alloc(heap, stat, 2);", name));
        line(code, tab + 1, &format!("link(heap, {} + 0, {});", name, val0));
        line(code, tab + 1, &format!("link(heap, {} + 1, {});", name, val1));
        let oper_name = match *oper {
          rt::ADD => "ADD",
          rt::SUB => "SUB",
          rt::MUL => "MUL",
          rt::DIV => "DIV",
          rt::MOD => "MOD",
          rt::AND => "AND",
          rt::OR  => "OR",
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
        if INLINE_NUMBERS {
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
    .map(|_var @ rt::RuleVar { param, field, erase: _ }| match field {
      Some(field) => {
        format!("ask_arg(heap, ask_arg(heap, term, {}), {})", param, field)
      }
      None => {
        format!("ask_arg(heap, term, {})", param)
      }
    })
    .collect();
  let mut globs: HashMap<u64, String> = HashMap::new();
  compile_term(code, tab, &mut vars, &mut nams, &mut globs, term)
}

fn get_var(var: &rt::RuleVar) -> String {
  let rt::RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => {
      format!("ask_arg(heap, ask_arg(heap, term, {}), {})", param, i)
    }
    None => {
      format!("ask_arg(heap, term, {})", param)
    }
  }
}

fn line(code: &mut String, tab: u64, line: &str) {
  for _ in 0..tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}
