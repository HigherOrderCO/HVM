#![allow(clippy::identity_op)]

use regex::Regex;
use std::collections::HashMap;
use std::io::Write;

use crate::builder as bd;
use crate::language as lang;
use crate::rulebook as rb;
use crate::runtime as rt;

pub fn compile_code_and_save(code: &str, file_name: &str, heap_size: usize, parallel: bool) -> Result<(), String> {
  let as_clang = compile_code(code, heap_size, parallel)?;
  let mut file = std::fs::OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(true)
    .open(file_name)
    .map_err(|err| err.to_string())?;
  file.write_all(as_clang.as_bytes()).map_err(|err| err.to_string())?;
  Ok(())
}

fn compile_code(code: &str, heap_size: usize, parallel: bool) -> Result<String, String> {
  let file = lang::read_file(code)?;
  let book = rb::gen_rulebook(&file);
  bd::build_runtime_functions(&book);
  Ok(compile_book(&book, heap_size, parallel))
}

fn compile_name(name: &str) -> String {
  // TODO: this can still cause some name collisions.
  // Note: avoiding the use of `$` because it is not an actually valid
  // identifier character in C.
  let name = name.replace('_', "__");
  let name = name.replace('.', "_");
  format!("_{}_", name.to_uppercase())
}

// Generate static initializer for C array
// from a Rust HashMap
fn gen_static_array_init(count : usize, default: String, mp: HashMap<u64,String>) -> String {
  // Step 1: put in correct order using Vec
  let mut arr = vec![default;count];
  for (idx, entry) in mp {
    arr[idx as usize] = entry;
  }  
  // Step 2: print vec to string  
  let mut out = String::new();
  line(&mut out,0,"{");
  for entry in arr {
    line(&mut out,1,&format!("{},",entry));
  }
  line(&mut out,1,"};");
  return out;
}

fn compile_book(comp: &rb::RuleBook, heap_size: usize, parallel: bool) -> String {

  // Generate c_ids
  let mut c_ids = String::new(); 
  for (id, name) in &comp.id_to_name {
    line(&mut c_ids, 0, &format!("#define {} ({})", &compile_name(name), id));
  }

  let num_ids = comp.id_to_name.len();

  // Generate static arrays id2ar and id2nm
  let id2nm = &format!("static char * id_to_name_data[] = {}",
    gen_static_array_init(num_ids, "IMPOSSIBLE".into(),
      comp.id_to_name.iter().map(|(k,v)| (k.clone(),format!("\"{}\"",v))).collect()));
  let id2ar = &format!("static u64 id_to_arity_data[] = {}",
    gen_static_array_init(num_ids,"-1".into(),
      comp.id_to_arit.iter().map(|(k,v)| (k.clone(),v.to_string())).collect()));

  // Generate rule-reduction code
  let mut inits = String::new();
  let mut codes = String::new();
  for (name, (_arity, rules)) in &comp.rule_group {
    let (init, code) = compile_func(comp, &name, rules, 7);

    line(&mut inits, 6, &format!("case {}: {{", &compile_name(name)));
    inits.push_str(&init);
    line(&mut inits, 6, "};");

    line(&mut codes, 6, &format!("case {}: {{", &compile_name(name)));
    codes.push_str(&code);
    line(&mut codes, 7, "break;");
    line(&mut codes, 6, "};");
  }

  c_runtime_template(heap_size, &c_ids, &inits, &codes, &id2nm, comp.id_to_name.len() as u64, &id2ar, comp.id_to_name.len() as u64, parallel)
}

fn compile_func(comp: &rb::RuleBook, fn_name: &str, rules: &[lang::Rule], tab: u64) -> (String, String) {
  let dynfun = bd::build_dynfun(comp, fn_name, rules);

  let mut init = String::new();
  let mut code = String::new();

  // Converts redex vector to stricts vector
  // TODO: avoid code duplication, this same algo is on builder.rs
  let _arity = dynfun.redex.len() as u64;
  let mut stricts = Vec::new();
  for (i, is_redex) in dynfun.redex.iter().enumerate() {
    if *is_redex {
      stricts.push(i as u64);
    }
  }

  // Computes the initializer, which calls reduce recursively
  line(&mut init, tab + 0, &format!("if (ask_ari(mem, term) == {}) {{", dynfun.redex.len()));
  if stricts.is_empty() {
    line(&mut init, tab + 1, "init = 0;");
  } else {
    line(&mut init, tab + 1, "stk_push(&stack, host);");
    for (i, strict) in stricts.iter().enumerate() {
      if i < stricts.len() - 1 {
        line(
          &mut init,
          tab + 1,
          &format!("stk_push(&stack, get_loc(term, {}) | 0x80000000);", strict),
        );
      } else {
        line(&mut init, tab + 1, &format!("host = get_loc(term, {});", strict));
      }
    }
  }
  line(&mut init, tab + 1, "continue;");
  line(&mut init, tab + 0, "}");

  // Applies the cal_par rule to superposed args
  for (i, is_redex) in dynfun.redex.iter().enumerate() {
    if *is_redex {
      line(&mut code, tab + 0, &format!("if (get_tag(ask_arg(mem,term,{})) == SUP) {{", i));
      line(
        &mut code,
        tab + 1,
        &format!("cal_par(mem, host, term, ask_arg(mem, term, {}), {});", i, i),
      );
      line(&mut code, tab + 1, "continue;");
      line(&mut code, tab + 0, "}");
    }
  }

  // For each rule condition vector
  for (r, dynrule) in dynfun.rules.iter().enumerate() {
    let mut matched: Vec<String> = Vec::new();

    // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
    for (i, cond) in dynrule.cond.iter().enumerate() {
      let i = i as u64;
      if rt::get_tag(*cond) == rt::NUM {
        let same_tag = format!("get_tag(ask_arg(mem, term, {})) == NUM", i);
        let same_val = format!("get_num(ask_arg(mem, term, {})) == {}u", i, rt::get_num(*cond));
        matched.push(format!("({} && {})", same_tag, same_val));
      }
      if rt::get_tag(*cond) == rt::CTR {
        let some_tag = format!("get_tag(ask_arg(mem, term, {})) == CTR", i);
        let some_ext = format!("get_ext(ask_arg(mem, term, {})) == {}u", i, rt::get_ext(*cond));
        matched.push(format!("({} && {})", some_tag, some_ext));
      }
        // If this is a strict argument, then we're in a default variable
      if rt::get_tag(*cond) == rt::VAR && dynfun.redex[i as usize] {

        // This is a Kind2-specific optimization. Check 'HOAS_OPT'.
        if dynrule.hoas && r != dynfun.rules.len() - 1 {

          // Matches number literals
          let is_num
            = format!("get_tag(ask_arg(mem, term, {})) == NUM", i);

          // Matches constructor labels
          let is_ctr = format!("({} && {})",
            format!("get_tag(ask_arg(mem, term, {})) == CTR", i),
            format!("ask_ari(mem, ask_arg(mem, term, {})) == 0u", i));

          // Matches HOAS numbers and constructors
          let is_hoas_ctr_num = format!("({} && {} && {})",
            format!("get_tag(ask_arg(mem, term, {})) == CTR", i),
            format!("get_ext(ask_arg(mem, term, {})) >= HOAS_CT0", i),
            format!("get_ext(ask_arg(mem, term, {})) <= HOAS_NUM", i));

          matched.push(format!("({} || {} || {})", is_num, is_ctr, is_hoas_ctr_num));

        // Only match default variables on CTRs and NUMs
        } else {
          let is_ctr = format!("get_tag(ask_arg(mem, term, {})) == CTR", i);
          let is_num = format!("get_tag(ask_arg(mem, term, {})) == NUM", i);
          matched.push(format!("({} || {})", is_ctr, is_num));
        }

      }
    }

    let conds = if matched.is_empty() { String::from("1") } else { matched.join(" && ") };
    line(&mut code, tab + 0, &format!("if ({}) {{", conds));

    // Increments the gas count
    line(&mut code, tab + 1, "inc_cost(mem);");

    // Builds the right-hand side term (ex: `(Succ (Add a b))`)
    //let done = compile_func_rule_body(&mut code, tab + 1, &dynrule.body, &dynrule.vars);
    let done = compile_func_rule_term(&mut code, tab + 1, &dynrule.term, &dynrule.vars);
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
    for dynvar @ bd::DynVar { param: _, field: _, erase } in dynrule.vars.iter() {
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

fn compile_func_rule_term(
  code: &mut String,
  tab: u64,
  term: &bd::DynTerm,
  vars: &[bd::DynVar],
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
      line(code, tab, &format!("u64 {} = alloc(mem, 2);", name));
      if glob != 0 {
        // FIXME: sanitizer still can't detect if a scopeless lambda doesn't use its bound
        // variable, so we must write an Era() here. When it does, we can remove this line.
        line(code, tab, &format!("link(mem, {} + 0, Era());", name));
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
    term: &bd::DynTerm,
  ) -> String {
    const INLINE_NUMBERS: bool = true;
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
      bd::DynTerm::Glo { glob } => {
        format!("Var({})", alloc_lam(code, tab, nams, globs, *glob))
      }
      bd::DynTerm::Dup { eras, expr, body } => {
        //if INLINE_NUMBERS {
        //line(code, tab + 0, &format!("if (get_tag({}) == NUM && get_tag({}) == NUM) {{", val0, val1));
        //}

        let copy = fresh(nams, "cpy");
        let dup0 = fresh(nams, "dp0");
        let dup1 = fresh(nams, "dp1");
        let expr = compile_term(code, tab, vars, nams, globs, expr);
        line(code, tab, &format!("u64 {} = {};", copy, expr));
        line(code, tab, &format!("u64 {};", dup0));
        line(code, tab, &format!("u64 {};", dup1));
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if (get_tag({}) == NUM) {{", copy));
          line(code, tab + 1, "inc_cost(mem);");
          line(code, tab + 1, &format!("{} = {};", dup0, copy));
          line(code, tab + 1, &format!("{} = {};", dup1, copy));
          line(code, tab + 0, "} else {");
        }
        let name = fresh(nams, "dup");
        let coln = fresh(nams, "col");
        //let colx = *dups;
        //*dups += 1;
        line(code, tab + 1, &format!("u64 {} = alloc(mem, 3);", name));
        line(code, tab + 1, &format!("u64 {} = gen_dupk(mem);", coln));
        if eras.0 {
          line(code, tab + 1, &format!("link(mem, {} + 0, Era());", name));
        }
        if eras.1 {
          line(code, tab + 1, &format!("link(mem, {} + 1, Era());", name));
        }
        line(code, tab + 1, &format!("link(mem, {} + 2, {});", name, copy));
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
      bd::DynTerm::Let { expr, body } => {
        let expr = compile_term(code, tab, vars, nams, globs, expr);
        vars.push(expr);
        let body = compile_term(code, tab, vars, nams, globs, body);
        vars.pop();
        body
      }
      bd::DynTerm::Lam { eras, glob, body } => {
        let name = alloc_lam(code, tab, nams, globs, *glob);
        vars.push(format!("Var({})", name));
        let body = compile_term(code, tab, vars, nams, globs, body);
        vars.pop();
        if *eras {
          line(code, tab, &format!("link(mem, {} + 0, Era());", name));
        }
        line(code, tab, &format!("link(mem, {} + 1, {});", name, body));
        format!("Lam({})", name)
      }
      bd::DynTerm::App { func, argm } => {
        let name = fresh(nams, "app");
        let func = compile_term(code, tab, vars, nams, globs, func);
        let argm = compile_term(code, tab, vars, nams, globs, argm);
        line(code, tab, &format!("u64 {} = alloc(mem, 2);", name));
        line(code, tab, &format!("link(mem, {} + 0, {});", name, func));
        line(code, tab, &format!("link(mem, {} + 1, {});", name, argm));
        format!("App({})", name)
      }
      bd::DynTerm::Ctr { func, args } => {
        let ctr_args: Vec<String> =
          args.iter().map(|arg| compile_term(code, tab, vars, nams, globs, arg)).collect();
        let name = fresh(nams, "ctr");
        line(code, tab, &format!("u64 {} = alloc(mem, {});", name, ctr_args.len()));
        for (i, arg) in ctr_args.iter().enumerate() {
          line(code, tab, &format!("link(mem, {} + {}, {});", name, i, arg));
        }
        format!("Ctr({}, {}, {})", ctr_args.len(), func, name)
      }
      bd::DynTerm::Cal { func, args } => {
        let cal_args: Vec<String> =
          args.iter().map(|arg| compile_term(code, tab, vars, nams, globs, arg)).collect();
        let name = fresh(nams, "cal");
        line(code, tab, &format!("u64 {} = alloc(mem, {});", name, cal_args.len()));
        for (i, arg) in cal_args.iter().enumerate() {
          line(code, tab, &format!("link(mem, {} + {}, {});", name, i, arg));
        }
        format!("Cal({}, {}, {})", cal_args.len(), func, name)
      }
      bd::DynTerm::Num { numb } => {
        format!("Num({}ull)", numb)
      }
      bd::DynTerm::Op2 { oper, val0, val1 } => {
        let retx = fresh(nams, "ret");
        let name = fresh(nams, "op2");
        let val0 = compile_term(code, tab, vars, nams, globs, val0);
        let val1 = compile_term(code, tab, vars, nams, globs, val1);
        line(code, tab + 0, &format!("u64 {};", retx));
        // Optimization: do inline operation, avoiding Op2 allocation, when operands are already number
        if INLINE_NUMBERS {
          line(
            code,
            tab + 0,
            &format!("if (get_tag({}) == NUM && get_tag({}) == NUM) {{", val0, val1),
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
            rt::LTN => line(code, tab + 1, &format!("{} = Num({} <  {} ? 1 : 0);", retx, a, b)),
            rt::LTE => line(code, tab + 1, &format!("{} = Num({} <= {} ? 1 : 0);", retx, a, b)),
            rt::EQL => line(code, tab + 1, &format!("{} = Num({} == {} ? 1 : 0);", retx, a, b)),
            rt::GTE => line(code, tab + 1, &format!("{} = Num({} >= {} ? 1 : 0);", retx, a, b)),
            rt::GTN => line(code, tab + 1, &format!("{} = Num({} >  {} ? 1 : 0);", retx, a, b)),
            rt::NEQ => line(code, tab + 1, &format!("{} = Num({} != {} ? 1 : 0);", retx, a, b)),
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
    .map(|_var @ bd::DynVar { param, field, erase: _ }| match field {
      Some(field) => {
        format!("ask_arg(mem, ask_arg(mem, term, {}), {})", param, field)
      }
      None => {
        format!("ask_arg(mem, term, {})", param)
      }
    })
    .collect();
  let mut globs: HashMap<u64, String> = HashMap::new();
  compile_term(code, tab, &mut vars, &mut nams, &mut globs, term)
}

fn get_var(var: &bd::DynVar) -> String {
  let bd::DynVar { param, field, erase: _ } = var;
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
  for _ in 0..tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}

/// String pattern that will be replaced on the template code.
/// Syntax:
/// ```c
/// /*! <TAG> !*/
/// ```
/// or:
/// ```c
/// /*! <TAG> */ ... /* <TAG> !*/
/// ```
// Note: `(?s)` is the flag that allows `.` to match `\n`
const REPLACEMENT_TOKEN_PATTERN: &str =
  r"(?s)(?:/\*! *(\w+?) *!\*/)|(?:/\*! *(\w+?) *\*/.+?/\* *(\w+?) *!\*/)";

fn c_runtime_template(
  heap_size: usize,
  c_ids: &str,
  inits: &str,
  codes: &str,
  id2nm: &str,
  nmlen: u64,
  id2ar: &str,
  arlen: u64,
  parallel: bool,
) -> String {
  const C_RUNTIME_TEMPLATE: &str = include_str!("runtime.c");
  // Instantiate the template with the given sections' content

  const C_HEAP_SIZE_TAG: &str = "GENERATED_HEAP_SIZE";
  const C_PARALLEL_FLAG_TAG: &str = "GENERATED_PARALLEL_FLAG";
  const C_NUM_THREADS_TAG: &str = "GENERATED_NUM_THREADS";
  const C_CONSTRUCTOR_IDS_TAG: &str = "GENERATED_CONSTRUCTOR_IDS";
  const C_REWRITE_RULES_STEP_0_TAG: &str = "GENERATED_REWRITE_RULES_STEP_0";
  const C_REWRITE_RULES_STEP_1_TAG: &str = "GENERATED_REWRITE_RULES_STEP_1";
  const C_NAME_COUNT_TAG: &str = "GENERATED_NAME_COUNT";
  const C_ID_TO_NAME_DATA_TAG: &str = "GENERATED_ID_TO_NAME_DATA";
  const C_ARITY_COUNT_TAG: &str = "GENERATED_ARITY_COUNT";
  const C_ID_TO_ARITY_DATA_TAG: &str = "GENERATED_ID_TO_ARITY_DATA";

  // TODO: Sanity checks: all tokens we're looking for must be present in the
  // `runtime.c` file.

  let re = Regex::new(REPLACEMENT_TOKEN_PATTERN).unwrap();

  // Instantiate the template with the given sections' content

  let result = re.replace_all(C_RUNTIME_TEMPLATE, |caps: &regex::Captures| {
    let tag = if let Some(cap1) = caps.get(1) {
      cap1.as_str()
    } else if let Some(cap2) = caps.get(2) {
      let cap2 = cap2.as_str();
      if let Some(cap3) = caps.get(3) {
        let cap3 = cap3.as_str();
        debug_assert!(cap2 == cap3, "Closing block tag name must match opening tag: {}.", cap2);
      }
      cap2
    } else {
      panic!("Replacement token must have a tag.")
    };

    let heap_size = &heap_size.to_string();
    let parallel_flag = if parallel { "#define PARALLEL" } else { "" };
    let num_threads = &num_cpus::get().to_string();
    let nmlen = &nmlen.to_string();
    let arlen = &arlen.to_string();
    match tag {
      C_HEAP_SIZE_TAG => heap_size,
      C_PARALLEL_FLAG_TAG => parallel_flag,
      C_NUM_THREADS_TAG => num_threads,
      C_CONSTRUCTOR_IDS_TAG => c_ids,
      C_REWRITE_RULES_STEP_0_TAG => inits,
      C_REWRITE_RULES_STEP_1_TAG => codes,
      C_NAME_COUNT_TAG => nmlen,
      C_ID_TO_NAME_DATA_TAG => id2nm,
      C_ARITY_COUNT_TAG => arlen,
      C_ID_TO_ARITY_DATA_TAG => id2ar,
      _ => panic!("Unknown replacement tag."),
    }
    .to_string()
  });

  (*result).to_string()
}
