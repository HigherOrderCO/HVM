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
  return format!(
    r#"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define PARALLEL

#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)

// Types
// -----

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long int u64;
typedef pthread_t Thd;

// Consts
// ------

const u64 U64_PER_KB = 0x80;
const u64 U64_PER_MB = 0x20000;
const u64 U64_PER_GB = 0x8000000;

#ifdef PARALLEL
#define MAX_WORKERS (8)
#else
#define MAX_WORKERS (1)
#endif
const u64 MAX_DYNFUNS = 65536;
#define MAX_ARITY (16)
const u64 MEM_SPACE = U64_PER_GB; // each worker has 1 GB of the 8 GB total

// Terms
// -----

typedef u64 Lnk;

const u64 VAL = 1;
const u64 EXT = 0x100000000; 
const u64 ARI = 0x100000000000000;
const u64 TAG = 0x1000000000000000;

const u64 DP0 = 0x0;
const u64 DP1 = 0x1;
const u64 VAR = 0x2;
const u64 ARG = 0x3;
const u64 ERA = 0x4;
const u64 LAM = 0x5;
const u64 APP = 0x6;
const u64 PAR = 0x7;
const u64 CTR = 0x8;
const u64 CAL = 0x9;
const u64 OP2 = 0xA;
const u64 U32 = 0xB;
const u64 F32 = 0xC;
const u64 OUT = 0xE;
const u64 NIL = 0xF;

const u64 ADD = 0x0;
const u64 SUB = 0x1;
const u64 MUL = 0x2;
const u64 DIV = 0x3;
const u64 MOD = 0x4;
const u64 AND = 0x5;
const u64 OR  = 0x6;
const u64 XOR = 0x7;
const u64 SHL = 0x8;
const u64 SHR = 0x9;
const u64 LTN = 0xA;
const u64 LTE = 0xB;
const u64 EQL = 0xC;
const u64 GTE = 0xD;
const u64 GTN = 0xE;
const u64 NEQ = 0xF;

//GENERATED_CONSTRUCTOR_IDS_START//
{}
//GENERATED_CONSTRUCTOR_IDS_END//

// Threads
// -------

typedef struct {{
  u64 size;
  u64* data;
}} Arr;

typedef struct {{
  u64* data;
  u64  size;
  u64  mcap;
}} Stk;

typedef struct {{
  u64  tid;
  Lnk* node;
  u64  size;
  Stk  free[MAX_ARITY];
  u64  cost;

  #ifdef PARALLEL
  u64             has_work;
  pthread_mutex_t has_work_mutex;
  pthread_cond_t  has_work_signal;

  u64             has_result;
  pthread_mutex_t has_result_mutex;
  pthread_cond_t  has_result_signal;

  Thd  thread;
  #endif
}} Worker;

// Globals
// -------

Worker workers[MAX_WORKERS];

// Array
// -----

void array_write(Arr* arr, u64 idx, u64 value) {{
  arr->data[idx] = value;
}}

u64 array_read(Arr* arr, u64 idx) {{
  return arr->data[idx];
}}

// Stack
// -----

u64 stk_growth_factor = 16;

void stk_init(Stk* stack) {{
  stack->size = 0;
  stack->mcap = stk_growth_factor;
  stack->data = malloc(stack->mcap * sizeof(u64));
}}

void stk_free(Stk* stack) {{
  free(stack->data);
}}

void stk_push(Stk* stack, u64 val) {{
  if (UNLIKELY(stack->size == stack->mcap)) {{ 
    stack->mcap = stack->mcap * stk_growth_factor;
    stack->data = realloc(stack->data, stack->mcap * sizeof(u64));
  }}
  stack->data[stack->size++] = val;
}}

u64 stk_pop(Stk* stack) {{
  if (LIKELY(stack->size > 0)) {{
    // TODO: shrink? -- impacts performance considerably
    //if (stack->size == stack->mcap / stk_growth_factor) {{
      //stack->mcap = stack->mcap / stk_growth_factor;
      //stack->data = realloc(stack->data, stack->mcap * sizeof(u64));
      //printf("shrink %llu\n", stack->mcap);
    //}}
    return stack->data[--stack->size];
  }} else {{
    return -1;
  }}
}}

u64 stk_find(Stk* stk, u64 val) {{
  for (u64 i = 0; i < stk->size; ++i) {{
    if (stk->data[i] == val) {{
      return i;
    }}
  }}
  return -1;
}}

// Memory
// ------

Lnk Var(u64 pos) {{
  return (VAR * TAG) | pos;
}}

Lnk Dp0(u64 col, u64 pos) {{
  return (DP0 * TAG) | (col * EXT) | pos;
}}

Lnk Dp1(u64 col, u64 pos) {{
  return (DP1 * TAG) | (col * EXT) | pos;
}}

Lnk Arg(u64 pos) {{
  return (ARG * TAG) | pos;
}}

Lnk Era() {{
  return (ERA * TAG);
}}

Lnk Lam(u64 pos) {{
  return (LAM * TAG) | pos;
}}

Lnk App(u64 pos) {{
  return (APP * TAG) | pos;
}}

Lnk Par(u64 col, u64 pos) {{
  return (PAR * TAG) | (col * EXT) | pos;
}}

Lnk Op2(u64 ope, u64 pos) {{
  return (OP2 * TAG) | (ope * EXT) | pos;
}}

Lnk U_32(u64 val) {{
  return (U32 * TAG) | val;
}}

Lnk Nil() {{
  return NIL * TAG;
}}

Lnk Ctr(u64 ari, u64 fun, u64 pos) {{
  return (CTR * TAG) | (ari * ARI) | (fun * EXT) | pos;
}}

Lnk Cal(u64 ari, u64 fun, u64 pos) {{ 
  return (CAL * TAG) | (ari * ARI) | (fun * EXT) | pos;
}}

Lnk Out(u64 arg, u64 fld) {{
  return (OUT * TAG) | (arg << 8) | fld;
}}

u64 get_tag(Lnk lnk) {{
  return lnk / TAG;
}}

u64 get_ext(Lnk lnk) {{
  return (lnk / EXT) & 0xFFFFFF;
}}

u64 get_val(Lnk lnk) {{
  return lnk & 0xFFFFFFFF;
}}

u64 get_ari(Lnk lnk) {{
  return (lnk / ARI) & 0xF;
}}

u64 get_loc(Lnk lnk, u64 arg) {{
  return get_val(lnk) + arg;
}}

Lnk ask_lnk(Worker* mem, u64 loc) {{
  return mem->node[loc];
}}

Lnk ask_arg(Worker* mem, Lnk term, u64 arg) {{
  return ask_lnk(mem, get_loc(term, arg));
}}

u64 link(Worker* mem, u64 loc, Lnk lnk) {{
  mem->node[loc] = lnk;
  //array_write(mem->nodes, loc, lnk);
  if (get_tag(lnk) <= VAR) {{
    mem->node[get_loc(lnk, get_tag(lnk) == DP1 ? 1 : 0)] = Arg(loc);
    //array_write(mem->nodes, get_loc(lnk, get_tag(lnk) == DP1 ? 1 : 0), Arg(loc));
  }}
  return lnk;
}}

u64 alloc(Worker* mem, u64 size) {{
  if (UNLIKELY(size == 0)) {{
    return 0;
  }} else {{
    u64 reuse = stk_pop(&mem->free[size]);
    if (reuse != -1) {{
      return reuse;
    }}
    u64 loc = mem->size;
    mem->size += size;
    return mem->tid * MEM_SPACE + loc;
    //return __atomic_fetch_add(&mem->nodes->size, size, __ATOMIC_RELAXED);
  }}
}}

void clear(Worker* mem, u64 loc, u64 size) {{
  stk_push(&mem->free[size], loc);
}}

// Debug
// -----

void debug_print_lnk(Lnk x) {{
  u64 tag = get_tag(x);
  u64 ext = get_ext(x);
  u64 val = get_val(x);
  switch (tag) {{
    case DP0: printf("DP0"); break;
    case DP1: printf("DP1"); break;
    case VAR: printf("VAR"); break;
    case ARG: printf("ARG"); break;
    case ERA: printf("ERA"); break;
    case LAM: printf("LAM"); break;
    case APP: printf("APP"); break;
    case PAR: printf("PAR"); break;
    case CTR: printf("CTR"); break;
    case CAL: printf("CAL"); break;
    case OP2: printf("OP2"); break;
    case U32: printf("U32"); break;
    case F32: printf("F32"); break;
    case OUT: printf("OUT"); break;
    case NIL: printf("NIL"); break;
    default : printf("???"); break;
  }}
  printf(":%llx:%llx", ext, val);
}}

// Garbage Collection
// ------------------

void collect(Worker* mem, Lnk term) {{
  switch (get_tag(term)) {{
    case DP0: {{
      link(mem, get_loc(term,0), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,1),0));
      break;
    }}
    case DP1: {{
      link(mem, get_loc(term,1), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,0),0));
      break;
    }}
    case VAR: {{
      link(mem, get_loc(term,0), Era());
      break;
    }}
    case LAM: {{
      if (get_tag(ask_arg(mem,term,0)) != ERA) {{
        link(mem, get_loc(ask_arg(mem,term,0),0), Era());
      }}
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }}
    case APP: {{
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }}
    case PAR: {{
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }}
    case OP2: {{
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      break;
    }}
    case U32: {{
      break;
    }}
    case CTR: case CAL: {{
      u64 arity = get_ari(term);
      for (u64 i = 0; i < arity; ++i) {{
        collect(mem, ask_arg(mem,term,i));
      }}
      clear(mem, get_loc(term,0), arity);
      break;
    }}
  }}
}}

// Terms
// -----

void inc_cost(Worker* mem) {{
  mem->cost++;
}}

void subst(Worker* mem, Lnk lnk, Lnk val) {{
  if (get_tag(lnk) != ERA) {{
    link(mem, get_loc(lnk,0), val);
  }} else {{
    collect(mem, val);
  }}
}}

Lnk cal_par(Worker* mem, u64 host, Lnk term, Lnk argn, u64 n) {{
  inc_cost(mem);
  u64 arit = get_ari(term);
  u64 func = get_ext(term);
  u64 fun0 = get_loc(term, 0);
  u64 fun1 = alloc(mem, arit);
  u64 par0 = get_loc(argn, 0);
  for (u64 i = 0; i < arit; ++i) {{
    if (i != n) {{
      u64 leti = alloc(mem, 3);
      u64 argi = ask_arg(mem, term, i);
      link(mem, fun0+i, Dp0(get_ext(argn), leti));
      link(mem, fun1+i, Dp1(get_ext(argn), leti));
      link(mem, leti+2, argi);
    }} else {{
      link(mem, fun0+i, ask_arg(mem, argn, 0));
      link(mem, fun1+i, ask_arg(mem, argn, 1));
    }}
  }}
  link(mem, par0+0, Cal(arit, func, fun0));
  link(mem, par0+1, Cal(arit, func, fun1));
  u64 done = Par(get_ext(argn), par0);
  link(mem, host, done);
  return done;
}}

Lnk reduce(Worker* mem, u64 root, u64 slen) {{
  Stk stack;
  stk_init(&stack);

  u64 init = 1;
  u32 host = (u32)root;

  while (1) {{

    u64 term = ask_lnk(mem, host);

    //printf("reducing: host=%d size=%llu init=%llu ", host, stack.size, init); debug_print_lnk(term); printf("\n");
    //for (u64 i = 0; i < size; ++i) {{
      //printf("- %llu ", stack[i]); debug_print_lnk(ask_lnk(mem, stack[i]>>1)); printf("\n");
    //}}
    
    if (init == 1) {{
      switch (get_tag(term)) {{
        case APP: {{
          stk_push(&stack, host);
          //stack[size++] = host;
          init = 1;
          host = get_loc(term, 0);
          continue;
        }}
        case DP0:
        case DP1: {{
          // TODO: reason about this, comment
          atomic_flag* flag = ((atomic_flag*)(mem->node + get_loc(term,0))) + 6;
          if (atomic_flag_test_and_set(flag) != 0) {{
            continue;
          }}
          stk_push(&stack, host);
          host = get_loc(term, 2);
          continue;
        }}
        case OP2: {{
          //printf("op2 %llu %llu\n", slen, stack.size);
          if (slen == 1 || stack.size > 0) {{
            stk_push(&stack, host);
            stk_push(&stack, get_loc(term, 0) | 0x80000000);
            //stack[size++] = host;
            //stack[size++] = get_loc(term, 0) | 0x80000000;
            host = get_loc(term, 1);
            continue;
          }}
          break;
        }}
        case CAL: {{
          u64 fun = get_ext(term);
          u64 ari = get_ari(term);
          
          switch (fun)
          //GENERATED_REWRITE_RULES_STEP_0_START//
          {{
{}
          }}
          //GENERATED_REWRITE_RULES_STEP_0_END//

          break;
        }}
      }}

    }} else {{

      switch (get_tag(term)) {{
        case APP: {{
          u64 arg0 = ask_arg(mem, term, 0);
          switch (get_tag(arg0)) {{
            case LAM: {{
              inc_cost(mem);
              subst(mem, ask_arg(mem, arg0, 0), ask_arg(mem, term, 1));
              u64 done = link(mem, host, ask_arg(mem, arg0, 1));
              clear(mem, get_loc(term,0), 2);
              clear(mem, get_loc(arg0,0), 2);
              init = 1;
              continue;
            }}
            case PAR: {{
              inc_cost(mem);
              u64 app0 = get_loc(term, 0);
              u64 app1 = get_loc(arg0, 0);
              u64 let0 = alloc(mem, 3);
              u64 par0 = alloc(mem, 2);
              link(mem, let0+2, ask_arg(mem, term, 1));
              link(mem, app0+1, Dp0(get_ext(arg0), let0));
              link(mem, app0+0, ask_arg(mem, arg0, 0));
              link(mem, app1+0, ask_arg(mem, arg0, 1));
              link(mem, app1+1, Dp1(get_ext(arg0), let0));
              link(mem, par0+0, App(app0));
              link(mem, par0+1, App(app1));
              u64 done = Par(get_ext(arg0), par0);
              link(mem, host, done);
              break;
            }}
          }}
          break;
        }}
        case DP0:
        case DP1: {{
          u64 arg0 = ask_arg(mem, term, 2);
          switch (get_tag(arg0)) {{
            case LAM: {{
              inc_cost(mem);
              u64 let0 = get_loc(term, 0);
              u64 par0 = get_loc(arg0, 0);
              u64 lam0 = alloc(mem, 2);
              u64 lam1 = alloc(mem, 2);
              link(mem, let0+2, ask_arg(mem, arg0, 1));
              link(mem, par0+1, Var(lam1));
              u64 arg0_arg_0 = ask_arg(mem, arg0, 0);
              link(mem, par0+0, Var(lam0));
              subst(mem, arg0_arg_0, Par(get_ext(term), par0));
              u64 term_arg_0 = ask_arg(mem,term,0);
              link(mem, lam0+1, Dp0(get_ext(term), let0));
              subst(mem, term_arg_0, Lam(lam0));
              u64 term_arg_1 = ask_arg(mem,term,1);                      
              link(mem, lam1+1, Dp1(get_ext(term), let0));
              subst(mem, term_arg_1, Lam(lam1));
              u64 done = Lam(get_tag(term) == DP0 ? lam0 : lam1);
              link(mem, host, done);
              init = 1;
              continue;
            }}
            case PAR: {{
              if (get_ext(term) == get_ext(arg0)) {{
                inc_cost(mem);
                subst(mem, ask_arg(mem,term,0), ask_arg(mem,arg0,0));
                subst(mem, ask_arg(mem,term,1), ask_arg(mem,arg0,1));
                u64 done = link(mem, host, ask_arg(mem, arg0, get_tag(term) == DP0 ? 0 : 1));
                clear(mem, get_loc(term,0), 3);
                clear(mem, get_loc(arg0,0), 2);
                init = 1;
                continue;
              }} else {{
                inc_cost(mem);
                u64 par0 = alloc(mem, 2);
                u64 let0 = get_loc(term,0);
                u64 par1 = get_loc(arg0,0);
                u64 let1 = alloc(mem, 3);
                link(mem, let0+2, ask_arg(mem,arg0,0));
                link(mem, let1+2, ask_arg(mem,arg0,1));
                u64 term_arg_0 = ask_arg(mem,term,0);
                u64 term_arg_1 = ask_arg(mem,term,1);
                link(mem, par1+0, Dp1(get_ext(term),let0));
                link(mem, par1+1, Dp1(get_ext(term),let1));
                link(mem, par0+0, Dp0(get_ext(term),let0));
                link(mem, par0+1, Dp0(get_ext(term),let1));
                subst(mem, term_arg_0, Par(get_ext(arg0),par0));
                subst(mem, term_arg_1, Par(get_ext(arg0),par1));
                u64 done = Par(get_ext(arg0), get_tag(term) == DP0 ? par0 : par1);
                link(mem, host, done);
                break;
              }}
            }}
          }}
          if (get_tag(arg0) == U32) {{
            inc_cost(mem);
            subst(mem, ask_arg(mem,term,0), arg0);
            subst(mem, ask_arg(mem,term,1), arg0);
            u64 done = arg0;
            link(mem, host, arg0);
            break;
          }}
          if (get_tag(arg0) == CTR) {{
            inc_cost(mem);
            u64 func = get_ext(arg0);
            u64 arit = get_ari(arg0);
            if (arit == 0) {{
              subst(mem, ask_arg(mem,term,0), Ctr(0, func, 0));
              subst(mem, ask_arg(mem,term,1), Ctr(0, func, 0));
              clear(mem, get_loc(term,0), 3);
              u64 done = link(mem, host, Ctr(0, func, 0));
            }} else {{
              u64 ctr0 = get_loc(arg0,0);
              u64 ctr1 = alloc(mem, arit);
              u64 term_arg_0 = ask_arg(mem,term,0);
              u64 term_arg_1 = ask_arg(mem,term,1);
              for (u64 i = 0; i < arit; ++i) {{
                u64 leti = i == 0 ? get_loc(term,0) : alloc(mem, 3);
                u64 arg0_arg_i = ask_arg(mem, arg0, i);
                link(mem, ctr0+i, Dp0(get_ext(term), leti));
                link(mem, ctr1+i, Dp1(get_ext(term), leti));
                link(mem, leti+2, arg0_arg_i);
              }}
              subst(mem, term_arg_0, Ctr(arit, func, ctr0));
              subst(mem, term_arg_1, Ctr(arit, func, ctr1));
              u64 done = Ctr(arit, func, get_tag(term) == DP0 ? ctr0 : ctr1);
              link(mem, host, done);
            }}
            break;
          }}
          atomic_flag* flag = ((atomic_flag*)(mem->node + get_loc(term,0))) + 6;
          atomic_flag_clear(flag);
          break;
        }}
        case OP2: {{
          //printf("! op2 %llu %llu\n", slen, stack.size);
          u64 arg0 = ask_arg(mem, term, 0);
          u64 arg1 = ask_arg(mem, term, 1);
          if (get_tag(arg0) == U32 && get_tag(arg1) == U32) {{
            inc_cost(mem);
            u64 a = get_val(arg0);
            u64 b = get_val(arg1);
            u64 c = 0;
            switch (get_ext(term)) {{
              case ADD: c = (a +  b) & 0xFFFFFFFF; break;
              case SUB: c = (a -  b) & 0xFFFFFFFF; break;
              case MUL: c = (a *  b) & 0xFFFFFFFF; break;
              case DIV: c = (a /  b) & 0xFFFFFFFF; break;
              case MOD: c = (a %  b) & 0xFFFFFFFF; break;
              case AND: c = (a &  b) & 0xFFFFFFFF; break;
              case OR : c = (a |  b) & 0xFFFFFFFF; break;
              case XOR: c = (a ^  b) & 0xFFFFFFFF; break;
              case SHL: c = (a << b) & 0xFFFFFFFF; break;
              case SHR: c = (a >> b) & 0xFFFFFFFF; break;
              case LTN: c = (a <  b) ? 1 : 0;      break;
              case LTE: c = (a <= b) ? 1 : 0;      break;
              case EQL: c = (a == b) ? 1 : 0;      break;
              case GTE: c = (a >= b) ? 1 : 0;      break;
              case GTN: c = (a >  b) ? 1 : 0;      break;
              case NEQ: c = (a != b) ? 1 : 0;      break;
            }}
            u64 done = U_32(c);
            clear(mem, get_loc(term,0), 2);
            link(mem, host, done);
            break;
          }}
          if (get_tag(arg0) == PAR) {{
            inc_cost(mem);
            u64 op20 = get_loc(term, 0);
            u64 op21 = get_loc(arg0, 0);
            u64 let0 = alloc(mem, 3);
            u64 par0 = alloc(mem, 2);
            link(mem, let0+2, arg1);
            link(mem, op20+1, Dp0(get_ext(arg0), let0));
            link(mem, op20+0, ask_arg(mem, arg0, 0));
            link(mem, op21+0, ask_arg(mem, arg0, 1));
            link(mem, op21+1, Dp1(get_ext(arg0), let0));
            link(mem, par0+0, Op2(get_ext(term), op20));
            link(mem, par0+1, Op2(get_ext(term), op21));
            u64 done = Par(get_ext(arg0), par0);
            link(mem, host, done);
            break;
          }}
          if (get_tag(arg1) == PAR) {{
            inc_cost(mem);
            u64 op20 = get_loc(term, 0);
            u64 op21 = get_loc(arg1, 0);
            u64 let0 = alloc(mem, 3);
            u64 par0 = alloc(mem, 2);
            link(mem, let0+2, arg0);
            link(mem, op20+0, Dp0(get_ext(arg1), let0));
            link(mem, op20+1, ask_arg(mem, arg1, 0));
            link(mem, op21+1, ask_arg(mem, arg1, 1));
            link(mem, op21+0, Dp1(get_ext(arg1), let0));
            link(mem, par0+0, Op2(get_ext(term), op20));
            link(mem, par0+1, Op2(get_ext(term), op21));
            u64 done = Par(get_ext(arg1), par0);
            link(mem, host, done);
            break;
          }}
          break;
        }}
        case CAL: {{
          u64 fun = get_ext(term);
          u64 ari = get_ari(term);

          switch (fun)
          //GENERATED_REWRITE_RULES_STEP_1_START//
          {{
{}
          }}
          //GENERATED_REWRITE_RULES_STEP_1_END//

          break;
        }}
      }}
    }}

    u64 item = stk_pop(&stack);
    if (item == -1) {{
      break;
    }} else {{
      init = item >> 31;
      host = item & 0x7FFFFFFF;
      continue;
    }}

  }}

  return ask_lnk(mem, root);
}}

// sets the nth bit of a bit-array represented as a u64 array
void set_bit(u64* bits, u64 bit) {{
  bits[bit >> 6] |= (1ULL << (bit & 0x3f));
}}

// gets the nth bit of a bit-array represented as a u64 array
u8 get_bit(u64* bits, u64 bit) {{
  return (bits[bit >> 6] >> (bit & 0x3F)) & 1;
}}

#ifdef PARALLEL
void normal_fork(u64 tid, u64 host, u64 sidx, u64 slen);
u64  normal_join(u64 tid);
#endif

#define normal_seen_mcap (16777216) // uses 128 MB, covers heaps up to 8 GB
u64 normal_seen_data[normal_seen_mcap];

void normal_init() {{
  for (u64 i = 0; i < normal_seen_mcap; ++i) {{
    normal_seen_data[i] = 0;
  }}
}}

Lnk normal_go(Worker* mem, u64 host, u64 sidx, u64 slen) {{
  Lnk term = ask_lnk(mem, host);
  //printf("normal %llu %llu | ", sidx, slen); debug_print_lnk(term); printf("\n");
  if (get_bit(normal_seen_data, host)) {{
    return term;
  }} else {{
    term = reduce(mem, host, slen);
    set_bit(normal_seen_data, host);
    u64 rec_size = 0;
    u64 rec_locs[16];
    switch (get_tag(term)) {{
      case LAM: {{
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }}
      case APP: {{
        rec_locs[rec_size++] = get_loc(term,0);
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }}
      case PAR: {{
        rec_locs[rec_size++] = get_loc(term,0);
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }}
      case DP0: {{
        rec_locs[rec_size++] = get_loc(term,2);
        break;
      }}
      case DP1: {{
        rec_locs[rec_size++] = get_loc(term,2);
        break;
      }}
      case OP2: {{
        if (slen > 1) {{
          rec_locs[rec_size++] = get_loc(term,0);
          rec_locs[rec_size++] = get_loc(term,1);
          break;
        }}
      }}
      case CTR: case CAL: {{
        u64 arity = (u64)get_ari(term);
        for (u64 i = 0; i < arity; ++i) {{
          rec_locs[rec_size++] = get_loc(term,i);
        }}
        break;
      }}
    }}
    #ifdef PARALLEL

    //printf("ue %llu %llu\n", rec_size, slen);
    
    if (rec_size >= 2 && slen >= rec_size) {{

      u64 space = slen / rec_size;

      for (u64 i = 1; i < rec_size; ++i) {{
        //printf("spawn %llu %llu\n", sidx + i * space, space);
        normal_fork(sidx + i * space, rec_locs[i], sidx + i * space, space);
      }}

      link(mem, rec_locs[0], normal_go(mem, rec_locs[0], sidx, space));

      for (u64 i = 1; i < rec_size; ++i) {{
        link(mem, get_loc(term, i), normal_join(sidx + i * space));
      }}

    }} else {{

      for (u64 i = 0; i < rec_size; ++i) {{
        link(mem, rec_locs[i], normal_go(mem, rec_locs[i], sidx, slen));
      }}

    }}
    #else

    for (u64 i = 0; i < rec_size; ++i) {{
      link(mem, rec_locs[i], normal_go(mem, rec_locs[i], sidx, slen));
    }}

    #endif

    return term;
  }}
}}

Lnk normal(Worker* mem, u64 host, u64 sidx, u64 slen) {{
  // In order to allow parallelization of numeric operations, reduce() will treat OP2 as a CTR if
  // there is enough thread space. So, for example, normalizing a recursive "sum" function with 4
  // threads might return something like `(+ (+ 64 64) (+ 64 64))`. reduce() will treat the first
  // 2 layers as CTRs, allowing normal() to parallelize them. So, in order to finish the reduction,
  // we call `normal_go()` a second time, with no thread space, to eliminate lasting redexes.
  normal_init();
  normal_go(mem, host, sidx, slen);
  normal_init();
  return normal_go(mem, host, 0, 1);
}}


#ifdef PARALLEL

// Normalizes in a separate thread
void normal_fork(u64 tid, u64 host, u64 sidx, u64 slen) {{
  pthread_mutex_lock(&workers[tid].has_work_mutex);
  workers[tid].has_work = (sidx << 48) | (slen << 32) | host;
  pthread_cond_signal(&workers[tid].has_work_signal);
  pthread_mutex_unlock(&workers[tid].has_work_mutex);
}}

// Waits the result of a forked normalizer
u64 normal_join(u64 tid) {{
  while (1) {{
    pthread_mutex_lock(&workers[tid].has_result_mutex);
    while (workers[tid].has_result == -1) {{
      pthread_cond_wait(&workers[tid].has_result_signal, &workers[tid].has_result_mutex);
    }}
    u64 done = workers[tid].has_result;
    workers[tid].has_result = -1;
    pthread_mutex_unlock(&workers[tid].has_result_mutex);
    return done;
  }}
}}

// The normalizer worker
void *worker(void *arg) {{
  u64 tid = (u64)arg;
  while (1) {{
    pthread_mutex_lock(&workers[tid].has_work_mutex);
    while (workers[tid].has_work == -1) {{
      pthread_cond_wait(&workers[tid].has_work_signal, &workers[tid].has_work_mutex);
    }}
    u64 work = workers[tid].has_work;
    u64 sidx = (work >> 48) & 0xFFFF;
    u64 slen = (work >> 32) & 0xFFFF;
    u64 host = (work >>  0) & 0xFFFFFFFF;
    workers[tid].has_result = normal_go(&workers[tid], host, sidx, slen);
    workers[tid].has_work = -1;
    pthread_cond_signal(&workers[tid].has_result_signal);
    pthread_mutex_unlock(&workers[tid].has_work_mutex);
  }}
  return 0;
}}

#endif

u64 ffi_cost;
u64 ffi_size;

void ffi_normal(u8* mem_data, u32 mem_size, u32 host) {{

  // Init thread objects
  for (u64 t = 0; t < MAX_WORKERS; ++t) {{
    workers[t].tid = t;
    workers[t].size = t == 0 ? (u64)mem_size : 0l;
    workers[t].node = (u64*)mem_data;
    for (u64 a = 0; a < MAX_ARITY; ++a) {{
      stk_init(&workers[t].free[a]);
    }}
    workers[t].cost = 0;
    #ifdef PARALLEL
    workers[t].has_work = -1;
    pthread_mutex_init(&workers[t].has_work_mutex, NULL);
    pthread_cond_init(&workers[t].has_work_signal, NULL);
    workers[t].has_result = -1;
    pthread_mutex_init(&workers[t].has_result_mutex, NULL);
    pthread_cond_init(&workers[t].has_result_signal, NULL);
    // workers[t].thread = NULL;
    #endif
  }}

  // Spawns threads
  #ifdef PARALLEL
  for (u64 tid = 1; tid < MAX_WORKERS; ++tid) {{
    pthread_create(&workers[tid].thread, NULL, &worker, (void*)tid);
  }}
  #endif

  // Normalizes trm
  normal(&workers[0], (u64) host, 0, MAX_WORKERS);
  
  // Computes total cost and size
  ffi_cost = 0;
  ffi_size = 0;
  for (u64 tid = 0; tid < MAX_WORKERS; ++tid) {{
    ffi_cost += workers[tid].cost;
    ffi_size += workers[tid].size;
  }}

  // TODO: stop pending threads

  // Clears workers
  for (u64 tid = 0; tid < MAX_WORKERS; ++tid) {{
    for (u64 a = 0; a < MAX_ARITY; ++a) {{
      stk_free(&workers[tid].free[a]);
    }}
    #ifdef PARALLEL
    pthread_mutex_destroy(&workers[tid].has_work_mutex);
    pthread_cond_destroy(&workers[tid].has_work_signal);
    pthread_mutex_destroy(&workers[tid].has_result_mutex);
    pthread_cond_destroy(&workers[tid].has_result_signal);
    #endif
  }}
}}

// Readback
// --------

void readback_vars(Stk* vars, Worker* mem, Lnk term, Stk* seen) {{
  //printf("- readback_vars %llu ", get_loc(term,0)); debug_print_lnk(term); printf("\n");
  if (stk_find(seen, term) != -1) {{ // FIXME: probably very slow, change to a proper hashmap
    return;
  }} else {{
    stk_push(seen, term);
    switch (get_tag(term)) {{
      case LAM: {{
        u64 argm = ask_arg(mem, term, 0);
        u64 body = ask_arg(mem, term, 1);
        if (get_tag(argm) != ERA) {{
          stk_push(vars, Var(get_loc(term, 0)));
        }};
        readback_vars(vars, mem, body, seen);
        break;
      }}
      case APP: {{
        u64 lam = ask_arg(mem, term, 0);
        u64 arg = ask_arg(mem, term, 1);
        readback_vars(vars, mem, lam, seen);
        readback_vars(vars, mem, arg, seen);
        break;
      }}
      case PAR: {{
        u64 arg0 = ask_arg(mem, term, 0);
        u64 arg1 = ask_arg(mem, term, 1);
        readback_vars(vars, mem, arg0, seen);
        readback_vars(vars, mem, arg1, seen);
        break;
      }}
      case DP0: {{
        u64 arg = ask_arg(mem, term, 2);
        readback_vars(vars, mem, arg, seen);
        break;
      }}
      case DP1: {{
        u64 arg = ask_arg(mem, term, 2);
        readback_vars(vars, mem, arg, seen);
        break;
      }}
      case OP2: {{
        u64 arg0 = ask_arg(mem, term, 0);
        u64 arg1 = ask_arg(mem, term, 1);
        readback_vars(vars, mem, arg0, seen);
        readback_vars(vars, mem, arg1, seen);
        break;
      }}
      case CTR: case CAL: {{
        u64 arity = get_ari(term);
        for (u64 i = 0; i < arity; ++i) {{
          readback_vars(vars, mem, ask_arg(mem, term, i), seen);
        }}
        break;
      }}
    }}
  }}
}}

void readback_decimal_go(Stk* chrs, u64 n) {{
  //printf("--- A %llu\n", n);
  if (n > 0) {{
    readback_decimal_go(chrs, n / 10);
    stk_push(chrs, '0' + (n % 10));
  }}
}}

void readback_decimal(Stk* chrs, u64 n) {{
  if (n == 0) {{
    stk_push(chrs, '0');
  }} else {{
    readback_decimal_go(chrs, n);
  }}
}}

void readback_term(Stk* chrs, Worker* mem, Lnk term, Stk* vars, Stk* dirs, char** id_to_name_data, u64 id_to_name_mcap) {{
  //printf("- readback_term: "); debug_print_lnk(term); printf("\n");
  switch (get_tag(term)) {{
    case LAM: {{
      stk_push(chrs, '%');
      if (get_tag(ask_arg(mem, term, 0)) == ERA) {{
        stk_push(chrs, '~');
      }} else {{
        stk_push(chrs, 'x');
        readback_decimal(chrs, stk_find(vars, Var(get_loc(term, 0))));
      }};
      stk_push(chrs, ' ');
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      break;
    }}
    case APP: {{
      stk_push(chrs, '(');
      readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ' ');
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ')');
      break;
    }}
    case PAR: {{
      u64 col = get_ext(term);
      if (dirs[col].size > 0) {{
        u64 head = stk_pop(&dirs[col]);
        if (head == 0) {{
          readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
          stk_push(&dirs[col], head);
        }} else {{
          readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
          stk_push(&dirs[col], head);
        }}
      }} else {{
        stk_push(chrs, '<');
        readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
        stk_push(chrs, ' ');
        readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
        stk_push(chrs, '>');
      }}
      break;
    }}
    case DP0: case DP1: {{
      u64 col = get_ext(term);
      u64 val = ask_arg(mem, term, 2);
      stk_push(&dirs[col], get_tag(term) == DP0 ? 0 : 1);
      readback_term(chrs, mem, ask_arg(mem, term, 2), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_pop(&dirs[col]);
      break;
    }}
    case OP2: {{
      stk_push(chrs, '(');
      readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
      switch (get_ext(term)) {{
        case ADD: {{ stk_push(chrs, '+'); break; }}
        case SUB: {{ stk_push(chrs, '-'); break; }}
        case MUL: {{ stk_push(chrs, '*'); break; }}
        case DIV: {{ stk_push(chrs, '/'); break; }}
        case MOD: {{ stk_push(chrs, '%'); break; }}
        case AND: {{ stk_push(chrs, '&'); break; }}
        case OR: {{ stk_push(chrs, '|'); break; }}
        case XOR: {{ stk_push(chrs, '^'); break; }}
        case SHL: {{ stk_push(chrs, '<'); stk_push(chrs, '<'); break; }}
        case SHR: {{ stk_push(chrs, '>'); stk_push(chrs, '>'); break; }}
        case LTN: {{ stk_push(chrs, '<'); break; }}
        case LTE: {{ stk_push(chrs, '<'); stk_push(chrs, '='); break; }}
        case EQL: {{ stk_push(chrs, '='); stk_push(chrs, '='); break; }}
        case GTE: {{ stk_push(chrs, '>'); stk_push(chrs, '='); break; }}
        case GTN: {{ stk_push(chrs, '>'); break; }}
        case NEQ: {{ stk_push(chrs, '!'); stk_push(chrs, '='); break; }}
      }}
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ')');
      break;
    }}
    case U32: {{
      //printf("- u32\n");
      readback_decimal(chrs, get_val(term));
      //printf("- u32 done\n");
      break;
    }}
    case CTR: case CAL: {{
      u64 func = get_ext(term);
      u64 arit = get_ari(term);
      stk_push(chrs, '(');
      if (func < id_to_name_mcap && id_to_name_data[func] != NULL) {{
        for (u64 i = 0; id_to_name_data[func][i] != '\0'; ++i) {{
          stk_push(chrs, id_to_name_data[func][i]);
        }}
      }} else {{
        stk_push(chrs, '$');
        readback_decimal(chrs, func); // TODO: function names
      }}
      for (u64 i = 0; i < arit; ++i) {{
        stk_push(chrs, ' ');
        readback_term(chrs, mem, ask_arg(mem, term, i), vars, dirs, id_to_name_data, id_to_name_mcap);
      }}
      stk_push(chrs, ')');
      break;
    }}
    case VAR: {{
      stk_push(chrs, 'x');
      readback_decimal(chrs, stk_find(vars, term));
      break;
    }}
    default: {{
      stk_push(chrs, '?');
      break;
    }}
  }}
}}

void readback(char* code_data, u64 code_mcap, Worker* mem, Lnk term, char** id_to_name_data, u64 id_to_name_mcap) {{
  //printf("reading back\n");

  // Constants
  const u64 dirs_mcap = 65536; // max different colors we're able to readback

  // Used vars
  Stk seen;
  Stk chrs;
  Stk vars;
  Stk* dirs;

  // Initialization
  stk_init(&seen);
  stk_init(&chrs);
  stk_init(&vars);
  dirs = (Stk*)malloc(sizeof(Stk) * dirs_mcap);
  for (u64 i = 0; i < dirs_mcap; ++i) {{
    stk_init(&dirs[i]);
  }}

  // Readback
  readback_vars(&vars, mem, term, &seen);
  readback_term(&chrs, mem, term, &vars, dirs, id_to_name_data, id_to_name_mcap);

  // Generates C string
  for (u64 i = 0; i < chrs.size && i < code_mcap; ++i) {{
    code_data[i] = chrs.data[i];
  }}
  code_data[chrs.size < code_mcap ? chrs.size : code_mcap] = '\0';

  // Cleanup
  stk_free(&seen);
  stk_free(&chrs);
  stk_free(&vars);
  for (u64 i = 0; i < dirs_mcap; ++i) {{
    stk_free(&dirs[i]);
  }}
}}

// Main
// ----

Lnk parse_arg(char* code, char** id_to_name_data, u64 id_to_name_size) {{
  if (code[0] >= '0' && code[0] <= '9') {{
    return U_32(strtol(code, 0, 10));
  }} else {{
    return U_32(0);
  }}
}}

// Uncomment to test without Deno FFI
int main(int argc, char* argv[]) {{

  Worker mem;
  struct timeval stop, start;

  // Id-to-Name map
  const u64 id_to_name_size = {};
  char* id_to_name_data[id_to_name_size];
{}

  // Builds main term
  mem.size = 0;
  mem.node = (u64*)malloc(8 * 134217728 * sizeof(u64)); // 8gb
  if (argc <= 1) {{
    mem.node[mem.size++] = Cal(0, _MAIN_, 0);
  }} else {{
    mem.node[mem.size++] = Cal(argc - 1, _MAIN_, 1);
    for (u64 i = 1; i < argc; ++i) {{
      mem.node[mem.size++] = parse_arg(argv[i], id_to_name_data, id_to_name_size);
    }}
  }}

  // Reduces and benchmarks
  //printf("Reducing.\n");
  gettimeofday(&start, NULL);
  ffi_normal((u8*)mem.node, mem.size, 0);
  gettimeofday(&stop, NULL);

  // Prints result statistics
  u64 delta_time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
  double rwt_per_sec = (double)ffi_cost / (double)delta_time;
  printf("Rewrites: %llu (%.2f MR/s).\n", ffi_cost, rwt_per_sec);
  printf("Mem.Size: %llu words.\n", ffi_size);
  printf("\n");

  // Prints result normal form
  const u64 code_mcap = 256 * 256 * 256; // max code size = 16 MB
  char* code_data = (char*)malloc(code_mcap * sizeof(char)); 
  readback(code_data, code_mcap, &mem, mem.node[0], id_to_name_data, id_to_name_size);
  printf("%s\n", code_data);

  // Cleanup
  free(code_data);
  free(mem.node);
}}

  "#,
    c_ids, inits, codes, names_count, id2nm
  );
}
