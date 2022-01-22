// re-implementing from:
// https://github.com/Kindelia/LambdaVM/blob/new_v2/src/Compile/Compile.ts

// Note: in a future, we should compile from a `Vec<DynFun>` to a `Vec<Function>`. This will
// greatly decrease the size of this work, since most of the logic here is to shared by the
// `build_dynfun` function on `dynfun.rs`. The JIT compiler should also start from `DynFun`.
// So, the ideal compilation pipeline would be:
// LamboltCode -> LamboltFile -> RuleBook -> Vec<DynFun> -> Vec<RuntimeFunction> (interpreted)
//                                                       -> Vec<RuntimeFunction> (JIT-compiled)
//                                                       -> CLangFile            (compiled)

use std::collections::{HashMap, HashSet};
use std::fmt;

use askama::Template;
use ropey::{Rope, RopeBuilder};

use crate::lambolt as lb;
use crate::rulebook as rb;
use crate::runtime as rt;

const INDENT: &str = "  ";
const LINE_BREAK: &str = "\n";

#[derive(Template)]
#[template(path = "runtime.c", escape = "none", syntax = "c")]
struct CodeTemplate<'a> {
  use_dynamic_flag: &'a str,
  use_static_flag: &'a str,
  constructor_ids: &'a str,
  rewrite_rules_step_0: &'a str,
  rewrite_rules_step_1: &'a str,
}

#[derive(Default)]
struct CodeBuilder {
  rope_builder: RopeBuilder,
}

impl CodeBuilder {
  pub fn new() -> Self {
    CodeBuilder { rope_builder: RopeBuilder::new() }
  }

  // Append a string
  pub fn add(&mut self, chunk: &str) -> &mut Self {
    self.rope_builder.append(chunk);
    self
  }

  // Append given number of indentation
  pub fn idt(&mut self, idt: u32) -> &mut Self {
    for _ in 0..idt {
      self.add(INDENT);
    }
    self
  }

  // Append line break
  pub fn ln(&mut self) -> &mut Self {
    self.add(LINE_BREAK)
  }

  // pub fn nope(&self) {}

  pub fn finish(self) -> Rope {
    self.rope_builder.finish()
  }
}

impl fmt::Write for CodeBuilder {
  fn write_str(&mut self, txt: &str) -> fmt::Result {
    self.add(txt);
    Ok(())
  }
}

#[macro_export]
macro_rules! line {
  ($b:expr, $idt:expr, $($arg:tt)*) => {{
    use std::fmt::Write;
    $b.idt($idt);
    $b.write_fmt(format_args!($($arg)*)).unwrap();
    $b.ln();
  }}
}

#[derive(Clone, Copy, Debug)]
pub enum Target {
  C,
}

#[derive(Clone, Copy, Debug)]
pub enum Mode {
  Dynamic,
  Static,
}

pub fn compile(rulebook: &rb::RuleBook, target: Target, mode: Mode) -> String {
  let use_dynamic_flag = emit_use_dynamic(target, matches!(mode, Mode::Dynamic));
  let use_static_flag = emit_use_static(target, matches!(mode, Mode::Static));

  let constructor_ids = emit_constructor_ids(target, &rulebook.name_to_id);
  let rewrite_rules_step_0 = emit_group_step_0(target, rulebook);
  let rewrite_rules_step_1 = emit_group_step_1(target, rulebook);

  let constructor_ids = &constructor_ids.to_string();
  let rewrite_rules_step_0 = &rewrite_rules_step_0.to_string();
  let rewrite_rules_step_1 = &rewrite_rules_step_1.to_string();

  let template = CodeTemplate {
    use_dynamic_flag,
    use_static_flag,
    constructor_ids,
    rewrite_rules_step_0,
    rewrite_rules_step_1,
  };
  template.render().unwrap()
}

fn emit_constructor_name(name: &str) -> String {
  format!("${}", name.to_uppercase().replace(".", "$"))
}

fn emit_constructor_ids(target: Target, name_to_id: &HashMap<String, u64>) -> Rope {
  let mut builder = CodeBuilder::new();

  for (name, id) in name_to_id.iter() {
    builder.add(emit_const(target));
    builder.add(" ");
    builder.add(&emit_constructor_name(name));
    builder.add(" = ");
    builder.add(&emit_u64(target, *id));
    builder.add(";\n");
  }

  builder.finish()
}

fn emit_group_step_0(target: Target, comp: &rb::RuleBook) -> Rope {
  let mut builder = CodeBuilder::new();

  let base_idt = 6;

  for (name, rules_info) in comp.func_rules.iter() {
    let name = &emit_constructor_name(name);
    builder.idt(base_idt).add("case ").add(name).add(": {").ln();

    // let mut reduce_at: HashSet<usize> = HashSet::new();
    // let mut stricts: Vec<usize> = Vec::new();
    let mut to_reduce: Vec<usize> = Vec::new();

    let arity = rules_info.0;
    let rules = &rules_info.1;
    for rule in rules {
      if let lb::Term::Ctr { ref name, ref args } = *rule.lhs {
        for (i, arg) in args.iter().enumerate() {
          match &**arg {
            lb::Term::Ctr { .. } | lb::Term::U32 { .. } => {
              to_reduce.push(i);
            }
            default => {}
          }
        }
      }
    }

    if to_reduce.is_empty() {
      builder.idt(base_idt + 1).add("init = 0;").ln();
      builder.idt(base_idt + 1).add("continue;").ln();
    } else {
      builder.idt(base_idt + 1).add("stk_push(&stack, host);").ln();
      for (i, pos) in to_reduce.iter().enumerate() {
        let pos = &pos.to_string();
        if i < to_reduce.len() - 1 {
          builder
            .idt(base_idt + 1)
            .add("stk_push(&stack, get_loc(term, ")
            .add(pos)
            .add(") | 0x80000000);")
            .ln();
        } else {
          builder.idt(base_idt + 1).add("host = get_loc(term, ").add(pos).add(");").ln();
        }
      }
      builder.idt(base_idt + 1).add("continue;").ln();
    }

    builder.idt(base_idt).add("}").ln();
  }

  builder.finish()
}

struct Step1Ctx {
  // locs: {[name: string]: string} = {};
// args: {[name: string]: string} = {};
// uses: {[name: string]: number} = {};
// dups = 0;
// text = "";
// size = 0;
}

fn emit_group_step_1(target: Target, comp: &rb::RuleBook) -> Rope {
  let mut bd = CodeBuilder::new();
  let ctx = Step1Ctx {};

  let idt = 6;

  for (name, rules) in comp.func_rules.iter() {
    let name = &emit_constructor_name(name);
    line!(bd, idt, "case {}: {{", name);
  }
  bd.finish()
}

fn emit_const(target: Target) -> &'static str {
  match target {
    Target::C => "const u64",
  }
}

fn emit_var(target: Target) -> &'static str {
  match target {
    Target::C => "u64",
  }
}

fn emit_u64(target: Target, num: u64) -> String {
  match target {
    Target::C => format!("{}", num),
  }
}

fn emit_gas(target: Target) -> &'static str {
  match target {
    Target::C => "inc_cost(mem)",
  }
}

fn emit_use_dynamic(target: Target, use_dynamic: bool) -> &'static str {
  match target {
    Target::C => {
      if use_dynamic {
        "#define USE_DYNAMIC"
      } else {
        "#undef USE_DYNAMIC"
      }
    }
  }
}

fn emit_use_static(target: Target, use_static: bool) -> &'static str {
  match target {
    Target::C => {
      if use_static {
        "#define USE_STATIC"
      } else {
        "#undef USE_STATIC"
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lambolt;
  use crate::rulebook;

  #[test]
  fn test_emit_constructor_name() {
    let original = "Some.Test.Term";
    let emitted = emit_constructor_name(original);
    assert_eq!(emitted, "$SOME$TEST$TERM");
  }

  #[test]
  fn test() {
    // let code = "(Main) = ((λf λx (f (f x))) (λf λx (f (f x))))";
    let code = "
      (Double (Succ pred)) = (Succ (Succ (Double pred)))
      (Double (Zero))      = (Zero)
      (Foo a b)            = λc λd (Ue a b c d)
      (Main)               = (Double (Zero))
    ";
    let file = lambolt::read_file(code);
    let comp = rulebook::gen_rulebook(&file);
    let result = super::compile(&comp, Target::C, Mode::Static);
    println!("{}", result);
  }
}
