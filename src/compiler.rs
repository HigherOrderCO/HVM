// re-implementing from:
// https://github.com/Kindelia/LambdaVM/blob/new_v2/src/Compile/Compile.ts

use askama::Template;
use ropey::{Rope, RopeBuilder};

use crate::compilable as cplb;
use crate::lambolt as lb;
use crate::runtime as rt;

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
    CodeBuilder {
      rope_builder: RopeBuilder::default(),
    }
  }

  pub fn append(&mut self, chunk: &str) {
    self.rope_builder.append(chunk);
  }

  pub fn finish(self) -> Rope {
    self.rope_builder.finish()
  }
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

pub fn compile(compilable: &cplb::Compilable, target: Target, mode: Mode) -> String {
  let use_dynamic_flag = emit_use_dynamic(target, matches!(mode, Mode::Dynamic));
  let use_static_flag = emit_use_static(target, matches!(mode, Mode::Static));

  // TODO: constructor_ids

  // TODO: rewrite_rules_step_0

  // TODO: rewrite_rules_step_1

  let template = CodeTemplate {
    use_dynamic_flag,
    use_static_flag,
    constructor_ids: "",
    rewrite_rules_step_0: "",
    rewrite_rules_step_1: "",
  };
  template.render().unwrap()
}

fn compile_constructor_name(name: &str) -> String {
  // TODO:  replace(/\./g,"$")
  format!("${}", name.to_uppercase())
}

fn compile_group_step_0(meta: &cplb::Compilable) -> Rope {
  todo!() // TODO
}

fn compile_group_step_1(meta: &cplb::Compilable) -> Rope {
  todo!() // TODO
}

// Creates a new line with an amount of tabs.
fn line(idt: u32, text: &str) -> Rope {
  // for (var i = 0; i < tab; ++i) {
  //   text = "  " + text;
  // }
  // return text + "\n";
  todo!() // TODO
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
  use super::{compile, Mode, Target};
  use crate::compilable;
  use crate::lambolt;

  #[test]
  fn test() {
    let code = "(Main) = ((λf λx (f (f x))) (λf λx (f (f x))))";
    let file = lambolt::read_file(code);
    let comp = compilable::gen_compilable(&file);
    let result = super::compile(&comp, Target::C, Mode::Static);
    // println!("{}", result);
  }
}
