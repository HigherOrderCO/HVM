#![allow(unreachable_code)]
#![allow(clippy::identity_op)]

mod compile;

pub fn compile(code: &str, name: &str) -> std::io::Result<()> {
  let cargo_rs = include_str!("./../../Cargo.toml")
    .replace("name = \"hvm\"", &format!("name = \"{name}\""))
    .replace("name = \"hvm\"", &format!("name = \"{name}\""));

  // hvm
  std::fs::create_dir(format!("./{name}")).ok();
  std::fs::write(format!("./{name}/Cargo.toml"), cargo_rs)?;
  std::fs::write(
    format!("./{name}/rust-toolchain.toml"),
    include_str!("./../../rust-toolchain.toml"),
  )?;

  // hvm/src
  std::fs::create_dir(format!("./{name}/src")).ok();
  std::fs::write(format!("./{name}/src/main.rs"), include_str!("./../main.rs"))?;
  std::fs::write(format!("./{name}/src/lib.rs"), include_str!("./../lib.rs"))?;
  std::fs::write(format!("./{name}/src/api.rs"), include_str!("./../api.rs"))?;

  // hvm/src/compiler
  std::fs::create_dir(format!("./{name}/src/compiler")).ok();
  std::fs::write(format!("./{name}/src/compiler/mod.rs"), include_str!("./../compiler/mod.rs"))?;
  std::fs::write(
    format!("./{name}/src/compiler/compile.rs"),
    include_str!("./../compiler/compile.rs"),
  )?;

  // hvm/src/language
  std::fs::create_dir(format!("./{name}/src/language")).ok();
  std::fs::write(format!("./{name}/src/language/mod.rs"), include_str!("./../language/mod.rs"))?;
  std::fs::write(
    format!("./{name}/src/language/parser.rs"),
    include_str!("./../language/parser.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/language/readback.rs"),
    include_str!("./../language/readback.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/language/rulebook.rs"),
    include_str!("./../language/rulebook.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/language/syntax.rs"),
    include_str!("./../language/syntax.rs"),
  )?;

  // hvm/src/runtime
  std::fs::create_dir(format!("./{name}/src/runtime")).ok();
  std::fs::write(format!("./{name}/src/runtime/mod.rs"), include_str!("./../runtime/mod.rs"))?;

  // hvm/src/runtime/base
  let (precomp_rs, reducer_rs) = compile::build_code(code).unwrap();
  std::fs::create_dir(format!("./{name}/src/runtime/base")).ok();
  std::fs::write(
    format!("./{name}/src/runtime/base/mod.rs"),
    include_str!("./../runtime/base/mod.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/base/debug.rs"),
    include_str!("./../runtime/base/debug.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/base/memory.rs"),
    include_str!("./../runtime/base/memory.rs"),
  )?;
  std::fs::write(format!("./{name}/src/runtime/base/precomp.rs"), precomp_rs)?;
  std::fs::write(
    format!("./{name}/src/runtime/base/program.rs"),
    include_str!("./../runtime/base/program.rs"),
  )?;
  std::fs::write(format!("./{name}/src/runtime/base/reducer.rs"), reducer_rs)?;

  // hvm/src/runtime/data
  std::fs::create_dir(format!("./{name}/src/runtime/data")).ok();
  std::fs::write(
    format!("./{name}/src/runtime/data/mod.rs"),
    include_str!("./../runtime/data/mod.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/f60.rs"),
    include_str!("./../runtime/data/f60.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/allocator.rs"),
    include_str!("./../runtime/data/allocator.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/barrier.rs"),
    include_str!("./../runtime/data/barrier.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/redex_bag.rs"),
    include_str!("./../runtime/data/redex_bag.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/u60.rs"),
    include_str!("./../runtime/data/u60.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/u64_map.rs"),
    include_str!("./../runtime/data/u64_map.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/data/visit_queue.rs"),
    include_str!("./../runtime/data/visit_queue.rs"),
  )?;

  // hvm/src/runtime/rule
  std::fs::create_dir(format!("./{name}/src/runtime/rule")).ok();
  std::fs::write(
    format!("./{name}/src/runtime/rule/mod.rs"),
    include_str!("./../runtime/rule/mod.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/rule/app.rs"),
    include_str!("./../runtime/rule/app.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/rule/dup.rs"),
    include_str!("./../runtime/rule/dup.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/rule/fun.rs"),
    include_str!("./../runtime/rule/fun.rs"),
  )?;
  std::fs::write(
    format!("./{name}/src/runtime/rule/op2.rs"),
    include_str!("./../runtime/rule/op2.rs"),
  )?;

  Ok(())
}
