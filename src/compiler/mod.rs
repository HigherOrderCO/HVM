#![allow(unreachable_code)]
#![allow(clippy::identity_op)]

mod compile;

pub fn compile(code: &str, name: &str) -> std::io::Result<()> {
  let cargo_rs = include_str!("./../../Cargo.toml")
    .replace("name = \"hvm\"", &format!("name = \"{}\"", name))
    .replace("name = \"hvm\"", &format!("name = \"{}\"", name));

  // hvm
  std::fs::create_dir(format!("./{}", name)).ok();
  std::fs::write(format!("./{}/Cargo.toml", name), cargo_rs)?;
  std::fs::write(
    format!("./{}/rust-toolchain.toml", name),
    include_str!("./../../rust-toolchain.toml"),
  )?;

  // hvm/src
  std::fs::create_dir(format!("./{}/src", name)).ok();
  std::fs::write(format!("./{}/src/main.rs", name), include_str!("./../main.rs"))?;
  std::fs::write(format!("./{}/src/lib.rs", name), include_str!("./../lib.rs"))?;
  std::fs::write(format!("./{}/src/api.rs", name), include_str!("./../api.rs"))?;

  // hvm/src/compiler
  std::fs::create_dir(format!("./{}/src/compiler", name)).ok();
  std::fs::write(format!("./{}/src/compiler/mod.rs", name), include_str!("./../compiler/mod.rs"))?;
  std::fs::write(
    format!("./{}/src/compiler/compile.rs", name),
    include_str!("./../compiler/compile.rs"),
  )?;

  // hvm/src/language
  std::fs::create_dir(format!("./{}/src/language", name)).ok();
  std::fs::write(format!("./{}/src/language/mod.rs", name), include_str!("./../language/mod.rs"))?;
  std::fs::write(
    format!("./{}/src/language/readback.rs", name),
    include_str!("./../language/readback.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/language/rulebook.rs", name),
    include_str!("./../language/rulebook.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/language/syntax.rs", name),
    include_str!("./../language/syntax.rs"),
  )?;

  // hvm/src/runtime
  std::fs::create_dir(format!("./{}/src/runtime", name)).ok();
  std::fs::write(format!("./{}/src/runtime/mod.rs", name), include_str!("./../runtime/mod.rs"))?;

  // hvm/src/runtime/base
  let (precomp_rs, reducer_rs) = compile::build_code(code).unwrap();
  std::fs::create_dir(format!("./{}/src/runtime/base", name)).ok();
  std::fs::write(
    format!("./{}/src/runtime/base/mod.rs", name),
    include_str!("./../runtime/base/mod.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/base/debug.rs", name),
    include_str!("./../runtime/base/debug.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/base/memory.rs", name),
    include_str!("./../runtime/base/memory.rs"),
  )?;
  std::fs::write(format!("./{}/src/runtime/base/precomp.rs", name), precomp_rs)?;
  std::fs::write(
    format!("./{}/src/runtime/base/program.rs", name),
    include_str!("./../runtime/base/program.rs"),
  )?;
  std::fs::write(format!("./{}/src/runtime/base/reducer.rs", name), reducer_rs)?;

  // hvm/src/runtime/data
  std::fs::create_dir(format!("./{}/src/runtime/data", name)).ok();
  std::fs::write(
    format!("./{}/src/runtime/data/mod.rs", name),
    include_str!("./../runtime/data/mod.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/f60.rs", name),
    include_str!("./../runtime/data/f60.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/allocator.rs", name),
    include_str!("./../runtime/data/allocator.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/barrier.rs", name),
    include_str!("./../runtime/data/barrier.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/redex_bag.rs", name),
    include_str!("./../runtime/data/redex_bag.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/u60.rs", name),
    include_str!("./../runtime/data/u60.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/u64_map.rs", name),
    include_str!("./../runtime/data/u64_map.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/data/visit_queue.rs", name),
    include_str!("./../runtime/data/visit_queue.rs"),
  )?;

  // hvm/src/runtime/rule
  std::fs::create_dir(format!("./{}/src/runtime/rule", name)).ok();
  std::fs::write(
    format!("./{}/src/runtime/rule/mod.rs", name),
    include_str!("./../runtime/rule/mod.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/rule/app.rs", name),
    include_str!("./../runtime/rule/app.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/rule/dup.rs", name),
    include_str!("./../runtime/rule/dup.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/rule/fun.rs", name),
    include_str!("./../runtime/rule/fun.rs"),
  )?;
  std::fs::write(
    format!("./{}/src/runtime/rule/op2.rs", name),
    include_str!("./../runtime/rule/op2.rs"),
  )?;

  return Ok(());
}
