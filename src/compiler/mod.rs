#![allow(unreachable_code)]
#![allow(clippy::identity_op)]

use std::{io, fs, path::{Path, PathBuf}};

mod compile;

fn copy(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
  let (from, to) = (from.as_ref().to_owned(), to.as_ref().to_owned());
  if fs::metadata(from.clone())?.is_dir() {
    let (mut stack_from, mut stack_to) = (vec![from], vec![to]);
    while let Some(from) = stack_from.pop() {
      let to = stack_to.pop().unwrap();
      fs::create_dir_all(&to)?;
      for entry in fs::read_dir(from)? {
        let entry = entry?;
        let filetype = entry.file_type()?;
        if filetype.is_dir() {
          stack_from.push(entry.path());
          stack_to.push(<PathBuf as AsRef<Path>>::as_ref(&to).join(entry.file_name()));
        } else {
          fs::copy(entry.path(), <PathBuf as AsRef<Path>>::as_ref(&to).join(entry.file_name()))?;
        }
      }
    }
  } else {
    fs::copy(from, to)?;
  }
  Ok(())
}

pub fn compile(code: &str, name: &str) -> std::io::Result<()> {
  let output = &format!("./{name}");
  fs::create_dir_all(format!("{output}/src/runtime/base")).ok();
  let cargo_rs = include_str!("./../../Cargo.toml")
    .replace("name = \"hvm\"", &format!("name = \"{name}\""))
    .replace("name = \"hvm\"", &format!("name = \"{name}\""));
  fs::write(format!("{output}/Cargo.toml"), cargo_rs)?;
  fs::write(format!("{output}/rust-toolchain.toml"), include_str!("./../../rust-toolchain.toml"))?;
  copy("./src/language", &format!("{output}/src/language"))?;
  copy("./src/runtime", &format!("{output}/src/runtime"))?;
  copy("./src/api.rs", &format!("{output}/src/api.rs"))?;
  copy("./src/cli.rs", &format!("{output}/src/cli.rs"))?;
  copy("./src/compiled-cli.rs", (&format!("{output}/src/main.rs")))?;
  copy("./src/lib.rs", &format!("{output}/src/lib.rs"))?;
  let (precomp_rs, reducer_rs) = compile::build_code(code).unwrap();
  fs::write(format!("{output}/src/runtime/base/precomp.rs"), precomp_rs)?;
  fs::write(format!("{output}/src/runtime/base/reducer.rs"), reducer_rs)?;
  Ok(())
}