use std::{
  collections::HashMap,
  error::Error,
  ffi::OsStr,
  fs,
  io::{Read, Write},
  path::{Path, PathBuf},
  process::{Command, Stdio},
};

use hvm::ast::Tree;
use insta::assert_snapshot;
use TSPL::Parser;

#[test]
fn test_run_programs() {
  test_dir(&manifest_relative("tests/programs/"));
}

#[test]
fn test_run_examples() {
  test_dir(&manifest_relative("examples/"));
}

fn test_dir(dir: &Path) {
  insta::glob!(dir, "**/*.hvm", test_file)
}

fn manifest_relative(sub: &str) -> PathBuf {
  format!("{}/{}", env!("CARGO_MANIFEST_DIR"), sub).into()
}

fn test_file(path: &Path) {
  let contents = fs::read_to_string(path).unwrap();
  if contents.contains("@test-skip = 1") {
    println!("skipping {path:?}");
    return;
  }
  if contents.contains("@test-io = 1") {
    test_io_file(path);
    return;
  }

  println!("testing {path:?}...");
  let rust_output = execute_hvm(&["run".as_ref(), path.as_os_str()], false).unwrap();
  assert_snapshot!(rust_output);

  println!("  testing {path:?}, C...");
  let c_output = execute_hvm(&["run-c".as_ref(), path.as_os_str()], false).unwrap();
  assert_eq!(c_output, rust_output, "{path:?}: C output does not match rust output");

  if cfg!(feature = "cuda") {
    println!("  testing {path:?}, CUDA...");
    let cuda_output = execute_hvm(&["run-cu".as_ref(), path.as_os_str()], false).unwrap();
    assert_eq!(
      cuda_output, rust_output,
      "{path:?}: CUDA output does not match rust output"
    );
  }
}

fn test_io_file(path: &Path) {
  println!("  testing (io) {path:?}, C...");
  let c_output = execute_hvm(&["run-c".as_ref(), path.as_os_str()], true).unwrap();
  assert_snapshot!(c_output);

  if cfg!(feature = "cuda") {
    println!("  testing (io) {path:?}, CUDA...");
    let cuda_output = execute_hvm(&["run-cu".as_ref(), path.as_os_str()], true).unwrap();
    assert_eq!(cuda_output, c_output, "{path:?}: CUDA output does not match C output");
  }
}

fn execute_hvm(args: &[&OsStr], send_io: bool) -> Result<String, Box<dyn Error>> {
  // Spawn the command
  let mut child = Command::new(env!("CARGO_BIN_EXE_hvm"))
    .args(args)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()?;

  // Capture the output of the command
  let mut stdout = child.stdout.take().ok_or("Couldn't capture stdout!")?;
  let mut stderr = child.stderr.take().ok_or("Couldn't capture stderr!")?;

  // Wait for the command to finish and get the exit status
  if send_io {
    let mut stdin = child.stdin.take().ok_or("Couldn't capture stdin!")?;
    stdin.write_all(b"io from the tests\n")?;
    drop(stdin);
  }
  let status = child.wait()?;

  // Read the output
  let mut output = String::new();
  stdout.read_to_string(&mut output)?;
  stderr.read_to_string(&mut output)?;

  Ok(if !status.success() {
    format!("exited with code {status}:\n{output}")
  } else {
    parse_output(&output).unwrap_or_else(|err| panic!("error parsing output:\n{err}\n\n{output}"))
  })
}

fn parse_output(output: &str) -> Result<String, String> {
  let mut lines = Vec::new();

  for line in output.lines() {
    if line.starts_with("Result:") {
      let mut parser = hvm::ast::CoreParser::new(line);
      parser.consume("Result:")?;
      let mut tree = parser.parse_tree()?;
      normalize_vars(&mut tree, &mut HashMap::new());
      lines.push(format!("Result: {}", tree.show()));
    } else if !line.starts_with("- ITRS:") && !line.starts_with("- TIME:") && !line.starts_with("- MIPS:") {
      // TODO: include iteration count in snapshot once consistent
      lines.push(line.to_string())
    }
  }

  Ok(lines.join("\n"))
}

fn normalize_vars(tree: &mut Tree, vars: &mut HashMap<String, usize>) {
  match tree {
    Tree::Var { nam } => {
      let next_var = vars.len();
      *nam = format!("x{}", vars.entry(std::mem::take(nam)).or_insert(next_var));
    }
    Tree::Era | Tree::Ref { .. } | Tree::Num { .. } => {}
    Tree::Con { fst, snd } | Tree::Dup { fst, snd } | Tree::Opr { fst, snd } | Tree::Swi { fst, snd } => {
      normalize_vars(fst, vars);
      normalize_vars(snd, vars);
    }
  }
}
