use crate::language;
use crate::runtime;

// Evaluates a HVM term to normal form
pub fn eval(
  file: &str,
  term: &str,
  funs: Vec<(String, runtime::Function)>,
  size: usize,
  tids: usize,
  dbug: bool,
) -> Result<(String, u64, u64), String> {
  // Parses and reads the input file
  let file = language::syntax::read_file(&format!("{}\nHVM_MAIN_CALL = {}", file, term))?;

  // Converts the file to a Rulebook
  let book = (&file).into();

  // Creates the runtime program
  let mut prog = runtime::Program::new();

  let begin = instant::Instant::now();

  // Adds the interpreted functions (from the Rulebook)
  prog.add_book(&book);

  // Adds the extra functions
  for (name, fun) in funs {
    prog.add_function(name, fun);
  }

  // Creates the runtime heap
  let heap = crate::runtime::Heap::new(size, tids);
  let tids = runtime::new_tids(tids);

  // Allocates the main term
  heap.link(0, runtime::Fun(*book.name_to_id.get("HVM_MAIN_CALL").unwrap(), 0));
  let host = 0;

  // Normalizes it
  let init = instant::Instant::now();
  runtime::normalize(&heap, &prog, &tids, host, dbug);
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a string
  let code = format!("{}", language::readback::as_term(&heap, &prog, host));

  // Frees used memory
  heap.collect(&prog.aris, tids[0], heap.load_ptr(host));
  heap.free(0, 0, 1);

  // Returns the result, rewrite cost and time elapsed
  Ok((code, heap.get_cost(), time))
}
