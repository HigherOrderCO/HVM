//#[cfg(not(target_arch = "wasm32"))]
//pub fn run_io(heap: &Heap, prog: &Program, tids: &[usize], host: u64) {
  //fn read_input() -> String {
    //let mut input = String::new();
    //stdin().read_line(&mut input).expect("string");
    //if let Some('\n') = input.chars().next_back() { input.pop(); }
    //if let Some('\r') = input.chars().next_back() { input.pop(); }
    //return input;
  //}
  //use std::io::{stdin,stdout,Write};
  //loop {
    //let term = reduce(heap, prog, tids, host); // FIXME: add parallelism
    //match get_tag(term) {
      //CTR => {
        //let fid = get_ext(term);
        //// IO.done a : (IO a)
        //if fid == IO_DONE {
          //let done = load_arg(heap, term, 0);
          //free(heap, 0, get_loc(term, 0), 1);
          //link(heap, host, done);
          //println!("");
          //println!("");
          //break;
        //}
        //// IO.do_input (String -> IO a) : (IO a)
        //if fid == IO_DO_INPUT {
          //let cont = load_arg(heap, term, 0);
          //let text = make_string(heap, tids[0], &read_input());
          //let app0 = alloc(heap, tids[0], 2);
          //link(heap, app0 + 0, cont);
          //link(heap, app0 + 1, text);
          //free(heap, 0, get_loc(term, 0), 1);
          //let done = App(app0);
          //link(heap, host, done);
        //}
        //// IO.do_output String (Wrd -> IO a) : (IO a)
        //if fid == IO_DO_OUTPUT {
          //if let Some(show) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            //print!("{}", show);
            //stdout().flush().ok();
            //let cont = load_arg(heap, term, 1);
            //let app0 = alloc(heap, tids[0], 2);
            //link(heap, app0 + 0, cont);
            //link(heap, app0 + 1, Wrd(0));
            //free(heap, 0, get_loc(term, 0), 2);
            //let text = load_arg(heap, term, 0);
            //collect(heap, prog, 0, text);
            //let done = App(app0);
            //link(heap, host, done);
          //} else {
            //println!("Runtime type error: attempted to print a non-string.");
            //println!("{}", crate::language::readback::as_code(heap, prog, get_loc(term, 0)));
            //std::process::exit(0);
          //}
        //}
        //// IO.do_fetch String (String -> IO a) : (IO a)
        //if fid == IO_DO_FETCH {
          //if let Some(url) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            //let body = reqwest::blocking::get(url).unwrap().text().unwrap(); // FIXME: treat
            //let cont = load_arg(heap, term, 2);
            //let app0 = alloc(heap, tids[0], 2);
            //let text = make_string(heap, tids[0], &body);
            //link(heap, app0 + 0, cont);
            //link(heap, app0 + 1, text);
            //free(heap, 0, get_loc(term, 0), 3);
            //let opts = load_arg(heap, term, 1); // FIXME: use options
            //collect(heap, prog, 0, opts);
            //let done = App(app0);
            //link(heap, host, done);
          //} else {
            //println!("Runtime type error: attempted to print a non-string.");
            //println!("{}", crate::language::readback::as_code(heap, prog, get_loc(term, 0)));
            //std::process::exit(0);
          //}
        //}
        //// IO.do_store String String (Wrd -> IO a) : (IO a)
        //if fid == IO_DO_STORE {
          //if let Some(key) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            //if let Some(val) = readback_string(heap, prog, tids, get_loc(term, 1)) {
              //std::fs::write(key, val).ok(); // TODO: Handle errors
              //let cont = load_arg(heap, term, 2);
              //let app0 = alloc(heap, tids[0], 2);
              //link(heap, app0 + 0, cont);
              //link(heap, app0 + 1, Wrd(0));
              //free(heap, 0, get_loc(term, 0), 2);
              //let key = load_arg(heap, term, 0);
              //collect(heap, prog, 0, key);
              //free(heap, 0, get_loc(term, 1), 2);
              //let val = load_arg(heap, term, 1);
              //collect(heap, prog, 0, val);
              //let done = App(app0);
              //link(heap, host, done);
            //} else {
              //println!("Runtime type error: attempted to store a non-string.");
              //println!("{}", crate::language::readback::as_code(heap, prog, get_loc(term, 1)));
              //std::process::exit(0);
            //}
          //} else {
            //println!("Runtime type error: attempted to store to a non-string key.");
            //println!("{}", crate::language::readback::as_code(heap, prog, get_loc(term, 0)));
            //std::process::exit(0);
          //}
        //}
        //// IO.do_load String (String -> IO a) : (IO a)
        //if fid == IO_DO_LOAD {
          //if let Some(key) = readback_string(heap, prog, tids, get_loc(term, 0)) {
            //let file = std::fs::read(key).unwrap(); // TODO: Handle errors
            //let file = std::str::from_utf8(&file).unwrap();
            //let cont = load_arg(heap, term, 1); 
            //let text = make_string(heap, tids[0], file);
            //let app0 = alloc(heap, tids[0], 2);
            //link(heap, app0 + 0, cont);
            //link(heap, app0 + 1, text);
            //free(heap, 0, get_loc(term, 0), 2);
            //let done = App(app0);
            //link(heap, host, done);
          //} else {
            //println!("Runtime type error: attempted to read from a non-string key.");
            //println!("{}", crate::language::readback::as_code(heap, prog, get_loc(term, 0)));
            //std::process::exit(0);
          //}
        //}
        //break;
      //}
      //_ => {
        //break;
      //}
    //}
  //}
//}

//pub fn make_string(heap: &Heap, tid: usize, text: &str) -> Ptr {
  //let mut term = Ctr(STRING_NIL, 0);
  //for chr in text.chars().rev() { // TODO: reverse
    //let ctr0 = alloc(heap, tid, 2);
    //link(heap, ctr0 + 0, Wrd(chr as u64));
    //link(heap, ctr0 + 1, term);
    //term = Ctr(STRING_CONS, ctr0);
  //}
  //return term;
//}

//// TODO: finish this
//pub fn readback_string(heap: &Heap, prog: &Program, tids: &[usize], host: u64) -> Option<String> {
  //let mut host = host;
  //let mut text = String::new();
  //loop {
    //let term = reduce(heap, prog, tids, host);
    //if get_tag(term) == CTR {
      //let fid = get_ext(term);
      //if fid == STRING_NIL {
        //break;
      //}
      //if fid == STRING_CONS {
        //let chr = reduce(heap, prog, tids, get_loc(term, 0));
        //if get_tag(chr) == NUM {
          //text.push(std::char::from_u32(get_num(chr) as u32).unwrap_or('?'));
          //host = get_loc(term, 1);
          //continue;
        //} else {
          //return None;
        //}
      //}
      //return None;
    //} else {
      //return None;
    //}
  //}
  //return Some(text);
//}
