use crate::runtime::{*};
use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

// Debug
// -----

pub fn show_ptr(x: Ptr) -> String {
  if x == 0 {
    String::from("~")
  } else {
    let tag = get_tag(x);
    let ext = get_ext(x);
    let val = get_val(x);
    let tgs = match tag {
      DP0 => "Dp0",
      DP1 => "Dp1",
      VAR => "Var",
      ARG => "Arg",
      ERA => "Era",
      LAM => "Lam",
      APP => "App",
      SUP => "Sup",
      CTR => "Ctr",
      FUN => "Fun",
      OP2 => "Op2",
      U60 => "Data.U60",
      F60 => "F60",
      _   => "?",
    };
    format!("{}({:07x}, {:08x})", tgs, ext, val)
  }
}

pub fn show_heap(heap: &Heap) -> String {
  let mut text: String = String::new();
  for idx in 0 .. heap.node.len() {
    let ptr = load_ptr(heap, idx as u64);
    if ptr != 0 {
      text.push_str(&format!("{:04x} | ", idx));
      text.push_str(&show_ptr(ptr));
      text.push('\n');
    }
  }
  text
}

pub fn show_at(heap: &Heap, prog: &Program, host: u64, tlocs: &[AtomicU64]) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    heap: &Heap,
    prog: &Program,
    host: u64,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    let term = load_ptr(heap, host);
    if term == 0 {
      return;
    }
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      SUP => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, get_loc(term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, get_loc(term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      CTR | FUN => {
        let arity = arity_of(&prog.aris, term);
        for i in 0..arity {
          find_lets(heap, prog, get_loc(term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    heap: &Heap,
    prog: &Program,
    host: u64,
    names: &HashMap<u64, String>,
    tlocs: &[AtomicU64],
  ) -> String {
    let term = load_ptr(heap, host);
    let done;
    if term == 0 {
      done = format!("<>");
    } else {
      done = match get_tag(term) {
        DP0 => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("a{}", name)
          } else {
            format!("a^{}", get_loc(term, 0))
          }
        }
        DP1 => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("b{}", name)
          } else {
            format!("b^{}", get_loc(term, 0))
          }
        }
        VAR => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("x{}", name)
          } else {
            format!("x^{}", get_loc(term, 0))
          }
        }
        LAM => {
          let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("<lam>")));
          format!("λ{} {}", name, go(heap, prog, get_loc(term, 1), names, tlocs))
        }
        APP => {
          let func = go(heap, prog, get_loc(term, 0), names, tlocs);
          let argm = go(heap, prog, get_loc(term, 1), names, tlocs);
          format!("({} {})", func, argm)
        }
        SUP => {
          //let kind = get_ext(term);
          let func = go(heap, prog, get_loc(term, 0), names, tlocs);
          let argm = go(heap, prog, get_loc(term, 1), names, tlocs);
          format!("{{{} {}}}", func, argm)
        }
        OP2 => {
          let oper = get_ext(term);
          let val0 = go(heap, prog, get_loc(term, 0), names, tlocs);
          let val1 = go(heap, prog, get_loc(term, 1), names, tlocs);
          let symb = match oper {
            0x0 => "+",
            0x1 => "-",
            0x2 => "*",
            0x3 => "/",
            0x4 => "%",
            0x5 => "&",
            0x6 => "|",
            0x7 => "^",
            0x8 => "<<",
            0x9 => ">>",
            0xA => "<",
            0xB => "<=",
            0xC => "=",
            0xD => ">=",
            0xE => ">",
            0xF => "!=",
            _   => "<oper>",
          };
          format!("({} {} {})", symb, val0, val1)
        }
        U60 => {
          format!("{}", u60::val(get_val(term)))
        }
        F60 => {
          format!("{}", f60::val(get_val(term)))
        }
        CTR | FUN => {
          let func = get_ext(term);
          let arit = arity_of(&prog.aris, term);
          let args: Vec<String> = (0..arit).map(|i| go(heap, prog, get_loc(term, i), names, tlocs)).collect();
          let name = &prog.nams.get(&func).unwrap_or(&String::from("<?>")).clone();
          format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
        }
        ERA => "*".to_string(),
        _ => format!("<era:{}>", get_tag(term)),
      };
    }
    for (tid, tid_loc) in tlocs.iter().enumerate() {
      if host == tid_loc.load(Ordering::Relaxed) {
        return format!("<{}>{}", tid, done);
      }
    }
    return done;
  }
  find_lets(heap, prog, host, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(heap, prog, host, &names, tlocs);
  for (_key, pos) in itertools::sorted(lets.iter()) {
    // todo: reverse
    let what = String::from("?h");
    //let kind = kinds.get(&pos).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 = if load_ptr(heap, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 = if load_ptr(heap, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!("\ndup {} {} = {};", nam0, nam1, go(heap, prog, pos + 2, &names, tlocs)));
  }
  text
}

pub fn validate_heap(heap: &Heap) {
  for idx in 0 .. heap.node.len() {
    // If it is an ARG, it must be pointing to a VAR/DP0/DP1 that points to it
    let arg = load_ptr(heap, idx as u64);
    if get_tag(arg) == ARG {
      let var = load_ptr(heap, get_loc(arg, 0));
      let oks = match get_tag(var) {
        VAR => { get_loc(var, 0) == idx as u64 }
        DP0 => { get_loc(var, 0) == idx as u64 }
        DP1 => { get_loc(var, 0) == idx as u64 - 1 }
        _   => { false }
      };
      if !oks {
        panic!("Invalid heap state, due to arg at '{:04x}' of:\n{}", idx, show_heap(heap));
      }
    }
  }
}
