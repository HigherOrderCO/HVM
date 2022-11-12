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
      U60 => "U60",
      F60 => "F60",
      _   => "?",
    };
    format!("{}({:07x}, {:08x})", tgs, ext, val)
  }
}

pub fn show_heap(heap: &Heap) -> String {
  let mut text: String = String::new();
  for idx in 0 .. HEAP_SIZE {
    let ptr = heap.node.data[idx].load(Ordering::Relaxed);
    if ptr != 0 {
      text.push_str(&format!("{:04x} | ", idx));
      text.push_str(&show_ptr(ptr));
      text.push('\n');
    }
  }
  text
}

pub fn show_term(heap: &Heap, prog: &Program, term: Ptr, focus: u64) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    heap: &Heap,
    prog: &Program,
    term: Ptr,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    if term == 0 {
      return;
    }
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      SUP => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, load_arg(heap, term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, load_arg(heap, term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(heap, prog, load_arg(heap, term, 0), lets, kinds, names, count);
        find_lets(heap, prog, load_arg(heap, term, 1), lets, kinds, names, count);
      }
      CTR | FUN => {
        let arity = arity_of(&prog.arit, term);
        for i in 0..arity {
          find_lets(heap, prog, load_arg(heap, term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    heap: &Heap,
    prog: &Program,
    term: Ptr,
    names: &HashMap<u64, String>,
    focus: u64,
  ) -> String {
    if term == 0 {
      return format!("<>");
    }
    let done = match get_tag(term) {
      DP0 => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("a{}", name);
        } else {
          return format!("a^{}", get_loc(term, 0));
        }
      }
      DP1 => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("b{}", name);
        } else {
          return format!("b^{}", get_loc(term, 0));
        }
      }
      VAR => {
        if let Some(name) = names.get(&get_loc(term, 0)) {
          return format!("x{}", name);
        } else {
          return format!("x^{}", get_loc(term, 0));
        }
      }
      LAM => {
        let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("<lam>")));
        format!("λ{} {}", name, go(heap, prog, load_arg(heap, term, 1), names, focus))
      }
      APP => {
        let func = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let argm = go(heap, prog, load_arg(heap, term, 1), names, focus);
        format!("({} {})", func, argm)
      }
      SUP => {
        let kind = get_ext(term);
        let func = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let argm = go(heap, prog, load_arg(heap, term, 1), names, focus);
        format!("#{}{{{} {}}}", kind, func, argm)
      }
      OP2 => {
        let oper = get_ext(term);
        let val0 = go(heap, prog, load_arg(heap, term, 0), names, focus);
        let val1 = go(heap, prog, load_arg(heap, term, 1), names, focus);
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
        let arit = arity_of(&prog.arit, term);
        let args: Vec<String> = (0..arit).map(|i| go(heap, prog, load_arg(heap, term, i), names, focus)).collect();
        let name = &prog.nams.get(&func).unwrap_or(&String::from("<?>")).clone();
        format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
      ERA => "*".to_string(),
      _ => format!("<era:{}>", get_tag(term)),
    };
    if term == focus {
      format!("${}", done)
    } else {
      done
    }
  }
  find_lets(heap, prog, term, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(heap, prog, term, &names, focus);
  for (_key, pos) in itertools::sorted(lets.iter()) {
    // todo: reverse
    let what = String::from("?h");
    let kind = kinds.get(&pos).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 = if load_ptr(heap, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 = if load_ptr(heap, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!("\ndup#{}[{:x}] {} {} = {};", kind, pos, nam0, nam1, go(heap, prog, load_ptr(heap, pos + 2), &names, focus)));
  }
  text
}

pub fn validate_heap(heap: &Heap) {
  for idx in 0 .. HEAP_SIZE {
    // If it is an ARG, it must be pointing to a VAR/DP0/DP1 that points to it
    let arg = heap.node.data[idx].load(Ordering::Relaxed);
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
