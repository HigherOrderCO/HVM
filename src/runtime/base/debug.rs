use crate::runtime::*;
use std::collections::{hash_map, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

// Debug
// -----

pub fn show_ptr(x: Ptr) -> String {
  if x == 0 {
    String::from("~")
  } else {
    let tag = x.tag();
    let ext = get_ext(x);
    let val = get_val(x);
    format!("{}({:07x}, {:08x})", tag.as_str(), ext, val)
  }
}

pub fn show_heap(heap: &Heap) -> String {
  let mut text: String = String::new();
  for idx in 0..heap.node.len() {
    let ptr = heap.load_ptr(idx as u64);
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
    let term = heap.load_ptr(host);
    if term == 0 {
      return;
    }
    match term.tag() {
      Tag::LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      Tag::APP => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      Tag::SUP => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      Tag::DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, get_loc(term, 2), lets, kinds, names, count);
        }
      }
      Tag::DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(heap, prog, get_loc(term, 2), lets, kinds, names, count);
        }
      }
      Tag::OP2 => {
        find_lets(heap, prog, get_loc(term, 0), lets, kinds, names, count);
        find_lets(heap, prog, get_loc(term, 1), lets, kinds, names, count);
      }
      Tag::CTR | Tag::FUN => {
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
    let term = heap.load_ptr(host);

    let done = if term == 0 {
      "<>".to_string()
    } else {
      match term.tag() {
        Tag::DP0 => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("a{}", name)
          } else {
            format!("a^{}", get_loc(term, 0))
          }
        }
        Tag::DP1 => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("b{}", name)
          } else {
            format!("b^{}", get_loc(term, 0))
          }
        }
        Tag::VAR => {
          if let Some(name) = names.get(&get_loc(term, 0)) {
            format!("x{}", name)
          } else {
            format!("x^{}", get_loc(term, 0))
          }
        }
        Tag::LAM => {
          let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("<lam>")));
          format!("λ{} {}", name, go(heap, prog, get_loc(term, 1), names, tlocs))
        }
        Tag::APP => {
          let func = go(heap, prog, get_loc(term, 0), names, tlocs);
          let argm = go(heap, prog, get_loc(term, 1), names, tlocs);
          format!("({} {})", func, argm)
        }
        Tag::SUP => {
          //let kind = get_ext(term);
          let func = go(heap, prog, get_loc(term, 0), names, tlocs);
          let argm = go(heap, prog, get_loc(term, 1), names, tlocs);
          format!("{{{} {}}}", func, argm)
        }
        Tag::OP2 => {
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
            _ => "<oper>",
          };
          format!("({} {} {})", symb, val0, val1)
        }
        Tag::U60 => {
          format!("{}", u60::val(get_val(term)))
        }
        Tag::F60 => {
          format!("{}", f60::val(get_val(term)))
        }
        Tag::CTR | Tag::FUN => {
          let func = get_ext(term);
          let arit = arity_of(&prog.aris, term);
          let args: Vec<String> =
            (0..arit).map(|i| go(heap, prog, get_loc(term, i), names, tlocs)).collect();
          let name = &prog.nams.get(&func).unwrap_or(&String::from("<?>")).clone();
          format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
        }
        Tag::ERA => "*".to_string(),
        _ => format!("<era:{}>", term.tag()),
      }
    };
    for (tid, tid_loc) in tlocs.iter().enumerate() {
      if host == tid_loc.load(Ordering::Relaxed) {
        return format!("<{}>{}", tid, done);
      }
    }
    done
  }
  find_lets(heap, prog, host, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(heap, prog, host, &names, tlocs);
  for (_key, pos) in itertools::sorted(lets.iter()) {
    // todo: reverse
    let what = String::from("?h");
    //let kind = kinds.get(&pos).unwrap_or(&0);
    let name = names.get(pos).unwrap_or(&what);
    let nam0 = if heap.load_ptr(pos + 0) == Tag::ERA.as_u64() {
      String::from("*")
    } else {
      format!("a{}", name)
    };
    let nam1 = if heap.load_ptr(pos + 1) == Tag::ERA.as_u64() {
      String::from("*")
    } else {
      format!("b{}", name)
    };
    text.push_str(&format!(
      "\ndup {} {} = {};",
      nam0,
      nam1,
      go(heap, prog, pos + 2, &names, tlocs)
    ));
  }
  text
}

pub fn validate_heap(heap: &Heap) {
  for idx in 0..heap.node.len() {
    // If it is an ARG, it must be pointing to a VAR/DP0/DP1 that points to it
    let arg = heap.load_ptr(idx as u64);
    if arg.tag() == Tag::ARG {
      let var = heap.load_ptr(get_loc(arg, 0));
      let oks = match var.tag() {
        Tag::VAR => get_loc(var, 0) == idx as u64,
        Tag::DP0 => get_loc(var, 0) == idx as u64,
        Tag::DP1 => get_loc(var, 0) == idx as u64 - 1,
        _ => false,
      };
      if !oks {
        panic!("Invalid heap state, due to arg at '{:04x}' of:\n{}", idx, show_heap(heap));
      }
    }
  }
}
