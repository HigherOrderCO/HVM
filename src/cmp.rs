// ast.rs (NEW)
// ============

//./ast.rs//

// hvm.rs (NEW)
// ============

//./hvm.rs//

// hvm.c (NEW)
// ===========

//./hvm.c//

// cmp.rs (OLD)
// ============

//./old_cmp.rs//

// HOLE-FILLER GOAL
// ================

//The `ast.rs` file has been developed for an old version of HVM, which had a different runtime
//with different tags. It compiled HVM to Rust. Now, we're compiling HVM to C instead. Changes:

//VR1/VR2/RD1/RD2, which were pointers to locations in memory, were simplified and merged into a
//single VAR tag, which is a named link. The OP1 has been removed; only OP2 is used now. The MAT
//tag has been renamed to SWI, the LAM tag has been renamed to CON, and the TUP tag has been
//removed. Also, CON/DUP tags don't have a label anymore, as they used to.

//Also, since the VAR tags have changed, the linking algorithm has been simplified. In special, Ptrs
//have been renamed to Port, and there is no more the concept of Trg (i.e., a port that points to a
//port). This isn't necessary anymore. Similarly, all the link functions have been simplified into a
//single 'link()' function that links two ports.

//Your GOAL is to study and understand the old cmp.rs file to your best efforts. Then, create a
//BRAND NEW cmp.rs file that generates compiled hvm.c functions. Create the new cmp.rs below:

//Note that this is a new repository. Check the hvm.rs and ast.rs files below. You can ONLY use
//functions exported by these two files. Other functions (even when mentioned inside the old
//cmp.rs) are NOT available and can NOT be used.

//Keep the exact SAME compilation strategy.

// cmp.rs (NEW)
// ============

use crate::ast;
use crate::hvm;

use std::collections::HashMap;

pub fn gen_ident(tab: usize) -> String {
  return "  ".repeat(tab);
}

fn gen_fresh(count: &mut usize) -> String {
  *count += 1;
  format!("k{}", count)
}

pub fn compile_book(book: &hvm::Book) -> String {
  let mut code = String::new();

  // TODO: generate the C 'interact_call_native' fn below:

  code.push_str(&format!("bool interact_call_native(GNet *net, TMem *tm, Port a, Port b) {{\n"));
  code.push_str(&format!("  u32 fid = get_val(a);\n"));
  code.push_str(&format!("  switch (fid) {{\n"));
  for (fid, def) in book.defs.iter().enumerate() {
    code.push_str(&format!("    case {}: return interact_call_{}(net, tm, a, b);\n", fid, &def.name));
  }
  code.push_str(&format!("    default: return false;\n"));
  code.push_str(&format!("  }}\n"));
  code.push_str(&format!("}}\n\n"));

  for fid in 0..book.defs.len() {
    code.push_str(&compile_def(book, 0, fid as hvm::Val));
    code.push_str(&format!("\n"));
  }

  return code;
}

pub fn compile_def(book: &hvm::Book, tab: usize, fid: hvm::Val) -> String {
  let def = &book.defs[fid as usize];
  let fun = &def.name;

  // Initializes context
  let fresh = &mut 0;
  let subst = &mut HashMap::new();
  
  // Generates function
  let mut code = String::new();
  code.push_str(&format!("{}bool interact_call_{}(GNet *net, TMem *tm, Port a, Port b) {{\n", gen_ident(tab), fun));
  for redex in &def.rbag {
    let (f,x) = (redex.get_fst(), redex.get_snd());
    let f_nam = format!("_{}", gen_fresh(fresh));
    code.push_str(&format!("{}Port {} = {};\n", gen_ident(tab+1), f_nam, &compile_atom(f)));
    code.push_str(&compile_link_hot(book, fresh, subst, tab+1, def, x, &f_nam));
  }
  code.push_str(&compile_link_hot(book, fresh, subst, tab+1, def, def.node[0].get_fst(), "b"));
  code.push_str(&format!("{}return true;\n", gen_ident(tab+1)));
  code.push_str(&format!("{}}}\n", gen_ident(tab)));

  return code;
}

pub fn compile_link_hot(
  book  : &hvm::Book,
  fresh : &mut usize,
  subst : &mut HashMap<hvm::Val, String>,
  tab   : usize,
  def   : &hvm::Def,
  a     : hvm::Port,
  b     : &str,
) -> String {
  let mut code = String::new();
  code.push_str(&compile_link_cold(book, fresh, subst, tab, def, a, b));
  return code;
}

//fn got(vars: &HashMap<hvm::Port, String>, def: &run::Def, a: run::Ptr) -> Option<String> {
  //if a.is_var() {
    //let got = def.node[a.get_val() as usize];
    //let slf = if a.tag() == run::VR1 { got.0 } else { got.1 };
    //return vars.get(&slf).cloned();
  //} else {
    //return None;
  //}
//}

// ((a b) (c d))
pub fn compile_node(
  book  : &hvm::Book,
  fresh : &mut usize,
  subst : &mut HashMap<hvm::Val, String>,
  tab   : usize,
  def   : &hvm::Def,
  a     : hvm::Port,
) -> String {
  let mut code = String::new();
  if a.is_nod() {
    let nd = gen_fresh(fresh);
    let p1 = def.node[a.get_val() as usize].get_fst();
    let p2 = def.node[a.get_val() as usize].get_snd();
    code.push_str(&format!("{}Val {} = node_alloc();\n", gen_ident(tab), nd));
    code.push_str("node_create(net, tm->node_loc[{}], new_pair({},{}));", nd, p1, p2);
  } else {
    return format!("vars_create(net, tm->vars_loc[{:04x}], new_port(VAR, tm->vars_loc[{:04x}]));", a.get_val(), a.get_val());
  }
}

fn compile_atom(port: hvm::Port) -> String {
  return format!("Ptr::new({},0x{:08x})", compile_tag(port.get_tag()), port.get_val());
}


pub fn compile_tag(tag: hvm::Tag) -> &'static str {
  match tag {
    hvm::VAR => "VAR",
    hvm::REF => "REF",
    hvm::ERA => "ERA",
    hvm::NUM => "NUM",
    hvm::OPR => "OPR",
    hvm::SWI => "SWI",
    hvm::CON => "CON",
    hvm::DUP => "DUP",
    _ => unreachable!(),
  }
}






// [(a b)] ~ X
// -----------
// [a] ~ X
// [b] ~ X
















