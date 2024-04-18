//// An interaction combinator language
//// ----------------------------------
//// This file implements a textual syntax to interact with the runtime. It includes a pure AST for
//// nets, as well as functions for parsing, stringifying, and converting pure ASTs to runtime nets.
//// On the runtime, a net is represented by a list of active trees, plus a root tree. The textual
//// syntax reflects this representation. The grammar is specified on this repo's README.

//use crate::run;
//use std::collections::BTreeMap;
//use std::collections::HashMap;
//use std::collections::HashSet;
//use std::iter::Peekable;
//use std::str::Chars;

//// AST
//// ---

//#[derive(Clone, Hash, PartialEq, Eq, Debug)]
//pub enum Tree {
  //Era,
  //Con { lft: Box<Tree>, rgt: Box<Tree> },
  //Tup { lft: Box<Tree>, rgt: Box<Tree> },
  //Dup { lab: run::Lab, lft: Box<Tree>, rgt: Box<Tree> },
  //Var { nam: String },
  //Ref { nam: run::Val },
  //Num { val: run::Val },
  //Op1 { opr: run::Lab, lft: run::Val, rgt: Box<Tree> },
  //Op2 { opr: run::Lab, lft: Box<Tree>, rgt: Box<Tree> },
  //Mat { sel: Box<Tree>, ret: Box<Tree> },
//}

//type Redex = Vec<(Tree, Tree)>;

//#[derive(Clone, Hash, PartialEq, Eq, Debug)]
//pub struct Net {
  //pub root: Tree,
  //pub rdex: Redex,
//}

//pub type Book = BTreeMap<String, Net>;

//// Parser
//// ------

//fn skip(chars: &mut Peekable<Chars>) {
  //while let Some(c) = chars.peek() {
    //if *c == '/' {
      //chars.next();
      //while let Some(c) = chars.peek() {
        //if *c == '\n' {
          //break;
        //}
        //chars.next();
      //}
    //} else if !c.is_ascii_whitespace() {
      //break;
    //} else {
      //chars.next();
    //}
  //}
//}

//pub fn consume(chars: &mut Peekable<Chars>, text: &str) -> Result<(), String> {
  //skip(chars);
  //for c in text.chars() {
    //if chars.next() != Some(c) {
      //return Err(format!("Expected '{}', found {:?}", text, chars.peek()));
    //}
  //}
  //return Ok(());
//}

//pub fn parse_decimal(chars: &mut Peekable<Chars>) -> Result<u64, String> {
  //let mut num: u64 = 0;
  //skip(chars);
  //if !chars.peek().map_or(false, |c| c.is_digit(10)) {
    //return Err(format!("Expected a decimal number, found {:?}", chars.peek()));
  //}
  //while let Some(c) = chars.peek() {
    //if !c.is_digit(10) {
      //break;
    //}
    //num = num * 10 + c.to_digit(10).unwrap() as u64;
    //chars.next();
  //}
  //Ok(num)
//}

//pub fn parse_name(chars: &mut Peekable<Chars>) -> Result<String, String> {
  //let mut txt = String::new();
  //skip(chars);
  //if !chars.peek().map_or(false, |c| c.is_alphanumeric() || *c == '_' || *c == '.') {
    //return Err(format!("Expected a name character, found {:?}", chars.peek()))
  //}
  //while let Some(c) = chars.peek() {
    //if !c.is_alphanumeric() && *c != '_' && *c != '.' {
      //break;
    //}
    //txt.push(*c);
    //chars.next();
  //}
  //Ok(txt)
//}

//pub fn parse_opx_lit(chars: &mut Peekable<Chars>) -> Result<String, String> {
  //let mut opx = String::new();
  //skip(chars);
  //while let Some(c) = chars.peek() {
    //if !"+-=*/%<>|&^!?".contains(*c) {
      //break;
    //}
    //opx.push(*c);
    //chars.next();
  //}
  //Ok(opx)
//}

//fn parse_opr(chars: &mut Peekable<Chars>) -> Result<run::Lab, String> {
  //let opx = parse_opx_lit(chars)?;
  //match opx.as_str() {
    //"+"  => Ok(run::ADD),
    //"-"  => Ok(run::SUB),
    //"*"  => Ok(run::MUL),
    //"/"  => Ok(run::DIV),
    //"%"  => Ok(run::MOD),
    //"==" => Ok(run::EQ),
    //"!=" => Ok(run::NE),
    //"<"  => Ok(run::LT),
    //">"  => Ok(run::GT),
    //"<=" => Ok(run::LTE),
    //">=" => Ok(run::GTE),
    //"&&" => Ok(run::AND),
    //"||" => Ok(run::OR),
    //"^"  => Ok(run::XOR),
    //"!"  => Ok(run::NOT),
    //"<<" => Ok(run::LSH),
    //">>" => Ok(run::RSH),
    //_ => Err(format!("Unknown operator: {}", opx)),
  //}
//}

//pub fn parse_tree(chars: &mut Peekable<Chars>) -> Result<Tree, String> {
  //skip(chars);
  //match chars.peek() {
    //Some('*') => {
      //chars.next();
      //Ok(Tree::Era)
    //}
    //Some('(') => {
      //chars.next();
      //let lft = Box::new(parse_tree(chars)?);
      //let rgt = Box::new(parse_tree(chars)?);
      //consume(chars, ")")?;
      //Ok(Tree::Con { lft, rgt })
    //}
    //Some('[') => {
      //chars.next();
      //let lab = 1;
      //let lft = Box::new(parse_tree(chars)?);
      //let rgt = Box::new(parse_tree(chars)?);
      //consume(chars, "]")?;
      //Ok(Tree::Tup { lft, rgt })
    //}
    //Some('{') => {
      //chars.next();
      //let lab = parse_decimal(chars)? as run::Lab;
      //let lft = Box::new(parse_tree(chars)?);
      //let rgt = Box::new(parse_tree(chars)?);
      //consume(chars, "}")?;
      //Ok(Tree::Dup { lab, lft, rgt })
    //}
    //Some('@') => {
      //chars.next();
      //skip(chars);
      //let name = parse_name(chars)?;
      //Ok(Tree::Ref { nam: name_to_val(&name) })
    //}
    //Some('#') => {
      //chars.next();
      //Ok(Tree::Num { val: parse_decimal(chars)? })
    //}
    //Some('<') => {
      //chars.next();
      //if chars.peek().map_or(false, |c| c.is_digit(10)) {
        //let lft = parse_decimal(chars)?;
        //let opr = parse_opr(chars)?;
        //let rgt = Box::new(parse_tree(chars)?);
        //consume(chars, ">")?;
        //Ok(Tree::Op1 { opr, lft, rgt })
      //} else {
        //let opr = parse_opr(chars)?;
        //let lft = Box::new(parse_tree(chars)?);
        //let rgt = Box::new(parse_tree(chars)?);
        //consume(chars, ">")?;
        //Ok(Tree::Op2 { opr, lft, rgt })
      //}
    //}
    //Some('?') => {
      //chars.next();
      //consume(chars, "<")?;
      //let sel = Box::new(parse_tree(chars)?);
      //let ret = Box::new(parse_tree(chars)?);
      //consume(chars, ">")?;
      //Ok(Tree::Mat { sel, ret })
    //}
    //_ => {
      //Ok(Tree::Var { nam: parse_name(chars)? })
    //},
  //}
//}

//pub fn parse_net(chars: &mut Peekable<Chars>) -> Result<Net, String> {
  //let mut rdex = Vec::new();
  //let root = parse_tree(chars)?;
  //while let Some(c) = { skip(chars); chars.peek() } {
    //if *c == '&' {
      //chars.next();
      //let tree1 = parse_tree(chars)?;
      //consume(chars, "~")?;
      //let tree2 = parse_tree(chars)?;
      //rdex.push((tree1, tree2));
    //} else {
      //break;
    //}
  //}
  //Ok(Net { root, rdex })
//}

//pub fn parse_book(chars: &mut Peekable<Chars>) -> Result<Book, String> {
  //let mut book = BTreeMap::new();
  //while let Some(c) = { skip(chars); chars.peek() } {
    //if *c == '@' {
      //chars.next();
      //let name = parse_name(chars)?;
      //consume(chars, "=")?;
      //let net = parse_net(chars)?;
      //book.insert(name, net);
    //} else {
      //break;
    //}
  //}
  //Ok(book)
//}

//fn do_parse<T>(code: &str, parse_fn: impl Fn(&mut Peekable<Chars>) -> Result<T, String>) -> T {
  //let chars = &mut code.chars().peekable();
  //match parse_fn(chars) {
    //Ok(result) => {
      //if chars.next().is_none() {
        //result
      //} else {
        //eprintln!("Unable to parse the whole input. Is this not an hvmc file?");
        //std::process::exit(1);
      //}
    //}
    //Err(err) => {
      //eprintln!("{}", err);
      //std::process::exit(1);
    //}
  //}
//}

//pub fn do_parse_tree(code: &str) -> Tree {
  //do_parse(code, parse_tree)
//}

//pub fn do_parse_net(code: &str) -> Net {
  //do_parse(code, parse_net)
//}

//pub fn do_parse_book(code: &str) -> Book {
  //do_parse(code, parse_book)
//}

//// Stringifier
//// -----------

//pub fn show_opr(opr: run::Lab) -> String {
  //match opr {
    //run::ADD => "+".to_string(),
    //run::SUB => "-".to_string(),
    //run::MUL => "*".to_string(),
    //run::DIV => "/".to_string(),
    //run::MOD => "%".to_string(),
    //run::EQ  => "==".to_string(),
    //run::NE  => "!=".to_string(),
    //run::LT  => "<".to_string(),
    //run::GT  => ">".to_string(),
    //run::LTE => "<=".to_string(),
    //run::GTE => ">=".to_string(),
    //run::AND => "&&".to_string(),
    //run::OR  => "||".to_string(),
    //run::XOR => "^".to_string(),
    //run::NOT => "!".to_string(),
    //run::LSH => "<<".to_string(),
    //run::RSH => ">>".to_string(),
    //_        => panic!("Unknown operator label."),
  //}
//}

//pub fn show_tree(tree: &Tree) -> String {
  //match tree {
    //Tree::Era => {
      //"*".to_string()
    //}
    //Tree::Con { lft, rgt } => {
      //format!("({} {})", show_tree(&*lft), show_tree(&*rgt))
    //}
    //Tree::Tup { lft, rgt } => {
      //format!("[{} {}]", show_tree(&*lft), show_tree(&*rgt))
    //}
    //Tree::Dup { lab, lft, rgt } => {
      //format!("{{{} {} {}}}", lab, show_tree(&*lft), show_tree(&*rgt))
    //}
    //Tree::Var { nam } => {
      //nam.clone()
    //}
    //Tree::Ref { nam } => {
      //format!("@{}", val_to_name(*nam))
    //}
    //Tree::Num { val } => {
      //format!("#{}", (*val).to_string())
    //}
    //Tree::Op1 { opr, lft, rgt } => {
      //format!("<{}{} {}>", lft, show_opr(*opr), show_tree(rgt))
    //}
    //Tree::Op2 { opr, lft, rgt } => {
      //format!("<{} {} {}>", show_opr(*opr), show_tree(&*lft), show_tree(&*rgt))
    //}
    //Tree::Mat { sel, ret } => {
      //format!("?<{} {}>", show_tree(&*sel), show_tree(&*ret))
    //}
  //}
//}

//pub fn show_net(net: &Net) -> String {
  //let mut result = String::new();
  //result.push_str(&format!("{}", show_tree(&net.root)));
  //for (a, b) in &net.rdex {
    //result.push_str(&format!("\n& {} ~ {}", show_tree(a), show_tree(b)));
  //}
  //return result;
//}

//pub fn show_book(book: &Book) -> String {
  //let mut result = String::new();
  //for (name, net) in book {
    //result.push_str(&format!("@{} = {}\n", name, show_net(net)));
  //}
  //return result;
//}

//pub fn show_runtime_tree(rt_net: &run::Net, ptr: run::Ptr) -> String {
  //show_tree(&tree_from_runtime_go(rt_net, ptr, PARENT_ROOT, &mut HashMap::new(), &mut 0))
//}

//pub fn show_runtime_net(rt_net: &run::Net) -> String {
  //show_net(&net_from_runtime(rt_net))
//}

//pub fn show_runtime_book(book: &run::Book) -> String {
  //show_book(&book_from_runtime(book))
//}

//// Conversion
//// ----------

//pub fn num_to_str(mut num: usize) -> String {
  //let mut txt = String::new();
  //num += 1;
  //while num > 0 {
    //num -= 1;
    //let c = ((num % 26) as u8 + b'a') as char;
    //txt.push(c);
    //num /= 26;
  //}
  //return txt.chars().rev().collect();
//}

//pub const fn tag_to_port(tag: run::Tag) -> run::Port {
  //match tag {
    //run::VR1 => run::P1,
    //run::VR2 => run::P2,
    //_        => unreachable!(),
  //}
//}

//pub fn port_to_tag(port: run::Port) -> run::Tag {
  //match port {
    //run::P1 => run::VR1,
    //run::P2 => run::VR2,
    //_        => unreachable!(),
  //}
//}

//pub fn name_to_letters(name: &str) -> Vec<u8> {
  //let mut letters = Vec::new();
  //for c in name.chars() {
    //letters.push(match c {
      //'0'..='9' => c as u8 - '0' as u8 + 0,
      //'A'..='Z' => c as u8 - 'A' as u8 + 10,
      //'a'..='z' => c as u8 - 'a' as u8 + 36,
      //'_'       => 62,
      //'.'       => 63,
      //_         => panic!("Invalid character in name"),
    //});
  //}
  //return letters;
//}

//pub fn letters_to_name(letters: Vec<u8>) -> String {
  //let mut name = String::new();
  //for letter in letters {
    //name.push(match letter {
       //0..= 9 => (letter - 0 + '0' as u8) as char,
      //10..=35 => (letter - 10 + 'A' as u8) as char,
      //36..=61 => (letter - 36 + 'a' as u8) as char,
      //62      => '_',
      //63      => '.',
      //_       => panic!("Invalid letter in name"),
    //});
  //}
  //return name;
//}

//pub fn val_to_letters(num: run::Val) -> Vec<u8> {
  //let mut letters = Vec::new();
  //let mut num = num;
  //while num > 0 {
    //letters.push((num % 64) as u8);
    //num /= 64;
  //}
  //letters.reverse();
  //return letters;
//}

//pub fn letters_to_val(letters: Vec<u8>) -> run::Val {
  //let mut num = 0;
  //for letter in letters {
    //num = num * 64 + letter as run::Val;
  //}
  //return num;
//}

//pub fn name_to_val(name: &str) -> run::Val {
  //letters_to_val(name_to_letters(name))
//}

//pub fn val_to_name(num: run::Val) -> String {
  //letters_to_name(val_to_letters(num))
//}

//// Injection and Readback
//// ----------------------

//// To runtime

//#[derive(Debug, Clone, PartialEq, Eq, Hash)]
//pub enum Parent {
  //Redex,
  //Node { loc: run::Loc, port: run::Port },
//}
//const PARENT_ROOT: Parent = Parent::Node { loc: run::ROOT.loc(), port: tag_to_port(run::ROOT.tag()) };

//pub fn tree_to_runtime_go(rt_net: &mut run::Net, tree: &Tree, vars: &mut HashMap<String, Parent>, parent: Parent) -> run::Ptr {
  //match tree {
    //Tree::Era => {
      //run::ERAS
    //}
    //Tree::Con { lft, rgt } => {
      //let loc = rt_net.alloc();
      //let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::LAM, 0, loc)
    //}
    //Tree::Tup { lft, rgt } => {
      //let loc = rt_net.alloc();
      //let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::TUP, 1, loc)
    //}
    //Tree::Dup { lab, lft, rgt } => {
      //let loc = rt_net.alloc();
      //let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::DUP, *lab, loc)
    //}
    //Tree::Var { nam } => {
      //if let Parent::Redex = parent {
        //panic!("By definition, can't have variable on active pairs.");
      //};
      //match vars.get(nam) {
        //Some(Parent::Redex) => {
          //unreachable!();
        //}
        //Some(Parent::Node { loc: other_loc, port: other_port }) => {
          //match parent {
            //Parent::Redex => { unreachable!(); }
            //Parent::Node { loc, port } => rt_net.heap.set(*other_loc, *other_port, run::Ptr::new(port_to_tag(port), 0, loc)),
          //}
          //return run::Ptr::new(port_to_tag(*other_port), 0, *other_loc);
        //}
        //None => {
          //vars.insert(nam.clone(), parent);
          //run::NULL
        //}
      //}
    //}
    //Tree::Ref { nam } => {
      //run::Ptr::big(run::REF, *nam)
    //}
    //Tree::Num { val } => {
      //run::Ptr::big(run::NUM, *val)
    //}
    //Tree::Op1 { opr, lft, rgt } => {
      //let loc = rt_net.alloc();
      //let p1 = run::Ptr::big(run::NUM, *lft);
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, rgt, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::OP1, *opr, loc)
    //}
    //Tree::Op2 { opr, lft, rgt } => {
      //let loc = rt_net.alloc();
      //let p1 = tree_to_runtime_go(rt_net, &*lft, vars, Parent::Node { loc, port: run::P1 });
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, &*rgt, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::OP2, *opr, loc)
    //}
    //Tree::Mat { sel, ret } => {
      //let loc = rt_net.alloc();
      //let p1 = tree_to_runtime_go(rt_net, &*sel, vars, Parent::Node { loc, port: run::P1 });
      //rt_net.heap.set(loc, run::P1, p1);
      //let p2 = tree_to_runtime_go(rt_net, &*ret, vars, Parent::Node { loc, port: run::P2 });
      //rt_net.heap.set(loc, run::P2, p2);
      //run::Ptr::new(run::MAT, 0, loc)
    //}
  //}
//}

//pub fn tree_to_runtime(rt_net: &mut run::Net, tree: &Tree) -> run::Ptr {
  //tree_to_runtime_go(rt_net, tree, &mut HashMap::new(), PARENT_ROOT)
//}

//pub fn net_to_runtime(rt_net: &mut run::Net, net: &Net) {
  //let mut vars = HashMap::new();
  //let root = tree_to_runtime_go(rt_net, &net.root, &mut vars, PARENT_ROOT);
  //rt_net.heap.set_root(root);
  //for (tree1, tree2) in &net.rdex {
    //let ptr1 = tree_to_runtime_go(rt_net, tree1, &mut vars, Parent::Redex);
    //let ptr2 = tree_to_runtime_go(rt_net, tree2, &mut vars, Parent::Redex);
    //rt_net.rdex.push((ptr1, ptr2));
  //}
//}

//#[derive(Debug)]
//pub struct Inside {
  //dups: bool, // has dups; TODO: collect dup labels
  //refs: HashSet<run::Val>, // ref names
//}

//// Checks if a runtime net has dups, and collects its ref ids
//pub fn runtime_net_inside(def: &run::Def) -> Inside {
  //let mut inside = Inside { dups: false, refs: HashSet::new() };
  //fn register(inside: &mut Inside, ptr: run::Ptr) {
    //if ptr.is_dup() {
      //inside.dups = true;
    //}
    //if ptr.is_ref() {
      //inside.refs.insert(ptr.val());
    //}
  //}
  //for i in 0 .. def.node.len() {
    //register(&mut inside, def.node[i].0);
    //register(&mut inside, def.node[i].1);
  //}
  //for i in 0 .. def.rdex.len() {
    //register(&mut inside, def.rdex[i].0);
    //register(&mut inside, def.rdex[i].1);
  //}
  //return inside;
//}

//// A runtime def is safe when neither it nor any of its dependencies have dup nodes.
//// FIXME: memoize to avoid duplicated work
//pub fn runtime_def_is_safe(inside: &Inside, rt_book: &run::Book, fid: run::Val, seen: &mut HashSet<run::Val>) -> bool {
  //if seen.contains(&fid) {
    //return true;
  //}
  //if inside.dups {
    //return false;
  //}
  //seen.insert(fid);
  //for ref_id in &inside.refs {
    //if !runtime_def_is_safe(inside, rt_book, *ref_id, seen) {
      //return false;
    //}
  //}
  //return true;
//}

//// Converts a book from the pure AST representation to the runtime representation.
//pub fn book_to_runtime(book: &Book) -> run::Book {
  //let mut rt_book = run::Book::new();

  //// Convert each network in 'book' to a runtime network and add to 'rt_book'
  //for (name, net) in book {
    //let fid = name_to_val(name);
    //let data = run::Heap::init(1 << 16);
    //let mut rt = run::Net::new(&data);
    //net_to_runtime(&mut rt, net);
    //rt_book.def(fid, runtime_net_to_runtime_def(&rt));
  //}

  //// Calculate the 'insides' of each runtime definition
  //let mut insides = HashMap::new();
  //for (fid, def) in &rt_book.defs {
    //insides.insert(*fid, runtime_net_inside(&def));
  //}

  //// Determine which definitions are safe and store their 'fid' in 'is_safe'
  //let mut is_safe = HashSet::new();
  //for (fid, _def) in &rt_book.defs {
    //if runtime_def_is_safe(&insides[&fid], &rt_book, *fid, &mut HashSet::new()) {
      //is_safe.insert(*fid);
    //}
  //}

  //// Set the 'safe' flag for each definition in 'rt_book' that is safe
  //for (fid, def) in &mut rt_book.defs {
    //if is_safe.contains(fid) {
      //def.safe = true;
    //}
  //}

  //rt_book
//}

//// Converts to a def.
//pub fn runtime_net_to_runtime_def(net: &run::Net) -> run::Def {
  //let mut node = vec![];
  //let mut rdex = vec![];
  //let safe = false;
  //for i in 0 .. net.heap.data.len() {
    //let p1 = net.heap.get(node.len() as run::Loc, run::P1);
    //let p2 = net.heap.get(node.len() as run::Loc, run::P2);
    //if p1 != run::NULL || p2 != run::NULL {
      //node.push((p1, p2));
    //} else {
      //break;
    //}
  //}
  //for i in 0 .. net.rdex.len() {
    //let p1 = net.rdex[i].0;
    //let p2 = net.rdex[i].1;
    //rdex.push((p1, p2));
  //}
  //return run::Def { safe, rdex, node };
//}

//// Reads back from a def.
//pub fn runtime_def_to_runtime_net<'a>(data: &'a run::Data, def: &run::Def) -> run::Net<'a> {
  //let mut net = run::Net::new(&data);
  //for (i, &(p1, p2)) in def.node.iter().enumerate() {
    //net.heap.set(i as run::Loc, run::P1, p1);
    //net.heap.set(i as run::Loc, run::P2, p2);
  //}
  //net.rdex = def.rdex.clone();
  //net
//}

//pub fn tree_from_runtime_go(rt_net: &run::Net, ptr: run::Ptr, parent: Parent, vars: &mut HashMap<Parent, String>, fresh: &mut usize) -> Tree {
  //match ptr.tag() {
    //run::ERA => {
      //Tree::Era
    //}
    //run::REF => {
      //Tree::Ref { nam: ptr.val() }
    //}
    //run::NUM => {
      //Tree::Num { val: ptr.val() }
    //}
    //run::OP1 => {
      //let opr = ptr.lab();
      //let lft = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P1), Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let Tree::Num { val } = lft else { unreachable!() };
      //let rgt = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P2), Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Op1 { opr, lft: val, rgt: Box::new(rgt) }
    //}
    //run::OP2 => {
      //let opr = ptr.lab();
      //let lft = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P1), Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let rgt = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P2), Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Op2 { opr, lft: Box::new(lft), rgt: Box::new(rgt) }
    //}
    //run::MAT => {
      //let sel = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P1), Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let ret = tree_from_runtime_go(rt_net, rt_net.heap.get(ptr.loc(), run::P2), Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Mat { sel: Box::new(sel), ret: Box::new(ret) }
    //}
    //run::VR1 | run::VR2 => {
      //let key = match ptr.tag() {
        //run::VR1 => Parent::Node { loc: ptr.loc(), port: run::P1 },
        //run::VR2 => Parent::Node { loc: ptr.loc(), port: run::P2 },
        //_        => unreachable!(),
      //};
      //if let Some(nam) = vars.get(&key) {
        //Tree::Var { nam: nam.clone() }
      //} else {
        //let nam = num_to_str(*fresh);
        //*fresh += 1;
        //vars.insert(parent, nam.clone());
        //Tree::Var { nam }
      //}
    //}
    //run::LAM => {
      //let p1  = rt_net.heap.get(ptr.loc(), run::P1);
      //let p2  = rt_net.heap.get(ptr.loc(), run::P2);
      //let lft = tree_from_runtime_go(rt_net, p1, Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let rgt = tree_from_runtime_go(rt_net, p2, Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Con { lft: Box::new(lft), rgt: Box::new(rgt) }
    //}
    //run::TUP => {
      //let p1  = rt_net.heap.get(ptr.loc(), run::P1);
      //let p2  = rt_net.heap.get(ptr.loc(), run::P2);
      //let lft = tree_from_runtime_go(rt_net, p1, Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let rgt = tree_from_runtime_go(rt_net, p2, Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Tup { lft: Box::new(lft), rgt: Box::new(rgt) }
    //}
    //run::DUP => {
      //let p1  = rt_net.heap.get(ptr.loc(), run::P1);
      //let p2  = rt_net.heap.get(ptr.loc(), run::P2);
      //let lft = tree_from_runtime_go(rt_net, p1, Parent::Node { loc: ptr.loc(), port: run::P1 }, vars, fresh);
      //let rgt = tree_from_runtime_go(rt_net, p2, Parent::Node { loc: ptr.loc(), port: run::P2 }, vars, fresh);
      //Tree::Dup { lab: ptr.lab(), lft: Box::new(lft), rgt: Box::new(rgt) }
    //}
    //_ => {
      //unreachable!()
    //}
  //}
//}

//pub fn tree_from_runtime(rt_net: &run::Net, ptr: run::Ptr) -> Tree {
  //let mut vars = HashMap::new();
  //let mut fresh = 0;
  //tree_from_runtime_go(rt_net, ptr, PARENT_ROOT, &mut vars, &mut fresh)
//}

//pub fn net_from_runtime(rt_net: &run::Net) -> Net {
  //let mut vars = HashMap::new();
  //let mut fresh = 0;
  //let mut rdex = Vec::new();
  //let root = tree_from_runtime_go(rt_net, rt_net.heap.get_root(), PARENT_ROOT, &mut vars, &mut fresh);
  //for &(a, b) in &rt_net.rdex {
    //let tree_a = tree_from_runtime_go(rt_net, a, Parent::Redex, &mut vars, &mut fresh);
    //let tree_b = tree_from_runtime_go(rt_net, b, Parent::Redex, &mut vars, &mut fresh);
    //rdex.push((tree_a, tree_b));
  //}
  //Net { root, rdex }
//}

//pub fn book_from_runtime(rt_book: &run::Book) -> Book {
  //let mut book = BTreeMap::new();
  //for (fid, def) in rt_book.defs.iter() {
    //if def.node.len() > 0 {
      //let name = val_to_name(*fid);
      //let data = run::Heap::init(def.node.len());
      //let net  = net_from_runtime(&runtime_def_to_runtime_net(&data, &def));
      //book.insert(name, net);
    //}
  //}
  //book
//}
