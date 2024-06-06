use TSPL::{new_parser, Parser};
use highlight_error::highlight_error;
use crate::hvm;
use std::{collections::{btree_map::Entry, BTreeMap, BTreeSet}, fmt::{Debug, Display}};

// Types
// -----

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Numb(pub u32);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Tree {
  Var { nam: String },
  Ref { nam: String },
  Era,
  Num { val: Numb },
  Con { fst: Box<Tree>, snd: Box<Tree> },
  Dup { fst: Box<Tree>, snd: Box<Tree> },
  Opr { fst: Box<Tree>, snd: Box<Tree> },
  Swi { fst: Box<Tree>, snd: Box<Tree> },
}

pub type Redex = (bool, Tree, Tree);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Net {
  pub root: Tree,
  pub rbag: Vec<Redex>,
}

pub struct Book {
  pub defs: BTreeMap<String, Net>,
}

// Parser
// ------

new_parser!(CoreParser);

impl<'i> CoreParser<'i> {

  pub fn parse_numb_sym(&mut self) -> Result<Numb, String> {
    self.consume("[")?;

    // numeric casts
    if let Some(cast) = match () {
      _ if self.try_consume("u24") => Some(hvm::TY_U24),
      _ if self.try_consume("i24") => Some(hvm::TY_I24),
      _ if self.try_consume("f24") => Some(hvm::TY_F24),
      _ => None
    } {
      // Casts can't be partially applied, so nothing should follow.
      self.consume("]")?;

      return Ok(Numb(hvm::Numb::new_sym(cast).0));
    }

    // Parses the symbol
    let op = hvm::Numb::new_sym(match () {
      // numeric operations
      _ if self.try_consume("+")   => hvm::OP_ADD,
      _ if self.try_consume("-")   => hvm::OP_SUB,
      _ if self.try_consume(":-")  => hvm::FP_SUB,
      _ if self.try_consume("*")   => hvm::OP_MUL,
      _ if self.try_consume("/")   => hvm::OP_DIV,
      _ if self.try_consume(":/")  => hvm::FP_DIV,
      _ if self.try_consume("%")   => hvm::OP_REM,
      _ if self.try_consume(":%")  => hvm::FP_REM,
      _ if self.try_consume("=")   => hvm::OP_EQ,
      _ if self.try_consume("!")   => hvm::OP_NEQ,
      _ if self.try_consume("<<")  => hvm::OP_SHL,
      _ if self.try_consume(":<<") => hvm::FP_SHL,
      _ if self.try_consume(">>")  => hvm::OP_SHR,
      _ if self.try_consume(":>>") => hvm::FP_SHR,
      _ if self.try_consume("<")   => hvm::OP_LT,
      _ if self.try_consume(">")   => hvm::OP_GT,
      _ if self.try_consume("&")   => hvm::OP_AND,
      _ if self.try_consume("|")   => hvm::OP_OR,
      _ if self.try_consume("^")   => hvm::OP_XOR,
      _ => self.expected("operator symbol")?,
    });

    self.skip_trivia();
    // Syntax for partial operations, like `[*2]`
    let num = if self.peek_one() != Some(']') {
      hvm::Numb::partial(op, hvm::Numb(self.parse_numb_lit()?.0))
    } else {
      op
    };

    // Closes symbol bracket
    self.consume("]")?;

    // Returns the symbol
    return Ok(Numb(num.0));
  }

  pub fn parse_numb_lit(&mut self) -> Result<Numb, String> {
    let ini = self.index;
    let num = self.take_while(|x| x.is_alphanumeric() || x == '+' || x == '-' || x == '.');
    let end = self.index;
    Ok(Numb(if num.contains('.') || num.contains("inf") || num.contains("NaN") {
      let val: f32 = num.parse().map_err(|err| format!("invalid number literal: {}\n{}", err, highlight_error(ini, end, self.input)))?;
      hvm::Numb::new_f24(val)
    } else if num.starts_with('+') || num.starts_with('-') {
      let val = Self::parse_int(&num[1..])? as i32;
      hvm::Numb::new_i24(if num.starts_with('-') { -val } else { val })
    } else {
      let val = Self::parse_int(num)? as u32;
      hvm::Numb::new_u24(val)
    }.0))
  }

  fn parse_int(input: &str) -> Result<u64, String> {
    if let Some(rest) = input.strip_prefix("0x") {
      u64::from_str_radix(rest, 16).map_err(|err| format!("{err:?}"))
    } else if let Some(rest) = input.strip_prefix("0b") {
      u64::from_str_radix(rest, 2).map_err(|err| format!("{err:?}"))
    } else {
      input.parse::<u64>().map_err(|err| format!("{err:?}"))
    }
  }

  pub fn parse_numb(&mut self) -> Result<Numb, String> {
    self.skip_trivia();

    // Parses symbols (SYM)
    if let Some('[') = self.peek_one() {
      return self.parse_numb_sym();
    // Parses numbers (U24,I24,F24)
    } else {
      return self.parse_numb_lit();
    }
  }

  pub fn parse_tree(&mut self) -> Result<Tree, String> {
    self.skip_trivia();
    //println!("aaa ||{}", &self.input[self.index..]);
    match self.peek_one() {
      Some('(') => {
        self.advance_one();
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume(")")?;
        Ok(Tree::Con { fst, snd })
      }
      Some('{') => {
        self.advance_one();
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume("}")?;
        Ok(Tree::Dup { fst, snd })
      }
      Some('$') => {
        self.advance_one();
        self.consume("(")?;
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume(")")?;
        Ok(Tree::Opr { fst, snd })
      }
      Some('?') => {
        self.advance_one();
        self.consume("(")?;
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume(")")?;
        Ok(Tree::Swi { fst, snd })
      }
      Some('@') => {
        self.advance_one();
        let nam = self.parse_name()?;
        Ok(Tree::Ref { nam })
      }
      Some('*') => {
        self.advance_one();
        Ok(Tree::Era)
      }
      _ => {
        if let Some(c) = self.peek_one() {
          if "0123456789+-[".contains(c) {
            return Ok(Tree::Num { val: self.parse_numb()? });
          }
        }
        let nam = self.parse_name()?;
        Ok(Tree::Var { nam })
      }
    }
  }

  pub fn parse_net(&mut self) -> Result<Net, String> {
    let root = self.parse_tree()?;
    let mut rbag = Vec::new();
    self.skip_trivia();
    while self.peek_one() == Some('&') {
      self.consume("&")?;
      let par = if let Some('!') = self.peek_one() { self.consume("!")?; true } else { false };
      let fst = self.parse_tree()?;
      self.consume("~")?;
      let snd = self.parse_tree()?;
      rbag.push((par,fst,snd));
      self.skip_trivia();
    }
    Ok(Net { root, rbag })
  }

  pub fn parse_book(&mut self) -> Result<Book, String> {
    let mut defs = BTreeMap::new();
    while !self.is_eof() {
      self.consume("@")?;
      let name = self.parse_name()?;
      self.consume("=")?;
      let net = self.parse_net()?;
      defs.insert(name, net);
    }
    Ok(Book { defs })
  }

  fn try_consume(&mut self, str: &str) -> bool {
    let matches = self.peek_many(str.len()) == Some(str);
    if matches {
      self.advance_many(str.len());
    }
    matches
  }
}

// Stringifier
// -----------

impl Numb {
  pub fn show(&self) -> String {
    let numb = hvm::Numb(self.0);
    match numb.get_typ() {
      hvm::TY_SYM => match numb.get_sym() as hvm::Tag {
        // casts
        hvm::TY_U24 => "[u24]".to_string(),
        hvm::TY_I24 => "[i24]".to_string(),
        hvm::TY_F24 => "[f24]".to_string(),
        // operations
        hvm::OP_ADD => "[+]".to_string(),
        hvm::OP_SUB => "[-]".to_string(),
        hvm::FP_SUB => "[:-]".to_string(),
        hvm::OP_MUL => "[*]".to_string(),
        hvm::OP_DIV => "[/]".to_string(),
        hvm::FP_DIV => "[:/]".to_string(),
        hvm::OP_REM => "[%]".to_string(),
        hvm::FP_REM => "[:%]".to_string(),
        hvm::OP_EQ  => "[=]".to_string(),
        hvm::OP_NEQ => "[!]".to_string(),
        hvm::OP_LT  => "[<]".to_string(),
        hvm::OP_GT  => "[>]".to_string(),
        hvm::OP_AND => "[&]".to_string(),
        hvm::OP_OR  => "[|]".to_string(),
        hvm::OP_XOR => "[^]".to_string(),
        hvm::OP_SHL => "[<<]".to_string(),
        hvm::FP_SHL => "[:<<]".to_string(),
        hvm::OP_SHR => "[>>]".to_string(),
        hvm::FP_SHR => "[:>>]".to_string(),
        _           => "[?]".to_string(),
      }
      hvm::TY_U24 => {
        let val = numb.get_u24();
        format!("{}", val)
      }
      hvm::TY_I24 => {
        let val = numb.get_i24();
        format!("{:+}", val)
      }
      hvm::TY_F24 => {
        let val = numb.get_f24();
        if val.is_infinite() {
          if val.is_sign_positive() {
            format!("+inf")
          } else {
            format!("-inf")
          }
        } else if val.is_nan() {
          format!("+NaN")
        } else {
          format!("{:?}", val)
        }
      }
      _ => {
        let typ = numb.get_typ();
        let val = numb.get_u24();
        format!("[{}0x{:07X}]", match typ {
          hvm::OP_ADD => "+",
          hvm::OP_SUB => "-",
          hvm::FP_SUB => ":-",
          hvm::OP_MUL => "*",
          hvm::OP_DIV => "/",
          hvm::FP_DIV => ":/",
          hvm::OP_REM => "%",
          hvm::FP_REM => ":%",
          hvm::OP_EQ  => "=",
          hvm::OP_NEQ => "!",
          hvm::OP_LT  => "<",
          hvm::OP_GT  => ">",
          hvm::OP_AND => "&",
          hvm::OP_OR  => "|",
          hvm::OP_XOR => "^",
          hvm::OP_SHL => "<<",
          hvm::FP_SHL => ":<<",
          hvm::OP_SHR => ">>",
          hvm::FP_SHR => ":>>",
          _           => "?",
        }, val)
      }
    }
  }
}

impl Tree {
  pub fn show(&self) -> String {
    match self {
      Tree::Var { nam } => nam.to_string(),
      Tree::Ref { nam } => format!("@{}", nam),
      Tree::Era => "*".to_string(),
      Tree::Num { val } => format!("{}", val.show()),
      Tree::Con { fst, snd } => format!("({} {})", fst.show(), snd.show()),
      Tree::Dup { fst, snd } => format!("{{{} {}}}", fst.show(), snd.show()),
      Tree::Opr { fst, snd } => format!("$({} {})", fst.show(), snd.show()),
      Tree::Swi { fst, snd } => format!("?({} {})", fst.show(), snd.show()),
    }
  }
}

impl Net {
  pub fn show(&self) -> String {
    let mut s = self.root.show();
    for (par, fst, snd) in &self.rbag {
      s.push_str(" &");
      s.push_str(if *par { "!" } else { " " });
      s.push_str(&fst.show());
      s.push_str(" ~ ");
      s.push_str(&snd.show());
    }
    s
  }
}

impl Book {
  pub fn show(&self) -> String {
    let mut s = String::new();
    for (name, net) in &self.defs {
      s.push_str("@");
      s.push_str(name);
      s.push_str(" = ");
      s.push_str(&net.show());
      s.push('\n');
    }
    s
  }
}

// Readback
// --------

impl Tree {
  pub fn readback(net: &hvm::GNet, port: hvm::Port, fids: &BTreeMap<hvm::Val, String>) -> Option<Tree> {
    //println!("reading {}", port.show());
    match port.get_tag() {
      hvm::VAR => {
        let got = net.enter(port);
        if got != port {
          return Tree::readback(net, got, fids);
        } else {
          return Some(Tree::Var { nam: format!("v{:x}", port.get_val()) });
        }
      }
      hvm::REF => {
        return Some(Tree::Ref { nam: fids.get(&port.get_val())?.clone() });
      }
      hvm::ERA => {
        return Some(Tree::Era);
      }
      hvm::NUM => {
        return Some(Tree::Num { val: Numb(port.get_val()) });
      }
      hvm::CON => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids)?;
        let snd = Tree::readback(net, pair.get_snd(), fids)?;
        return Some(Tree::Con { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::DUP => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids)?;
        let snd = Tree::readback(net, pair.get_snd(), fids)?;
        return Some(Tree::Dup { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::OPR => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids)?;
        let snd = Tree::readback(net, pair.get_snd(), fids)?;
        return Some(Tree::Opr { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::SWI => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids)?;
        let snd = Tree::readback(net, pair.get_snd(), fids)?;
        return Some(Tree::Swi { fst: Box::new(fst), snd: Box::new(snd) });
      }
      _ => {
        unreachable!()
      }
    }
  }
}

impl Net {
  pub fn readback(net: &hvm::GNet, book: &hvm::Book) -> Option<Net> {
    let mut fids = BTreeMap::new();
    for (fid, def) in book.defs.iter().enumerate() {
      fids.insert(fid as hvm::Val, def.name.clone());
    }
    let root = net.enter(hvm::ROOT);
    let root = Tree::readback(net, root, &fids)?;
    let rbag = Vec::new();
    return Some(Net { root, rbag });
  }
}

// Def Builder
// -----------

impl Tree {
  pub fn build(&self, def: &mut hvm::Def, fids: &BTreeMap<String, hvm::Val>, vars: &mut BTreeMap<String, hvm::Val>) -> hvm::Port {
    match self {
      Tree::Var { nam } => {
        if !vars.contains_key(nam) {
          vars.insert(nam.clone(), vars.len() as hvm::Val);
          def.vars += 1;
        }
        return hvm::Port::new(hvm::VAR, *vars.get(nam).unwrap());
      }
      Tree::Ref { nam } => {
        if let Some(fid) = fids.get(nam) {
          return hvm::Port::new(hvm::REF, *fid);
        } else {
          panic!("Unbound definition: {}", nam);
        }
      }
      Tree::Era => {
        return hvm::Port::new(hvm::ERA, 0);
      }
      Tree::Num { val } => {
        return hvm::Port::new(hvm::NUM, val.0);
      }
      Tree::Con { fst, snd } => {
        let index = def.node.len();
        def.node.push(hvm::Pair(0));
        let p1 = fst.build(def, fids, vars);
        let p2 = snd.build(def, fids, vars);
        def.node[index] = hvm::Pair::new(p1, p2);
        return hvm::Port::new(hvm::CON, index as hvm::Val);
      }
      Tree::Dup { fst, snd } => {
        def.safe = false;
        let index = def.node.len();
        def.node.push(hvm::Pair(0));
        let p1 = fst.build(def, fids, vars);
        let p2 = snd.build(def, fids, vars);
        def.node[index] = hvm::Pair::new(p1, p2);
        return hvm::Port::new(hvm::DUP, index as hvm::Val);
      },
      Tree::Opr { fst, snd } => {
        let index = def.node.len();
        def.node.push(hvm::Pair(0));
        let p1 = fst.build(def, fids, vars);
        let p2 = snd.build(def, fids, vars);
        def.node[index] = hvm::Pair::new(p1, p2);
        return hvm::Port::new(hvm::OPR, index as hvm::Val);
      },
      Tree::Swi { fst, snd } => {
        let index = def.node.len();
        def.node.push(hvm::Pair(0));
        let p1 = fst.build(def, fids, vars);
        let p2 = snd.build(def, fids, vars);
        def.node[index] = hvm::Pair::new(p1, p2);
        return hvm::Port::new(hvm::SWI, index as hvm::Val);
      },
    }
  }

  pub fn direct_dependencies<'name>(&'name self) -> BTreeSet<&'name str> {
    match self {
      Tree::Ref { nam } => BTreeSet::from([nam.as_str()]),
      Tree::Con { fst, snd } => &fst.direct_dependencies() | &snd.direct_dependencies(),
      Tree::Dup { fst, snd } => &fst.direct_dependencies() | &snd.direct_dependencies(),
      Tree::Opr { fst, snd } => &fst.direct_dependencies() | &snd.direct_dependencies(),
      Tree::Swi { fst, snd } => &fst.direct_dependencies() | &snd.direct_dependencies(),
      Tree::Num { val } => BTreeSet::new(),
      Tree::Var { nam } => BTreeSet::new(),
      Tree::Era => BTreeSet::new(),
    }
  }
}

impl Net {
  pub fn build(&self, def: &mut hvm::Def, fids: &BTreeMap<String, hvm::Val>, vars: &mut BTreeMap<String, hvm::Val>) {
    let index = def.node.len();
    def.root = self.root.build(def, fids, vars);
    for (par, fst, snd) in &self.rbag {
      let index = def.rbag.len();
      def.rbag.push(hvm::Pair(0));
      let p1 = fst.build(def, fids, vars);
      let p2 = snd.build(def, fids, vars);
      let rx = hvm::Pair::new(p1, p2);
      let rx = if *par { rx.set_par_flag() } else { rx };
      def.rbag[index] = rx;
    }
  }
}

impl Book {
  pub fn parse(code: &str) -> Result<Self, String> {
    CoreParser::new(code).parse_book()
  }

  pub fn build(&self) -> (hvm::Book, BTreeMap<String, usize>) {
    let mut name_to_fid = BTreeMap::new();
    let mut fid_to_name = BTreeMap::new();
    fid_to_name.insert(0, "main".to_string());
    name_to_fid.insert("main".to_string(), 0);
    for (_i, (name, _)) in self.defs.iter().enumerate() {
      if name != "main" {
        fid_to_name.insert(name_to_fid.len() as hvm::Val, name.clone());
        name_to_fid.insert(name.clone(), name_to_fid.len() as hvm::Val);
      }
    }
    let mut book = hvm::Book { defs: Vec::new() };
    let mut lookup = BTreeMap::new();
    for (fid, name) in &fid_to_name {
      let ast_def = self.defs.get(name).expect("missing `@main` definition");
      let mut def = hvm::Def {
        name: name.clone(),
        safe: true,
        root: hvm::Port(0),
        rbag: vec![],
        node: vec![],
        vars: 0,
      };
      ast_def.build(&mut def, &name_to_fid, &mut BTreeMap::new());
      book.defs.push(def);
      lookup.insert(name.clone(), book.defs.len() - 1);
    }
    self.propagate_safety(&mut book, &lookup);
    return (book, lookup);
  }

  /// Propagate unsafe definitions to those that reference them.
  ///
  /// When calling this function, it is expected that definitions that are directly
  /// unsafe are already marked as such in the `compiled_book`.
  fn propagate_safety(&self, compiled_book: &mut hvm::Book, lookup: &BTreeMap<String, usize>) {
    let rev_dependencies = self.direct_dependencies_reversed();
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    let mut stack: Vec<&str> = Vec::new();

    for (name, _) in self.defs.iter() {
      let def = &compiled_book.defs[lookup[name]];
      if !def.safe {
        stack.push(&name);
      }
    }

    while let Some(curr) = stack.pop() {
      if visited.contains(curr) {
        continue;
      }
      visited.insert(curr);

      let def = &mut compiled_book.defs[lookup[curr]];
      def.safe = false;

      for &next in rev_dependencies[curr].iter() {
        stack.push(next);
      }
    }
  }

  /// Calculates the dependencies of each definition but stores them reversed,
  /// that is, if definition `A` requires `B`, `B: A` is in the return map.
  /// This is used to propagate unsafe definitions to others that depend on them.
  /// 
  /// This solution has linear complexity on the number of definitions in the
  /// book and the number of direct references in each definition, but it also
  /// traverses each definition's trees entirely once. Assuming the tree traversals
  /// are O(h), which they're not with `BTreeSet`s, we have:
  ///
  /// Complexity: O(n*h + m)
  /// - `n` is the number of definitions in the book
  /// - `m` is the number of direct references in each definition
  /// - `h` is the accumulated height of each net's trees
  fn direct_dependencies_reversed<'name>(&'name self) -> BTreeMap<&'name str, BTreeSet<&'name str>> {
    let mut result = BTreeMap::new();
    for (name, _) in self.defs.iter() {
      result.insert(name.as_str(), BTreeSet::new());
    }

    let mut process = |tree: &'name Tree, name: &'name str| {
      for dependency in tree.direct_dependencies() {
        match result.entry(dependency) {
          Entry::Vacant(_) => panic!("global definition depends on undeclared reference"),
          Entry::Occupied(mut entry) => {
            // dependency => name
            entry.get_mut().insert(name);
          },
        }
      }
    };

    for (name, net) in self.defs.iter() {
      process(&net.root, name);
      for (_, r1, r2) in net.rbag.iter() {
        process(r1, name);
        process(r2, name);
      }
    }
    result
  }
}
