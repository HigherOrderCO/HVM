use TSPL::{new_parser, Parser};
use crate::hvm;
use std::collections::BTreeMap;

// Types
// -----

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Tree {
  Var { nam: String },
  Ref { nam: String },
  Era,
  Num { val: hvm::Val },
  Con { fst: Box<Tree>, snd: Box<Tree> },
  Dup { fst: Box<Tree>, snd: Box<Tree> },
  Opr { fst: Box<Tree>, snd: Box<Tree> },
  Swi { fst: Box<Tree>, snd: Box<Tree> },
}

pub type Pair = (Tree, Tree);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Net {
  pub root: Tree,
  pub rbag: Vec<Pair>,
}

pub struct Book {
  pub defs: BTreeMap<String, Net>,
}

// Parser
// ------

new_parser!(CoreParser);

impl<'i> CoreParser<'i> {
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
      Some('<') => {
        self.advance_one();
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume(">")?;
        Ok(Tree::Opr { fst, snd })
      }
      Some('?') => {
        self.advance_one();
        self.consume("<")?;
        let fst = Box::new(self.parse_tree()?);
        self.skip_trivia();
        let snd = Box::new(self.parse_tree()?);
        self.consume(">")?;
        Ok(Tree::Swi { fst, snd })
      }
      Some('@') => {
        self.advance_one();
        let nam = self.parse_name()?;
        Ok(Tree::Ref { nam })
      }
      Some('#') => {
        self.advance_one();
        let val = self.parse_u64()? as u32;
        Ok(Tree::Num { val })
      }
      Some('*') => {
        self.advance_one();
        Ok(Tree::Era)
      }
      _ => {
        let nam = self.parse_name()?;
        Ok(Tree::Var { nam })
      }
    }
  }

  pub fn parse_name(&mut self) -> Result<String, String> {
    let name = self.take_while(|c| c.is_alphanumeric() || "_.$".contains(c));
    if name.is_empty() {
      self.expected("name")
    } else {
      Ok(name.to_string())
    }
  }

  pub fn parse_net(&mut self) -> Result<Net, String> {
    let root = self.parse_tree()?;
    let mut rbag = Vec::new();
    self.skip_trivia();
    while self.peek_one() == Some('&') {
      self.consume("&")?;
      let fst = self.parse_tree()?;
      self.consume("~")?;
      let snd = self.parse_tree()?;
      rbag.push((fst, snd));
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
}

// Stringifier
// -----------

impl Tree {
  pub fn show(&self) -> String {
    match self {
      Tree::Var { nam } => nam.to_string(),
      Tree::Ref { nam } => format!("@{}", nam),
      Tree::Era => "*".to_string(),
      Tree::Num { val } => format!("#{}", val),
      Tree::Con { fst, snd } => format!("({} {})", fst.show(), snd.show()),
      Tree::Dup { fst, snd } => format!("{{{} {}}}", fst.show(), snd.show()),
      Tree::Opr { fst, snd } => format!("<{} {}>", fst.show(), snd.show()),
      Tree::Swi { fst, snd } => format!("?<{} {}>", fst.show(), snd.show()),
    }
  }
}

impl Net {
  pub fn show(&self) -> String {
    let mut s = self.root.show();
    for (fst, snd) in &self.rbag {
      s.push_str(" & ");
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
  pub fn readback(net: &hvm::GNet, port: hvm::Port, fids: &BTreeMap<hvm::Val, String>, vars: &BTreeMap<hvm::Val, String>) -> Option<Tree> {
    match port.get_tag() {
      hvm::VAR => {
        return Some(Tree::Var { nam: vars.get(&port.get_val())?.clone() });
      }
      hvm::REF => {
        return Some(Tree::Ref { nam: fids.get(&port.get_val())?.clone() });
      }
      hvm::ERA => {
        return Some(Tree::Era);
      }
      hvm::NUM => {
        return Some(Tree::Num { val: port.get_val() });  
      }
      hvm::CON => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids, vars)?;
        let snd = Tree::readback(net, pair.get_snd(), fids, vars)?;
        return Some(Tree::Con { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::DUP => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids, vars)?;
        let snd = Tree::readback(net, pair.get_snd(), fids, vars)?;
        return Some(Tree::Dup { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::OPR => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids, vars)?;
        let snd = Tree::readback(net, pair.get_snd(), fids, vars)?;
        return Some(Tree::Opr { fst: Box::new(fst), snd: Box::new(snd) });
      }
      hvm::SWI => {
        let pair = net.node_load(port.get_val() as usize);
        let fst = Tree::readback(net, pair.get_fst(), fids, vars)?;
        let snd = Tree::readback(net, pair.get_snd(), fids, vars)?;
        return Some(Tree::Swi { fst: Box::new(fst), snd: Box::new(snd) }); 
      }
      _ => {
        unreachable!()
      }
    }
  }
}

impl Net {
  // TODO: implement RBag readback
  pub fn readback(net: &hvm::GNet, fids: &BTreeMap<hvm::Val, String>, vars: &BTreeMap<hvm::Val, String>) -> Option<Net> {
    let root = Tree::readback(net, net.node_load(0).get_fst(), fids, vars)?;
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
        return hvm::Port::new(hvm::NUM, *val);
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
  
}

impl Net {
  pub fn build(&self, def: &mut hvm::Def, fids: &BTreeMap<String, hvm::Val>, vars: &mut BTreeMap<String, hvm::Val>) {
    let index = def.node.len();
    def.node.push(hvm::Pair(0));
    let root = self.root.build(def, fids, vars);
    def.node[index] = hvm::Pair::new(root, hvm::Port(0));
    for (fst, snd) in &self.rbag {
      let index = def.rbag.len();
      def.rbag.push(hvm::Pair(0));
      let p1 = fst.build(def, fids, vars);
      let p2 = snd.build(def, fids, vars);
      def.rbag[index] = hvm::Pair::new(p1, p2);
    }
  }
}

impl Book {
  pub fn build(&self) -> hvm::Book {
    let mut fids = BTreeMap::new();
    for (i, (name, _)) in self.defs.iter().enumerate() {
      fids.insert(name.clone(), i as hvm::Val);
    }
    let mut book = hvm::Book { defs: Vec::new() };
    for (name, net) in &self.defs {
      let mut def = hvm::Def {
        name: name.clone(),
        rbag: Vec::new(), 
        node: Vec::new(),
        vars: 0,
      };
      let mut vars = BTreeMap::new();
      net.build(&mut def, &fids, &mut vars);
      book.defs.push(def);
    }
    return book;
  }
}
