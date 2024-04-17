use TSPL::{new_parser, Parser};
use crate::hvm::{*};
use std::collections::BTreeMap;

// Types
// -----

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Tree {
  Var { nam: String },
  Ref { nam: String },
  Era,
  Num { val: Val },
  Con { fst: Box<Tree>, snd: Box<Tree> },
  Dup { fst: Box<Tree>, snd: Box<Tree> },
  Op2 { fst: Box<Tree>, snd: Box<Tree> },
  Swi { fst: Box<Tree>, snd: Box<Tree> },
}

type Pair = (Tree, Tree);

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
        Ok(Tree::Op2 { fst, snd })
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
      Err("Expected a name".to_string())
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
      Tree::Op2 { fst, snd } => format!("<{} {}>", fst.show(), snd.show()),
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
