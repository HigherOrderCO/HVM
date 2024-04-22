//./hvm.rs//

use TSPL::{new_parser, Parser};
use crate::hvm;
use std::collections::BTreeMap;

// Types
// -----

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Numb(u32);

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
  pub fn parse_numb(&mut self) -> Result<Numb, String> {
    // Parses flip flag
    let flp = if let Some(':') = self.peek_one() {
      self.consume(":")?;
      true
    } else {
      false
    };

    // Parses symbols (SYM)
    if let Some('[') = self.peek_one() {
      self.consume("[")?;

      // Parses the symbol
      let num = hvm::Numb(match self.peek_one().unwrap() {
        ' ' => Ok(0x00), '+' => Ok(0x10), '-' => Ok(0x20), '*' => Ok(0x30),
        '/' => Ok(0x40), '%' => Ok(0x50), '=' => Ok(0x60), '!' => Ok(0x70),
        '<' => Ok(0x80), '>' => Ok(0x90), '&' => Ok(0xA0), '|' => Ok(0xB0),
        '^' => Ok(0xC0), 'L' => Ok(0xD0), 'R' => Ok(0xE0), 'X' => Ok(0xF0),
        _   => self.expected("operator symbol"),
      }?);
      self.advance_one();

      // Sets the flip flag, if necessary
      let num = if flp { num.set_flp() } else { num };

      // Closes symbol bracket
      self.consume("]")?;

      // Returns the symbol
      return Ok(Numb(num.0));

    // Parses numbers (U24,I24,F24)
    } else {
      // Parses sign
      let sgn = match self.peek_one() {
        Some('+') => { self.consume("+")?; Some(1) }
        Some('-') => { self.consume("-")?; Some(-1) }
        _         => None,
      };

      // Parses main value 
      let num = self.parse_u64()? as u32;

      // Parses frac value (Float type)
      let fra = if let Some('.') = self.peek_one() {
        self.consume(".")?;
        Some(self.parse_u64()? as u32)
      } else {
        None
      };

      // Creates a float from value and fraction
      fn make_float(num: u32, fra: u32) -> f32 {
        num as f32 + fra as f32 / 10f32.powi(fra.to_string().len() as i32)
      }

      // Gets the numeric bit representation
      let num = match (sgn, fra) {
        (Some(s), Some(f)) => hvm::Numb::new_f24(s as f32 * make_float(num, f)),
        (Some(s), None   ) => hvm::Numb::new_i24(s as i32 * num as i32),
        (None   , Some(f)) => hvm::Numb::new_f24(make_float(num, f)),
        (None   , None   ) => hvm::Numb::new_u24(num),
      };

      // Sets the flip flag, if necessary
      let num = if flp { num.set_flp() } else { num };

      // Return the parsed number
      return Ok(Numb(num.0));
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
          if c.is_ascii_digit() || c == '+' || c == '-' || c == '[' || c == ':' {
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

impl Numb {
  pub fn show(&self) -> String {
    let numb = hvm::Numb(self.0);
    match numb.get_typ() {
      hvm::SYM => match numb.get_sym() {
        0x0 => "[X]".to_string(),
        0x1 => "[+]".to_string(),
        0x2 => "[-]".to_string(),
        0x3 => "[*]".to_string(),
        0x4 => "[/]".to_string(),
        0x5 => "[%]".to_string(),
        0x6 => "[=]".to_string(),
        0x7 => "[!]".to_string(),
        0x8 => "[<]".to_string(),
        0x9 => "[>]".to_string(),
        0xA => "[&]".to_string(),
        0xB => "[|]".to_string(),
        0xC => "[^]".to_string(),
        0xD => "[L]".to_string(),
        0xE => "[R]".to_string(),
        _   => "[?]".to_string(),
      },
      hvm::U24 => {
        let val = numb.get_u24();
        if numb.get_flp() {
          format!(":{}", val)
        } else {
          format!("{}", val)
        }
      },
      hvm::I24 => {
        let val = numb.get_i24();
        let sng = if val < 0 { "-" } else { "+" };
        if numb.get_flp() {
          format!(":{}{}", sng, val.abs())  
        } else {
          format!("{}{}", sng, val.abs())
        }
      },
      hvm::F24 => {
        let val = numb.get_f24();
        if numb.get_flp() {
          format!(":{:.3}", val)
        } else {
          format!("{:.3}", val)
        }
      },
      _ => "?".to_string(),
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
        return Some(Tree::Num { val: Numb(port.get_val()) });  
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
  // FIXME: should get root correctly
  //pub fn readback(net: &hvm::GNet, fids: &BTreeMap<hvm::Val, String>, vars: &BTreeMap<hvm::Val, String>) -> Option<Net> {
    //let root = Tree::readback(net, net.node_load(0).get_fst(), fids, vars)?;
    //let rbag = Vec::new();
    //return Some(Net { root, rbag });
  //}
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
}

impl Net {
  pub fn build(&self, def: &mut hvm::Def, fids: &BTreeMap<String, hvm::Val>, vars: &mut BTreeMap<String, hvm::Val>) {
    let index = def.node.len();
    def.root = self.root.build(def, fids, vars);
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
  pub fn parse(code: &str) -> Result<Self, String> {
    CoreParser::new(code).parse_book()
  }

  pub fn build(&self) -> hvm::Book {
    let mut name_to_fid = BTreeMap::new();
    let mut fid_to_name = BTreeMap::new();
    fid_to_name.insert(0, "main".to_string());
    name_to_fid.insert("main".to_string(), 0);
    for (i, (name, _)) in self.defs.iter().enumerate() {
      if name != "main" {
        fid_to_name.insert(name_to_fid.len() as hvm::Val, name.clone());
        name_to_fid.insert(name.clone(), name_to_fid.len() as hvm::Val);
      }
    }
    let mut book = hvm::Book { defs: Vec::new() };
    for (fid, name) in &fid_to_name {
      if let Some(ast_def) = self.defs.get(name) {
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
      }
    }
    return book;
  }
}
