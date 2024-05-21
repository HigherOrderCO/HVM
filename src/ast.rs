//./hvm.rs//

use crate::hvm;
use std::collections::BTreeMap;
use TSPL::{new_parser, Parser};

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
    pub fn parse_numb_sym(&mut self, flp: bool) -> Result<Numb, String> {
        self.consume("[")?;

        // Parses the symbol
        let num = match self.peek_one().unwrap() {
            '+' => Ok(hvm::Numb::new_sym(hvm::ADD as hvm::Val)),
            '-' => Ok(hvm::Numb::new_sym(hvm::SUB as hvm::Val)),
            '*' => Ok(hvm::Numb::new_sym(hvm::MUL as hvm::Val)),
            '/' => Ok(hvm::Numb::new_sym(hvm::DIV as hvm::Val)),
            '%' => Ok(hvm::Numb::new_sym(hvm::REM as hvm::Val)),
            '=' => Ok(hvm::Numb::new_sym(hvm::EQ as hvm::Val)),
            '!' => Ok(hvm::Numb::new_sym(hvm::NEQ as hvm::Val)),
            '<' => Ok(hvm::Numb::new_sym(hvm::LT as hvm::Val)),
            '>' => Ok(hvm::Numb::new_sym(hvm::GT as hvm::Val)),
            '&' => Ok(hvm::Numb::new_sym(hvm::AND as hvm::Val)),
            '|' => Ok(hvm::Numb::new_sym(hvm::OR as hvm::Val)),
            '^' => Ok(hvm::Numb::new_sym(hvm::XOR as hvm::Val)),
            _ => self.expected("operator symbol"),
        }?;
        self.advance_one();

        // Syntax for partial operations, like `[*2]`
        let num = if "0123456789+-[:".contains(self.peek_one().unwrap_or(' ')) {
            hvm::Numb::partial(num, hvm::Numb(self.parse_numb_lit(false)?.0))
        } else {
            num
        };

        // Closes symbol bracket
        self.consume("]")?;

        // Sets the flip flag, if necessary
        let num = if flp { num.set_flp() } else { num };

        // Returns the symbol
        return Ok(Numb(num.0));
    }

    pub fn parse_numb_lit(&mut self, flp: bool) -> Result<Numb, String> {
        // Parses sign
        let sgn = match self.peek_one() {
            Some('+') => {
                self.consume("+")?;
                Some(1)
            }
            Some('-') => {
                self.consume("-")?;
                Some(-1)
            }
            _ => None,
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
            (Some(s), None) => hvm::Numb::new_i24(s * num as i32),
            (None, Some(f)) => hvm::Numb::new_f24(make_float(num, f)),
            (None, None) => hvm::Numb::new_u24(num),
        };

        // Sets the flip flag, if necessary
        let num = if flp { num.set_flp() } else { num };

        // Return the parsed number
        Ok(Numb(num.0))
    }

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
            self.parse_numb_sym(flp)
        // Parses numbers (U24,I24,F24)
        } else {
            self.parse_numb_lit(flp)
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
                    if "0123456789+-[:".contains(c) {
                        return Ok(Tree::Num {
                            val: self.parse_numb()?,
                        });
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
            let par = if let Some('!') = self.peek_one() {
                self.consume("!")?;
                true
            } else {
                false
            };
            let fst = self.parse_tree()?;
            self.consume("~")?;
            let snd = self.parse_tree()?;
            rbag.push((par, fst, snd));
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
            hvm::SYM => match numb.get_sym() as hvm::Tag {
                hvm::ADD => "[+]".to_string(),
                hvm::SUB => "[-]".to_string(),
                hvm::MUL => "[*]".to_string(),
                hvm::DIV => "[/]".to_string(),
                hvm::REM => "[%]".to_string(),
                hvm::EQ => "[=]".to_string(),
                hvm::LT => "[<]".to_string(),
                hvm::GT => "[>]".to_string(),
                hvm::AND => "[&]".to_string(),
                hvm::OR => "[|]".to_string(),
                hvm::XOR => "[^]".to_string(),
                _ => "[?]".to_string(),
            },
            hvm::U24 => {
                let val = numb.get_u24();
                if numb.get_flp() {
                    format!(":{}", val)
                } else {
                    format!("{}", val)
                }
            }
            hvm::I24 => {
                let val = numb.get_i24();
                let sng = if val < 0 { "-" } else { "+" };
                if numb.get_flp() {
                    format!(":{}{}", sng, val.abs())
                } else {
                    format!("{}{}", sng, val.abs())
                }
            }
            hvm::F24 => {
                let val = numb.get_f24();
                if numb.get_flp() {
                    format!(":{:.3}", val)
                } else {
                    format!("{:.3}", val)
                }
            }
            _ => {
                let typ = numb.get_typ();
                let val = numb.get_u24();
                format!(
                    "[{}{:07X}]",
                    match typ {
                        hvm::ADD => "+",
                        hvm::SUB => "-",
                        hvm::MUL => "*",
                        hvm::DIV => "/",
                        hvm::REM => "%",
                        hvm::EQ => "=",
                        hvm::NEQ => "!",
                        hvm::LT => "<",
                        hvm::GT => ">",
                        hvm::AND => "&",
                        hvm::OR => "|",
                        hvm::XOR => "^",
                        _ => "?",
                    },
                    val
                )
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
            Tree::Num { val } => val.show(),
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
            s.push_str(" & ");
            s.push_str(if *par { "!" } else { "" });
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
            s.push('@');
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
    pub fn readback(
        net: &hvm::GNet,
        port: hvm::Port,
        fids: &BTreeMap<hvm::Val, String>,
    ) -> Option<Tree> {
        //println!("reading {}", port.show());
        match port.get_tag() {
            hvm::VAR => {
                let got = net.enter(port);
                if got != port {
                    Tree::readback(net, got, fids)
                } else {
                    Some(Tree::Var {
                        nam: format!("v{:x}", port.get_val()),
                    })
                }
            }
            hvm::REF => Some(Tree::Ref {
                nam: fids.get(&port.get_val())?.clone(),
            }),
            hvm::ERA => Some(Tree::Era),
            hvm::NUM => Some(Tree::Num {
                val: Numb(port.get_val()),
            }),
            hvm::CON => {
                let pair = net.node_load(port.get_val() as usize);
                let fst = Tree::readback(net, pair.get_fst(), fids)?;
                let snd = Tree::readback(net, pair.get_snd(), fids)?;
                Some(Tree::Con {
                    fst: Box::new(fst),
                    snd: Box::new(snd),
                })
            }
            hvm::DUP => {
                let pair = net.node_load(port.get_val() as usize);
                let fst = Tree::readback(net, pair.get_fst(), fids)?;
                let snd = Tree::readback(net, pair.get_snd(), fids)?;
                Some(Tree::Dup {
                    fst: Box::new(fst),
                    snd: Box::new(snd),
                })
            }
            hvm::OPR => {
                let pair = net.node_load(port.get_val() as usize);
                let fst = Tree::readback(net, pair.get_fst(), fids)?;
                let snd = Tree::readback(net, pair.get_snd(), fids)?;
                Some(Tree::Opr {
                    fst: Box::new(fst),
                    snd: Box::new(snd),
                })
            }
            hvm::SWI => {
                let pair = net.node_load(port.get_val() as usize);
                let fst = Tree::readback(net, pair.get_fst(), fids)?;
                let snd = Tree::readback(net, pair.get_snd(), fids)?;
                Some(Tree::Swi {
                    fst: Box::new(fst),
                    snd: Box::new(snd),
                })
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
        Some(Net { root, rbag })
    }
}

// Def Builder
// -----------

impl Tree {
    pub fn build(
        &self,
        def: &mut hvm::Def,
        fids: &BTreeMap<String, hvm::Val>,
        vars: &mut BTreeMap<String, hvm::Val>,
    ) -> hvm::Port {
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
                    hvm::Port::new(hvm::REF, *fid)
                } else {
                    panic!("Unbound definition: {}", nam);
                }
            }
            Tree::Era => hvm::Port::new(hvm::ERA, 0),
            Tree::Num { val } => hvm::Port::new(hvm::NUM, val.0),
            Tree::Con { fst, snd } => {
                let index = def.node.len();
                def.node.push(hvm::Pair(0));
                let p1 = fst.build(def, fids, vars);
                let p2 = snd.build(def, fids, vars);
                def.node[index] = hvm::Pair::new(p1, p2);
                hvm::Port::new(hvm::CON, index as hvm::Val)
            }
            Tree::Dup { fst, snd } => {
                def.safe = false;
                let index = def.node.len();
                def.node.push(hvm::Pair(0));
                let p1 = fst.build(def, fids, vars);
                let p2 = snd.build(def, fids, vars);
                def.node[index] = hvm::Pair::new(p1, p2);
                hvm::Port::new(hvm::DUP, index as hvm::Val)
            }
            Tree::Opr { fst, snd } => {
                let index = def.node.len();
                def.node.push(hvm::Pair(0));
                let p1 = fst.build(def, fids, vars);
                let p2 = snd.build(def, fids, vars);
                def.node[index] = hvm::Pair::new(p1, p2);
                hvm::Port::new(hvm::OPR, index as hvm::Val)
            }
            Tree::Swi { fst, snd } => {
                let index = def.node.len();
                def.node.push(hvm::Pair(0));
                let p1 = fst.build(def, fids, vars);
                let p2 = snd.build(def, fids, vars);
                def.node[index] = hvm::Pair::new(p1, p2);
                hvm::Port::new(hvm::SWI, index as hvm::Val)
            }
        }
    }
}

impl Net {
    pub fn build(
        &self,
        def: &mut hvm::Def,
        fids: &BTreeMap<String, hvm::Val>,
        vars: &mut BTreeMap<String, hvm::Val>,
    ) {
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
        book
    }
}
