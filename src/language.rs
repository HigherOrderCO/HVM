use std::fmt;


// Types
// =====

// Term
// ----

#[derive(Clone, Debug)]
pub enum Term {
  Var { name: String }, // TODO: add `global: bool`
  Dup { nam0: String, nam1: String, expr: BTerm, body: BTerm },
  Let { name: String, expr: BTerm, body: BTerm },
  Lam { name: String, body: BTerm },
  App { func: BTerm, argm: BTerm },
  Ctr { name: String, args: Vec<BTerm> },
  Num { numb: u64 },
  Op2 { oper: Oper, val0: BTerm, val1: BTerm },
}

pub type BTerm = Box<Term>;

#[derive(Clone, Copy, Debug)]
pub enum Oper {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  Lte,
  Ltn,
  Eql,
  Gte,
  Gtn,
  Neq,
}

// Rule
// ----

#[derive(Clone, Debug)]
pub struct Rule {
  pub lhs: BTerm,
  pub rhs: BTerm,
}

// File
// ----

pub struct File {
  pub rules: Vec<Rule>,
}

// Stringifier
// ===========

// Term
// ----

impl fmt::Display for Oper {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "{}",
      match self {
        Self::Add => "+",
        Self::Sub => "-",
        Self::Mul => "*",
        Self::Div => "/",
        Self::Mod => "%",
        Self::And => "&",
        Self::Or => "|",
        Self::Xor => "^",
        Self::Shl => "<<",
        Self::Shr => ">>",
        Self::Lte => "<=",
        Self::Ltn => "<",
        Self::Eql => "==",
        Self::Gte => ">=",
        Self::Gtn => ">",
        Self::Neq => "!=",
      }
    )
  }
}

impl fmt::Display for Term {
  // WARN: I think this could overflow, might need to rewrite it to be iterative instead of recursive?
  // NOTE: Another issue is complexity. This function is O(N^2). Should use ropes to be linear.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fn lst_sugar(term: &Term) -> Option<String> {
      fn go(term: &Term, text: &mut String, fst: bool) -> Option<()> {
        if let Term::Ctr { name, args } = term {
          if name == "List.cons" && args.len() == 2 {
            if !fst {
              text.push_str(", ");
            }
            text.push_str(&format!("{}", args[0]));
            go(&args[1], text, false)?;
            return Some(());
          }
          if name == "List.nil" && args.is_empty() {
            return Some(());
          }
        }
        None
      }
      let mut result = String::new();
      result.push('[');
      go(term, &mut result, true)?;
      result.push(']');
      Some(result)
    }

    fn str_sugar(term: &Term) -> Option<String> {
      fn go(term: &Term, text: &mut String) -> Option<()> {
        if let Term::Ctr { name, args } = term {
          if name == "String.cons" && args.len() == 2 {
            if let Term::Num { numb } = *args[0] {
              text.push(std::char::from_u32(numb as u32)?);
              go(&args[1], text)?;
            }
            return Some(());
          }
          if name == "String.nil" && args.is_empty() {
            return Some(());
          }
        }
        None
      }
      let mut result = String::new();
      result.push('"');
      go(term, &mut result)?;
      result.push('"');
      Some(result)
    }
    match self {
      Self::Var { name } => write!(f, "{}", name),
      Self::Dup { nam0, nam1, expr, body } => {
        write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body)
      }
      Self::Let { name, expr, body } => write!(f, "let {} = {}; {}", name, expr, body),
      Self::Lam { name, body } => write!(f, "λ{} {}", name, body),
      Self::App { func, argm } => {
        let mut args = vec![argm];
        let mut expr = func;
        while let Self::App { func, argm } = &**expr {
          args.push(argm);
          expr = func;
        }
        args.reverse();
        write!(f, "({} {})", expr, args.iter().map(|x| format!("{}",x)).collect::<Vec<String>>().join(" "))
      },
      Self::Ctr { name, args } => {
        // Ctr sugars
        let sugars = [str_sugar, lst_sugar];
        for sugar in sugars {
          if let Some(term) = sugar(self) {
            return write!(f, "{}", term);
          }
        }

        write!(f, "({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
      Self::Num { numb } => write!(f, "{}", numb),
      Self::Op2 { oper, val0, val1 } => write!(f, "({} {} {})", oper, val0, val1),
    }
  }
}

// Rule
// ----

impl fmt::Display for Rule {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{} = {}", self.lhs, self.rhs)
  }
}

// File
// ----

impl fmt::Display for File {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "{}",
      self.rules.iter().map(|rule| format!("{}", rule)).collect::<Vec<String>>().join("\n")
    )
  }
}

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub hvm_syntax); // synthesized by LALRPOP

pub fn read_term(code: &str) -> Result<Box<Term>, String> {
  hvm_syntax::RRecTermParser::new().parse(code).map_err(|err| format!("{:?}",err))
}

pub fn read_file(code: &str) -> Result<File, String> {
  hvm_syntax::FileParser::new().parse(code).map_err(|err| format!("{:?}",err))
}

#[allow(dead_code)]
pub fn read_rule(code: &str) -> Result<Rule, String> {
  hvm_syntax::RuleParser::new().parse(code).map_err(|err| format!("{:?}",err))
}
