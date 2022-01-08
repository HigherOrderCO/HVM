use std::fmt;

// Types
// =====

// Term
// ----
pub enum Term {
  Var {
    name: String,
  },
  Dup {
    nam0: String,
    nam1: String,
    expr: BTerm,
    body: BTerm,
  },
  Let {
    name: String,
    expr: BTerm,
    body: BTerm,
  },
  Lam {
    name: String,
    body: BTerm,
  },
  App {
    func: BTerm,
    argm: BTerm,
  },
  Ctr {
    name: String,
    args: Vec<Term>,
  },
  U32 {
    numb: u32,
  },
  Op2 {
    oper: Oper,
    val0: BTerm,
    val1: BTerm,
  },
}

pub type BTerm = Box<Term>;

pub enum Oper {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  AND,
  OR,
  XOR,
  SHL,
  SHR,
  LTN,
  LTE,
  EQL,
  GTE,
  GTN,
  NEQ,
}

// Rule
// ----

pub struct Rule {
  pub lhs: Term,
  pub rhs: Term,
}

// File
// ----

pub struct File(pub Vec<Rule>);

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
        Self::ADD => "+",
        Self::SUB => "-",
        Self::MUL => "*",
        Self::DIV => "/",
        Self::MOD => "%",
        Self::AND => "&",
        Self::OR => "|",
        Self::XOR => "^",
        Self::SHL => "<<",
        Self::SHR => ">>",
        Self::LTN => "<",
        Self::LTE => "<=",
        Self::EQL => "==",
        Self::GTE => ">=",
        Self::GTN => ">",
        Self::NEQ => "!=",
      }
    )
  }
}

impl fmt::Display for Term {
  // WARN: I think this could overflow, might need to rewrite it to be iterative instead of recursive?
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Var { name } => write!(f, "{}", name),
      Self::Dup {
        nam0,
        nam1,
        expr,
        body,
      } => write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body),
      Self::Let { name, expr, body } => write!(f, "let {} = {}; {}", name, expr, body),
      Self::Lam { name, body } => write!(f, "λ{} {}", name, body),
      Self::App { func, argm } => write!(f, "({} {})", func, argm),
      Self::Ctr { name, args } => write!(
        f,
        "({}{})",
        name,
        args.iter().map(|x| format!(" {}", x)).collect::<String>()
      ),
      Self::U32 { numb } => write!(f, "{}", numb),
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
      self
        .0
        .iter()
        .map(|rule| format!("{}", rule))
        .collect::<Vec<String>>()
        .join("\n")
    )
  }
}
