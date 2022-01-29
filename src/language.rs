use crate::parser;
use std::fmt;
use std::{
  collections::{BTreeMap, HashMap},
  fmt::format,
  u64,
};

// Types
// =====

// Term
// ----

#[derive(Clone, Debug)]
pub enum Term {
  Var { name: String },
  Dup { nam0: String, nam1: String, expr: BTerm, body: BTerm },
  Let { name: String, expr: BTerm, body: BTerm },
  Lam { name: String, body: BTerm },
  App { func: BTerm, argm: BTerm },
  Ctr { name: String, args: Vec<BTerm> },
  U32 { numb: u32 },
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
  Ltn,
  Lte,
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
        Self::Ltn => "<",
        Self::Lte => "<=",
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
    match self {
      Self::Var { name } => write!(f, "{}", name),
      Self::Dup { nam0, nam1, expr, body } => {
        write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body)
      }
      Self::Let { name, expr, body } => write!(f, "let {} = {}; {}", name, expr, body),
      Self::Lam { name, body } => write!(f, "λ{} {}", name, body),
      Self::App { func, argm } => write!(f, "({} {})", func, argm),
      Self::Ctr { name, args } => {
        write!(f, "({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
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
      self.rules.iter().map(|rule| format!("{}", rule)).collect::<Vec<String>>().join("\n")
    )
  }
}

// Parser
// ======

pub fn parse_let(state: parser::State) -> parser::Answer<Option<BTerm>> {
  return parser::guard(
    parser::text_parser("let "),
    Box::new(|state| {
      let (state, spk1) = parser::consume("let ", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, spk1) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, skp2) = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Let { name, expr, body })))
    }),
    state,
  );
}

pub fn parse_dup(state: parser::State) -> parser::Answer<Option<BTerm>> {
  return parser::guard(
    parser::text_parser("dup "),
    Box::new(|state| {
      let (state, spk1) = parser::consume("dup ", state)?;
      let (state, nam0) = parser::name1(state)?;
      let (state, nam1) = parser::name1(state)?;
      let (state, spk1) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, skp2) = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Dup { nam0, nam1, expr, body })))
    }),
    state,
  );
}

pub fn parse_lam(state: parser::State) -> parser::Answer<Option<BTerm>> {
  return parser::guard(
    parser::text_parser("λ"),
    Box::new(|state| {
      let (state, skp0) = parser::text("λ", state)?;
      let (state, name) = parser::name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  );
}

// TODO: move this to parse_lam to avoid duplicated code
pub fn parse_lam_ugly(state: parser::State) -> parser::Answer<Option<BTerm>> {
  return parser::guard(
    parser::text_parser("@"),
    Box::new(|state| {
      let (state, skp0) = parser::text("@", state)?;
      let (state, name) = parser::name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  );
}

pub fn parse_app(state: parser::State) -> parser::Answer<Option<BTerm>> {
  return parser::guard(
    parser::text_parser("("),
    Box::new(|state| {
      parser::list(
        parser::text_parser("("),
        parser::text_parser(""),
        parser::text_parser(")"),
        Box::new(parse_term),
        Box::new(|args| {
          if !args.is_empty() {
            args.into_iter().reduce(|a, b| Box::new(Term::App { func: a, argm: b })).unwrap()
          } else {
            Box::new(Term::U32 { numb: 0 })
          }
        }),
        state,
      )
    }),
    state,
  );
}

pub fn parse_ctr(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, open) = parser::text("(", state)?;
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('A'..='Z').contains(&head) || head == '.'))
    }),
    Box::new(|state| {
      let (state, open) = parser::text("(", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, args) = if open {
        parser::until(parser::text_parser(")"), Box::new(parse_term), state)?
      } else {
        (state, Vec::new())
      };
      Ok((state, Box::new(Term::Ctr { name, args })))
    }),
    state,
  )
}

pub fn parse_u32(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('0'..='9').contains(&head)))
    }),
    Box::new(|state| {
      let (state, numb) = parser::name1(state)?;
      if !numb.is_empty() {
        Ok((state, Box::new(Term::U32 { numb: numb.parse::<u32>().unwrap() })))
      } else {
        Ok((state, Box::new(Term::U32 { numb: 0 })))
      }
    }),
    state,
  )
}

pub fn parse_op2(state: parser::State) -> parser::Answer<Option<BTerm>> {
  fn is_op_char(chr: char) -> bool {
    false
      || chr == '+'
      || chr == '-'
      || chr == '*'
      || chr == '/'
      || chr == '%'
      || chr == '&'
      || chr == '|'
      || chr == '^'
      || chr == '<'
      || chr == '>'
      || chr == '='
      || chr == '!'
  }
  fn parse_oper(state: parser::State) -> parser::Answer<Oper> {
    fn op<'a>(symbol: &'static str, oper: Oper) -> parser::Parser<'a, Option<Oper>> {
      Box::new(move |state| {
        let (state, done) = parser::text(symbol, state)?;
        Ok((state, if done { Some(oper) } else { None }))
      })
    }
    parser::grammar(
      "Oper",
      &[
        op("+", Oper::Add),
        op("-", Oper::Sub),
        op("*", Oper::Mul),
        op("/", Oper::Div),
        op("%", Oper::Mod),
        op("&", Oper::And),
        op("|", Oper::Or),
        op("^", Oper::Xor),
        op("<<", Oper::Shl),
        op(">>", Oper::Shr),
        op("<", Oper::Ltn),
        op("<=", Oper::Lte),
        op("==", Oper::Eql),
        op(">=", Oper::Gte),
        op(">", Oper::Gtn),
        op("!=", Oper::Neq),
      ],
      state,
    )
  }
  parser::guard(
    Box::new(|state| {
      let (state, open) = parser::text("(", state)?;
      let (state, head) = parser::get_char(state)?;
      Ok((state, open && is_op_char(head)))
    }),
    Box::new(|state| {
      let (state, skp0) = parser::text("(", state)?;
      let (state, oper) = parse_oper(state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, skp1) = parser::text(")", state)?;
      Ok((state, Box::new(Term::Op2 { oper, val0, val1 })))
    }),
    state,
  )
}

pub fn parse_var(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('a'..='z').contains(&head) || head == '_'))
    }),
    Box::new(|state| {
      let (state, name) = parser::name(state)?;
      Ok((state, Box::new(Term::Var { name })))
    }),
    state,
  )
}

pub fn parse_term(state: parser::State) -> parser::Answer<BTerm> {
  parser::grammar(
    "Term",
    &[
      Box::new(parse_let),
      Box::new(parse_dup),
      Box::new(parse_lam),
      Box::new(parse_lam_ugly),
      Box::new(parse_ctr),
      Box::new(parse_op2),
      Box::new(parse_app),
      Box::new(parse_u32),
      Box::new(parse_var),
      Box::new(|state| Ok((state, None))),
    ],
    state,
  )
}

pub fn parse_rule(state: parser::State) -> parser::Answer<Option<Rule>> {
  return parser::guard(
    parser::text_parser(""),
    Box::new(|state| {
      let (state, lhs) = parse_term(state)?;
      let (state, spk) = parser::consume("=", state)?;
      let (state, rhs) = parse_term(state)?;
      Ok((state, Rule { lhs, rhs }))
    }),
    state,
  );
}

pub fn parse_file(state: parser::State) -> parser::Answer<File> {
  let mut rules = Vec::new();
  let mut state = state;
  loop {
    let (new_state, done) = parser::done(state)?;
    if done {
      break;
    }
    let (new_state, rule) = parse_rule(new_state)?;
    if let Some(rule) = rule {
      rules.push(rule);
    } else {
      return parser::expected("definition", 1, state);
    }
    state = new_state;
  }

  Ok((state, File { rules }))
}

pub fn read_term(code: &str) -> Box<Term> {
  parser::read(Box::new(parse_term), code)
}

pub fn read_file(code: &str) -> File {
  parser::read(Box::new(parse_file), code)
}

pub fn read_rule(code: &str) -> Option<Rule> {
  parser::read(Box::new(parse_rule), code)
}
