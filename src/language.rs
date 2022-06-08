use crate::parser;
use std::fmt::{self, Write};

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
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// returns if the resugaring succeeded
    fn lst_sugar(f: &mut fmt::Formatter<'_>, term: &Term) -> Result<bool, fmt::Error> {
      let mut buffer = String::new();
      let mut fst = true;
      let mut tm = term;
      while let Term::Ctr { name, args } = tm {
        if name == "Cons" && args.len() == 2 {
          if fst {
            fst = false;
            write!(buffer, "{}", args[0])?;
          } else {
            write!(buffer, ", {}", args[0])?;
          }
          tm = &args[1];
        } else if name == "Nil" && args.is_empty() {
          write!(f, "[{}]", buffer)?;
          return Ok(true);
        } else {
          break;
        }
      }
      Ok(false)
    }

    fn str_sugar(f: &mut fmt::Formatter<'_>, term: &Term) -> Result<bool, fmt::Error> {
      let mut buffer = String::new();
      let mut tm = term;
      while let Term::Ctr { name, args } = tm {
        if name == "StrCons" && args.len() == 2 {
          if let Term::U32 { numb } = &*args[0] {
            write!(buffer, "{}", char::try_from(*numb).map_err(|_| fmt::Error)?)?;
            tm = &args[1];
          } else {
            return Ok(false);
          }
        } else if name == "StrNil" && args.is_empty() {
          write!(f, "\"{}\"", buffer.escape_default())?;
          return Ok(true);
        } else {
          break;
        }
      }
      Ok(false)
    }

    match self {
      Self::Var { name } => write!(f, "{}", name),
      Self::Dup { nam0, nam1, expr, body } => {
        write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body)
      }
      Self::Let { name, expr, body } => write!(f, "let {} = {}; {}", name, expr, body),
      Self::Lam { name, body } => write!(f, "λ{} {}", name, body),
      Self::App { func, argm } => write!(f, "({} {})", func, argm),
      Self::Ctr { name, args } => {
        // Ctr sugars
        let sugars = [str_sugar, lst_sugar];
        for sugar in sugars {
          if sugar(f, self)? {
            return Ok(());
          }
        }

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
      let (state, _) = parser::consume("let ", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, _) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _) = parser::text(";", state)?;
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
      let (state, _) = parser::consume("dup ", state)?;
      let (state, nam0) = parser::name1(state)?;
      let (state, nam1) = parser::name1(state)?;
      let (state, _) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _) = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Dup { nam0, nam1, expr, body })))
    }),
    state,
  );
}

pub fn parse_lam(state: parser::State) -> parser::Answer<Option<BTerm>> {
  let parse_symbol =
    |x| parser::parser_or(&[parser::text_parser("λ"), parser::text_parser("@")], x);
  parser::guard(
    Box::new(parse_symbol),
    Box::new(move |state| {
      let (state, _) = parse_symbol(state)?;
      let (state, name) = parser::name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  )
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
      let (state, _) = parser::text("(", state)?;
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('A'..='Z').contains(&head)))
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
    matches!(chr, '+' | '-' | '*' | '/' | '%' | '&' | '|' | '^' | '<' | '>' | '=' | '!')
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
        op("<=", Oper::Lte),
        op("<", Oper::Ltn),
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
      let (state, _) = parser::text("(", state)?;
      let (state, oper) = parse_oper(state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, _) = parser::text(")", state)?;
      Ok((state, Box::new(Term::Op2 { oper, val0, val1 })))
    }),
    state,
  )
}

pub fn parse_var(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('a'..='z').contains(&head) || head == '_' || head == '$'))
    }),
    Box::new(|state| {
      let (state, name) = parser::name(state)?;
      Ok((state, Box::new(Term::Var { name })))
    }),
    state,
  )
}

pub fn parse_chr_sugar(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, head == '\''))
    }),
    Box::new(|state| {
      let (state, _) = parser::text("'", state)?;
      if let Some(c) = parser::head(state) {
        let state = parser::tail(state);
        let (state, _) = parser::text("'", state)?;
        Ok((state, Box::new(Term::U32 { numb: c as u32 })))
      } else {
        parser::expected("character", 1, state)
      }
    }),
    state,
  )
}

// TODO: unicode escape/support
pub fn parse_str_sugar(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, head == '"' || head == '`'))
    }),
    Box::new(|state| {
      let delim = parser::head(state).unwrap_or('\0');
      let state = parser::tail(state);
      let mut chars: Vec<char> = Vec::new();
      let mut state = state;
      loop {
        if let Some(next) = parser::head(state) {
          if next == delim || next == '\0' {
            state = parser::tail(state);
            break;
          } else if next == '\\' {
            let st = parser::tail(state);
            if let Some(next) = parser::head(st) {
              match next {
                't' => chars.push('\t'),
                'r' => chars.push('\r'),
                'n' => chars.push('\n'),
                '\'' => chars.push('\''),
                '"' => chars.push('"'),
                '\\' => chars.push('\\'),
                _ => return parser::expected("escape character", 1, st),
              }
              state = parser::tail(st);
            } else {
              return parser::expected("escape character", 0, state);
            }
          } else {
            chars.push(next);
            state = parser::tail(state);
          }
        } else {
          return parser::expected("characters", 0, state);
        }
      }

      let empty = Term::Ctr { name: "StrNil".to_string(), args: Vec::new() };
      let list = Box::new(chars.iter().rfold(empty, |t, h| Term::Ctr {
        name: "StrCons".to_string(),
        args: vec![Box::new(Term::U32 { numb: *h as u32 }), Box::new(t)],
      }));
      Ok((state, list))
    }),
    state,
  )
}

pub fn parse_lst_sugar(state: parser::State) -> parser::Answer<Option<BTerm>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, head == '['))
    }),
    Box::new(|state| {
      let (state, _head) = parser::text("[", state)?;
      // let mut elems: Vec<Box<Term>> = Vec::new();
      let state = state;
      let (state, elems) = parser::until(
        Box::new(|x| parser::text("]", x)),
        Box::new(|x| {
          let (state, term) = parse_term(x)?;
          let (state, _) = parser::maybe(Box::new(|x| parser::text(",", x)), state)?;
          Ok((state, term))
        }),
        state,
      )?;
      let empty = Term::Ctr { name: "Nil".to_string(), args: Vec::new() };
      let list = Box::new(elems.iter().rfold(empty, |t, h| Term::Ctr {
        name: "Cons".to_string(),
        args: vec![h.clone(), Box::new(t)],
      }));
      Ok((state, list))
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
      Box::new(parse_ctr),
      Box::new(parse_op2),
      Box::new(parse_app),
      Box::new(parse_u32),
      Box::new(parse_chr_sugar),
      Box::new(parse_str_sugar),
      Box::new(parse_lst_sugar),
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
      let (state, _) = parser::consume("=", state)?;
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

pub fn read_term(code: &str) -> Result<Box<Term>, String> {
  parser::read(Box::new(parse_term), code)
}

pub fn read_file(code: &str) -> Result<File, String> {
  parser::read(Box::new(parse_file), code)
}

#[allow(dead_code)]
pub fn read_rule(code: &str) -> Result<Option<Rule>, String> {
  parser::read(Box::new(parse_rule), code)
}
