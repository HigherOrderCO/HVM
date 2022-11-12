use crate::language::parser;
use crate::runtime::data::u60;
use crate::runtime::data::f60;

// Types
// =====

// Term
// ----

#[derive(Clone, Debug)]
pub enum Term {
  Var { name: String }, // TODO: add `global: bool`
  Dup { nam0: String, nam1: String, expr: Box<Term>, body: Box<Term> },
  Sup { val0: Box<Term>, val1: Box<Term> },
  Let { name: String, expr: Box<Term>, body: Box<Term> },
  Lam { name: String, body: Box<Term> },
  App { func: Box<Term>, argm: Box<Term> },
  Ctr { name: String, args: Vec<Box<Term>> },
  U6O { numb: u64 },
  F6O { numb: u64 },
  Op2 { oper: Oper, val0: Box<Term>, val1: Box<Term> },
}

#[derive(Clone, Copy, Debug)]
pub enum Oper {
  Add, Sub, Mul, Div,
  Mod, And, Or,  Xor,
  Shl, Shr, Lte, Ltn,
  Eql, Gte, Gtn, Neq,
}

// Rule
// ----

#[derive(Clone, Debug)]
pub struct Rule {
  pub lhs: Box<Term>,
  pub rhs: Box<Term>,
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

impl std::fmt::Display for Oper {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", match self {
      Self::Add => "+",
      Self::Sub => "-",
      Self::Mul => "*",
      Self::Div => "/",
      Self::Mod => "%",
      Self::And => "&",
      Self::Or  => "|",
      Self::Xor => "^",
      Self::Shl => "<<",
      Self::Shr => ">>",
      Self::Lte => "<=",
      Self::Ltn => "<",
      Self::Eql => "==",
      Self::Gte => ">=",
      Self::Gtn => ">",
      Self::Neq => "!=",
    })
  }
}

impl std::fmt::Display for Term {
  // WARN: I think this could overflow, might need to rewrite it to be iterative instead of recursive?
  // NOTE: Another issue is complexity. This function is O(N^2). Should use ropes to be linear.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            if let Term::U6O { numb } = *args[0] {
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
      Self::Dup { nam0, nam1, expr, body } => write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body),
      Self::Sup { val0, val1 } => write!(f, "{{{} {}}}", val0, val1),
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
      Self::U6O { numb } => write!(f, "{}", &u60::show(*numb)),
      Self::F6O { numb } => write!(f, "{}", &f60::show(*numb)),
      Self::Op2 { oper, val0, val1 } => write!(f, "({} {} {})", oper, val0, val1),
    }
  }
}

// Rule
// ----

impl std::fmt::Display for Rule {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{} = {}", self.lhs, self.rhs)
  }
}

// File
// ----

impl std::fmt::Display for File {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.rules.iter().map(|rule| format!("{}", rule)).collect::<Vec<String>>().join("\n"))
  }
}

// Parser
// ======

pub fn parse_let(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  return parser::guard(
    parser::text_parser("let "),
    Box::new(|state| {
      let (state, _)    = parser::consume("let ", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, _)    = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _)    = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Let { name, expr, body })))
    }),
    state,
  );
}

pub fn parse_dup(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  return parser::guard(
    parser::text_parser("dup "),
    Box::new(|state| {
      let (state, _)    = parser::consume("dup ", state)?;
      let (state, nam0) = parser::name1(state)?;
      let (state, nam1) = parser::name1(state)?;
      let (state, _)    = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _)    = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Dup { nam0, nam1, expr, body })))
    }),
    state,
  );
}

pub fn parse_sup(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  parser::guard(
    parser::text_parser("{"),
    Box::new(move |state| {
      let (state, _)    = parser::consume("{", state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, _)    = parser::consume("}", state)?;
      Ok((state, Box::new(Term::Sup { val0, val1 })))
    }),
    state,
  )
}

pub fn parse_lam(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  let parse_symbol =
    |x| parser::parser_or(&[parser::text_parser("λ"), parser::text_parser("@")], x);
  parser::guard(
    Box::new(parse_symbol),
    Box::new(move |state| {
      let (state, _)    = parse_symbol(state)?;
      let (state, name) = parser::name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  )
}

pub fn parse_app(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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
            Box::new(Term::U6O { numb: 0 })
          }
        }),
        state,
      )
    }),
    state,
  );
}

pub fn parse_ctr(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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

pub fn parse_num(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      Ok((state, ('0'..='9').contains(&head)))
    }),
    Box::new(|state| {
      let (state, text) = parser::name1(state)?;
      if !text.is_empty() {
        if text.starts_with("0x") {
          return Ok((state, Box::new(Term::U6O { numb: u60::new(u64::from_str_radix(&text[2..], 16).unwrap()) })));
        } else {
          if text.find(".").is_some() {
            return Ok((state, Box::new(Term::F6O { numb: f60::new(text.parse::<f64>().unwrap()) })));
          } else {
            return Ok((state, Box::new(Term::U6O { numb: u60::new(text.parse::<u64>().unwrap()) })));
          }
        }
      } else {
        Ok((state, Box::new(Term::U6O { numb: 0 })))
      }
    }),
    state,
  )
}

pub fn parse_op2(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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

pub fn parse_var(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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

pub fn parse_sym_sugar(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  use std::hash::Hasher;
  parser::guard(
    parser::text_parser("%"),
    Box::new(|state| {
      let (state, _)    = parser::text("%", state)?;
      let (state, name) = parser::name(state)?;
      let hash = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(name.as_bytes());
        hasher.finish()
      };
      Ok((state, Box::new(Term::U6O { numb: hash })))
    }),
    state,
  )
}

// ask x = fn; body
// ----------------
// (fn λx body)
pub fn parse_ask_sugar_named(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  return parser::guard(
    Box::new(|state| {
      let (state, asks) = parser::text("ask ", state)?;
      let (state, name) = parser::name(state)?;
      let (state, eqls) = parser::text("=", state)?;
      Ok((state, asks && name.len() > 0 && eqls))
    }),
    Box::new(|state| {
      let (state, _)    = parser::consume("ask ", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, _)    = parser::consume("=", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _)    = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::App { func, argm: Box::new(Term::Lam { name, body }) })))
    }),
    state,
  );
}

pub fn parse_ask_sugar_anon(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
  return parser::guard(
    parser::text_parser("ask "),
    Box::new(|state| {
      let (state, _)    = parser::consume("ask ", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _)    = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::App { func, argm: Box::new(Term::Lam { name: "*".to_string(), body }) })))
    }),
    state,
  );
}

pub fn parse_chr_sugar(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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
        Ok((state, Box::new(Term::U6O { numb: c as u64 })))
      } else {
        parser::expected("character", 1, state)
      }
    }),
    state,
  )
}

// TODO: parse escape sequences
pub fn parse_str_sugar(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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
          } else {
            chars.push(next);
            state = parser::tail(state);
          }
        }
      }
      let empty = Term::Ctr { name: "String.nil".to_string(), args: Vec::new() };
      let list = Box::new(chars.iter().rfold(empty, |t, h| Term::Ctr {
        name: "String.cons".to_string(),
        args: vec![Box::new(Term::U6O { numb: *h as u64 }), Box::new(t)],
      }));
      Ok((state, list))
    }),
    state,
  )
}

pub fn parse_lst_sugar(state: parser::State) -> parser::Answer<Option<Box<Term>>> {
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
      let empty = Term::Ctr { name: "List.nil".to_string(), args: Vec::new() };
      let list = Box::new(elems.iter().rfold(empty, |t, h| Term::Ctr {
        name: "List.cons".to_string(),
        args: vec![h.clone(), Box::new(t)],
      }));
      Ok((state, list))
    }),
    state,
  )
}

pub fn parse_term(state: parser::State) -> parser::Answer<Box<Term>> {
  parser::grammar(
    "Term",
    &[
      Box::new(parse_let),
      Box::new(parse_dup),
      Box::new(parse_lam),
      Box::new(parse_ctr),
      Box::new(parse_op2),
      Box::new(parse_app),
      Box::new(parse_sup),
      Box::new(parse_num),
      Box::new(parse_sym_sugar),
      Box::new(parse_chr_sugar),
      Box::new(parse_str_sugar),
      Box::new(parse_lst_sugar),
      Box::new(parse_ask_sugar_named),
      Box::new(parse_ask_sugar_anon),
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
