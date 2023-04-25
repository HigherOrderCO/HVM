use HOPA;
use crate::runtime::data::u60;
use crate::runtime::data::f60;
use std::convert::{TryFrom, TryInto};

// Types
// =====

// Term
// ----

#[derive(Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

// SMap
// ----

type SMap = (String, Vec<bool>);

// File
// ----

pub struct File {
  pub rules: Vec<Rule>,
  pub smaps: Vec<SMap>,
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
              return Some(());
            }
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

impl Term {
    /// returns a term representing the integer `value`
    pub fn integer(value: u64) -> Self {
        // HVM uses 60 bit numbers, converted at parsing,
        // so mask its last 4 bits, to match its behavior
        Self::U6O {
            numb: value & !(0b1111 << 60),
        }
    }

    /// returns a term representing the floating point number `value`
    pub fn float(value: f64) -> Self {
        // HVM uses f60 numbers, converted at parsing,
        // so shift it by 4 bits, to match its behavior
        Self::F6O {
            numb: value.to_bits() >> 4,
        }
    }

    /// returns a term representing the expression `(func arg)`
    pub fn application(func: Self, arg: Self) -> Self {
        Self::App {
            func: Box::new(func),
            argm: Box::new(arg),
        }
    }

    /// returns a term representing the variable called `name`
    pub fn variable(name: impl Into<String>) -> Self {
        Self::Var { name: name.into() }
    }

    /// returns a term representing the expression `λparam. body`
    pub fn lambda(param: impl Into<String>, body: Self) -> Self {
        Self::Lam {
            name: param.into(),
            body: Box::new(body),
        }
    }

    /// returns a term representing the expression `lhs @ rhs`,
    /// where `@` is a binary operator from [`Oper`]
    pub fn binary_operator(op: Oper, lhs: Self, rhs: Self) -> Self {
        Self::Op2 {
            oper: op,
            val0: Box::new(lhs),
            val1: Box::new(rhs),
        }
    }

    /// returns a term representing the expression `(name args...)`,
    /// where `name` is a constructor
    pub fn constructor(name: impl Into<String>, args: impl IntoIterator<Item = Self>) -> Self {
        Self::Ctr {
            name: name.into(),
            args: args.into_iter().map(Box::new).collect(),
        }
    }

    /// returns a term representing the expression `if cond { expr } else { alt }`
    pub fn if_expression(cond: Self, expr: Self, alt: Self) -> Self {
        Self::constructor("U60.if", [cond, expr, alt])
    }

    /// returns a term representing the list `[values...]`
    pub fn list(values: impl IntoIterator<Item = Self>) -> Self {
        let terminator = Self::constructor("List.nil", []);
        // a list [a, b, c, ...] is represented by cons a (cons b cons c (...))
        // so to built it starting from the terminator,
        // the order must be reversed first
        let values: Vec<_> = values.into_iter().collect();
        values
            .into_iter()
            .rev()
            .fold(terminator, |acc, val| Self::constructor("List.cons", [val, acc]))
    }

    /// returns a term representing the string with the given content
    pub fn string(content: impl Into<String>) -> Self {
        let terminator = Self::constructor("String.nil", []);
        content
            .into()
            .chars()
            .rev() // see the comment in [`list`] for why the order is reversed
            .into_iter()
            .fold(terminator, |acc, val| {
                Self::constructor("String.cons", [Self::integer(val as u64), acc])
            })
    }

    /// returns an integer if the term represents one,
    /// returns None otherwise
    pub fn as_integer(&self) -> Option<u64> {
        if let Self::U6O { numb: v } = self {
            Some(*v)
        } else {
            None
        }
    }

    /// returns a floating point number if the term represents one,
    /// returns None otherwise
    pub fn as_float(&self) -> Option<f64> {
        if let Self::F6O { numb: v } = self {
            // undo the shift done at construction
            Some(f64::from_bits(v << 4))
        } else {
            None
        }
    }

    /// returns a unicode character if the term represents one,
    /// returns None otherwise
    pub fn as_char(&self) -> Option<char> {
        self.as_integer()
            .and_then(|c| c.try_into().ok())
            .and_then(char::from_u32)
    }

    /// returns the string represented by the term if it represents one,
    /// returns None otherwise
    pub fn as_string(&self) -> Option<String> {
        let mut current_term = self;
        let mut string = String::new();
        loop {
            match current_term {
                Self::Ctr { name, args } if name == "String.cons" => {
                    string.push(args.get(0).and_then(|c| Self::as_char(c))?);
                    current_term = args.get(1)?;
                    continue;
                }
                Self::Ctr { name, args: _ } if name == "String.nil" => return Some(string),
                _ => return None,
            }
        }
    }

    /// returns an iterator over the list represented by the term if it represents one
    /// returns None otherwise.
    ///
    /// improper lists, i.e. those of the form `List.cons a b` where `b` is not a list,
    /// terminate after `a`.
    pub fn as_list(&self) -> Option<impl Iterator<Item=&Term>> {
        match self {
            Self::Ctr { name, args } if (name == "List.cons" || name == "List.nil") => {
                let mut current_term = Some(self);
                Some(std::iter::from_fn(move || {
                    match current_term {
                        Some(Self::Ctr { name, args }) if name == "List.cons" => {
                            let elem = args.get(0).map(|x| x as &Self);
                            current_term = args.get(1).map(|x| x as &Self);
                            elem
                        }
                        _ => None,
                    }
                }))
            }
            _ => None,
        }
    }
}

impl From<u64> for Term {
    fn from(value: u64) -> Self {
        Self::integer(value)
    }
}

impl From<f64> for Term {
    fn from(value: f64) -> Self {
        Self::float(value)
    }
}

impl From<char> for Term {
    fn from(value: char) -> Self {
        Self::integer(value as u64)
    }
}

impl From<String> for Term {
    fn from(value: String) -> Self {
        Self::string(value)
    }
}

impl TryFrom<Term> for u64 {
    type Error = Term;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        if let Some(num) = value.as_integer() {
            Ok(num)
        } else {
            Err(value)
        }
    }
}

impl TryFrom<Term> for f64 {
    type Error = Term;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        if let Some(num) = value.as_float() {
            Ok(num)
        } else {
            Err(value)
        }
    }
}

impl TryFrom<Term> for char {
    type Error = Term;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        if let Some(chr) = value.as_char() {
            Ok(chr)
        } else {
            Err(value)
        }
    }
}

impl TryFrom<Term> for String {
    type Error = Term;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        if let Some(string) = value.as_string() {
            Ok(string)
        } else {
            Err(value)
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

impl Rule {
    /// returns an a HVM rule of the form `lhs = rhs`
    pub fn new(lhs: Term, rhs: Term) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
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

pub fn parse_let(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("let "),
    Box::new(|state| {
      let (state, _)    = HOPA::force_there_take_exact("let ", state)?;
      let (state, name) = HOPA::there_nonempty_name(state)?;
      let (state, _)    = HOPA::force_there_take_exact("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _)    = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Let { name, expr, body })))
    }),
    state,
  );
}

pub fn parse_dup(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("dup "),
    Box::new(|state| {
      let (state, _)    = HOPA::force_there_take_exact("dup ", state)?;
      let (state, nam0) = HOPA::there_nonempty_name(state)?;
      let (state, nam1) = HOPA::there_nonempty_name(state)?;
      let (state, _)    = HOPA::force_there_take_exact("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _)    = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Dup { nam0, nam1, expr, body })))
    }),
    state,
  );
}

pub fn parse_sup(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    HOPA::do_there_take_exact("{"),
    Box::new(move |state| {
      let (state, _)    = HOPA::force_there_take_exact("{", state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, _)    = HOPA::force_there_take_exact("}", state)?;
      Ok((state, Box::new(Term::Sup { val0, val1 })))
    }),
    state,
  )
}

pub fn parse_lam(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  let parse_symbol = |x| {
    return HOPA::any(&[
      HOPA::do_there_take_exact("λ"),
      HOPA::do_there_take_exact("@"),
    ], x);
  };
  HOPA::guard(
    Box::new(parse_symbol),
    Box::new(move |state| {
      let (state, _)    = parse_symbol(state)?;
      let (state, name) = HOPA::there_name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  )
}

pub fn parse_app(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("("),
    Box::new(|state| {
      HOPA::list(
        HOPA::do_there_take_exact("("),
        HOPA::do_there_take_exact(""),
        HOPA::do_there_take_exact(")"),
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

pub fn parse_ctr(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, _) = HOPA::there_take_exact("(", state)?;
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, ('A'..='Z').contains(&head)))
    }),
    Box::new(|state| {
      let (state, open) = HOPA::there_take_exact("(", state)?;
      let (state, name) = HOPA::there_nonempty_name(state)?;
      let (state, args) = if open {
        HOPA::until(HOPA::do_there_take_exact(")"), Box::new(parse_term), state)?
      } else {
        (state, Vec::new())
      };
      Ok((state, Box::new(Term::Ctr { name, args })))
    }),
    state,
  )
}

pub fn parse_num(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, ('0'..='9').contains(&head)))
    }),
    Box::new(|state| {
      let (state, text) = HOPA::there_nonempty_name(state)?;
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

pub fn parse_op2(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  fn is_op_char(chr: char) -> bool {
    matches!(chr, '+' | '-' | '*' | '/' | '%' | '&' | '|' | '^' | '<' | '>' | '=' | '!')
  }
  fn parse_oper(state: HOPA::State) -> HOPA::Answer<Oper> {
    fn op<'a>(symbol: &'static str, oper: Oper) -> HOPA::Parser<'a, Option<Oper>> {
      Box::new(move |state| {
        let (state, done) = HOPA::there_take_exact(symbol, state)?;
        Ok((state, if done { Some(oper) } else { None }))
      })
    }
    HOPA::attempt("Oper", &[
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
    ], state)
  }
  HOPA::guard(
    Box::new(|state| {
      let (state, open) = HOPA::there_take_exact("(", state)?;
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, open && is_op_char(head)))
    }),
    Box::new(|state| {
      let (state, _) = HOPA::there_take_exact("(", state)?;
      let (state, oper) = parse_oper(state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, _) = HOPA::there_take_exact(")", state)?;
      Ok((state, Box::new(Term::Op2 { oper, val0, val1 })))
    }),
    state,
  )
}

pub fn parse_var(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, ('a'..='z').contains(&head) || head == '_' || head == '$'))
    }),
    Box::new(|state| {
      let (state, name) = HOPA::there_name(state)?;
      Ok((state, Box::new(Term::Var { name })))
    }),
    state,
  )
}

pub fn parse_sym_sugar(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  use std::hash::Hasher;
  HOPA::guard(
    HOPA::do_there_take_exact("%"),
    Box::new(|state| {
      let (state, _)    = HOPA::there_take_exact("%", state)?;
      let (state, name) = HOPA::there_name(state)?;
      let hash = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(name.as_bytes());
        hasher.finish()
      };
      Ok((state, Box::new(Term::U6O { numb: u60::new(hash) })))
    }),
    state,
  )
}

// ask x = fn; body
// ----------------
// (fn λx body)
pub fn parse_ask_sugar_named(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    Box::new(|state| {
      let (state, asks) = HOPA::there_take_exact("ask ", state)?;
      let (state, name) = HOPA::there_name(state)?;
      let (state, eqls) = HOPA::there_take_exact("=", state)?;
      Ok((state, asks && name.len() > 0 && eqls))
    }),
    Box::new(|state| {
      let (state, _)    = HOPA::force_there_take_exact("ask ", state)?;
      let (state, name) = HOPA::there_nonempty_name(state)?;
      let (state, _)    = HOPA::force_there_take_exact("=", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _)    = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::App { func, argm: Box::new(Term::Lam { name, body }) })))
    }),
    state,
  );
}

pub fn parse_ask_sugar_anon(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("ask "),
    Box::new(|state| {
      let (state, _)    = HOPA::force_there_take_exact("ask ", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _)    = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::App { func, argm: Box::new(Term::Lam { name: "*".to_string(), body }) })))
    }),
    state,
  );
}

pub fn parse_chr_sugar(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, head == '\''))
    }),
    Box::new(|state| {
      let (state, _) = HOPA::there_take_exact("'", state)?;
      if let Some(c) = HOPA::head(state) {
        let state = HOPA::tail(state);
        let (state, _) = HOPA::there_take_exact("'", state)?;
        Ok((state, Box::new(Term::U6O { numb: c as u64 })))
      } else {
        HOPA::expected("character", 1, state)
      }
    }),
    state,
  )
}

// TODO: parse escape sequences
pub fn parse_str_sugar(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, head == '"' || head == '`'))
    }),
    Box::new(|state| {
      let delim = HOPA::head(state).unwrap_or('\0');
      let state = HOPA::tail(state);
      let mut chars: Vec<char> = Vec::new();
      let mut state = state;
      loop {
        if let Some(next) = HOPA::head(state) {
          if next == delim || next == '\0' {
            state = HOPA::tail(state);
            break;
          } else {
            chars.push(next);
            state = HOPA::tail(state);
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

pub fn parse_lst_sugar(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    Box::new(|state| {
      let (state, head) = HOPA::there_take_head(state)?;
      Ok((state, head == '['))
    }),
    Box::new(|state| {
      let (state, _head) = HOPA::there_take_exact("[", state)?;
      // let mut elems: Vec<Box<Term>> = Vec::new();
      let state = state;
      let (state, elems) = HOPA::until(
        Box::new(|x| HOPA::there_take_exact("]", x)),
        Box::new(|x| {
          let (state, term) = parse_term(x)?;
          let (state, _) = HOPA::maybe(Box::new(|x| HOPA::there_take_exact(",", x)), state)?;
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

pub fn parse_if_sugar(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("if "),
    Box::new(|state| {
      let (state, _)    = HOPA::force_there_take_exact("if ", state)?;
      let (state, cond) = parse_term(state)?;
      let (state, _)    = HOPA::force_there_take_exact("{", state)?;
      let (state, if_t) = parse_term(state)?;
      let (state, _)    = HOPA::force_there_take_exact("}", state)?;
      let (state, _)    = HOPA::force_there_take_exact("else", state)?;
      let (state, _)    = HOPA::force_there_take_exact("{", state)?;
      let (state, if_f) = parse_term(state)?;
      let (state, _)    = HOPA::force_there_take_exact("}", state)?;
      Ok((state, Box::new(Term::Ctr { name: "U60.if".to_string(), args: vec![cond, if_t, if_f] })))
    }),
    state,
  );
}

pub fn parse_bng(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(HOPA::do_there_take_exact("!"), Box::new(|state| {
    let (state, _)    = HOPA::force_there_take_exact("!", state)?;
    let (state, term) = parse_term(state)?;
    Ok((state, term))
  }), state);
}

pub fn parse_term(state: HOPA::State) -> HOPA::Answer<Box<Term>> {
  HOPA::attempt("Term", &[
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
    Box::new(parse_if_sugar),
    Box::new(parse_bng),
    Box::new(parse_ask_sugar_named),
    Box::new(parse_ask_sugar_anon),
    Box::new(parse_var),
    Box::new(|state| Ok((state, None))),
  ], state)
}

pub fn parse_rule(state: HOPA::State) -> HOPA::Answer<Option<Rule>> {
  return HOPA::guard(
    HOPA::do_there_take_exact(""),
    Box::new(|state| {
      let (state, lhs) = parse_term(state)?;
      let (state, _) = HOPA::force_there_take_exact("=", state)?;
      let (state, rhs) = parse_term(state)?;
      Ok((state, Rule { lhs, rhs }))
    }),
    state,
  );
}

pub fn parse_smap(state: HOPA::State) -> HOPA::Answer<Option<SMap>> {
  pub fn parse_stct(state: HOPA::State) -> HOPA::Answer<bool> {
    let (state, stct) = HOPA::there_take_exact("!", state)?;
    let (state, _)    = parse_term(state)?;
    Ok((state, stct))
  }
  let (state, init) = HOPA::there_take_exact("(", state)?;
  if init {
    let (state, name) = HOPA::there_nonempty_name(state)?;
    let (state, args) = HOPA::until(HOPA::do_there_take_exact(")"), Box::new(parse_stct), state)?;
    return Ok((state, Some((name, args))));
  } else {
    return Ok((state, None));
  }
}

pub fn parse_file(state: HOPA::State) -> HOPA::Answer<File> {
  let mut rules = Vec::new();
  let mut smaps = Vec::new();
  let mut state = state;
  loop {
    let (new_state, done) = HOPA::there_end(state)?;
    if done {
      break;
    }
    let (_, smap) = parse_smap(new_state)?;
    if let Some(smap) = smap {
      smaps.push(smap);
    }
    let (new_state, rule) = parse_rule(new_state)?;
    if let Some(rule) = rule {
      rules.push(rule);
      state = new_state;
      continue;
    }
    return HOPA::expected("declaration", 1, state);
  }
  Ok((state, File { rules, smaps }))
}

pub fn read_term(code: &str) -> Result<Box<Term>, String> {
  HOPA::read(Box::new(parse_term), code)
}

pub fn read_file(code: &str) -> Result<File, String> {
  HOPA::read(Box::new(parse_file), code)
}

#[allow(dead_code)]
pub fn read_rule(code: &str) -> Result<Option<Rule>, String> {
  HOPA::read(Box::new(parse_rule), code)
}
