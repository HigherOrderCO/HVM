use crate::parser;
use ropey::Rope;
use std::fmt;
use std::fmt::Write;

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
    args: Vec<BTerm>,
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

#[derive(Clone, Copy, Debug)]
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

#[derive(Debug, Clone, Default)]
struct RopeBuilder {
  rope_builder: ropey::RopeBuilder,
}

impl RopeBuilder {
  pub fn new() -> Self {
    RopeBuilder::default()
  }

  pub fn append(&mut self, chunk: &str) {
    self.rope_builder.append(chunk);
  }

  pub fn finish(self) -> Rope {
    self.rope_builder.finish()
  }
}

impl std::fmt::Write for RopeBuilder {
  fn write_str(&mut self, s: &str) -> fmt::Result {
    self.append(s);
    Ok(())
  }
}

impl fmt::Display for Term {
  // WARN: I think this could overflow, might need to rewrite it to be iterative instead of recursive?
  // NOTE: Another issue is complexity. This function is O(N^2). Should use ropes to be linear.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Var { name } => write!(f, "{}", name),
      Self::Dup {
        nam0,
        nam1,
        expr,
        body,
      } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "dup {} {} = {}; {}", nam0, nam1, expr, body)?;
        write!(f, "{}", builder.finish())
      }
      Self::Let { name, expr, body } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "let {} = {}; {}", name, expr, body)?;
        write!(f, "{}", builder.finish())
      }
      Self::Lam { name, body } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "λ{} {}", name, body)?;
        write!(f, "{}", builder.finish())
      }
      Self::App { func, argm } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "({} {})", func, argm)?;
        write!(f, "{}", builder.finish())
      }
      Self::Ctr { name, args } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "({}", name)?;
        args.iter().try_for_each(|x| write!(builder, " {}", x))?;
        builder.append(")");
        write!(f, "{}", builder.finish())
      }
      Self::U32 { numb } => write!(f, "{}", numb),
      Self::Op2 { oper, val0, val1 } => {
        let mut builder = RopeBuilder::new();
        write!(builder, "({} {} {})", oper, val0, val1)?;
        write!(f, "{}", builder.finish())
      }
    }
  }
}

// Rule
// ----

impl fmt::Display for Rule {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut builder = RopeBuilder::new();
    write!(builder, "{} = {}", self.lhs, self.rhs)?;
    write!(f, "{}", builder.finish())
  }
}

// File
// ----

impl fmt::Display for File {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.rules.len() > 0 {
      let mut builder = RopeBuilder::new();
      write!(builder, "{}", self.rules.first().unwrap())?;
      self
        .rules
        .iter()
        .skip(1)
        .try_for_each(|x| write!(builder, "\n{}", x))?;
      write!(f, "{}", builder.finish())
    } else {
      write!(f, "")
    }
  }
}

// Parser
// ======

pub fn parse_let<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    parser::text_parser("let "),
    Box::new(|state| {
      let (state, name) = parser::name1(state)?;
      let (state, spk1) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, skp2) = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      return Ok((state, Box::new(Term::Let { name, expr, body })));
    }),
    state,
  );
}

pub fn parse_dup<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    parser::text_parser("dup "),
    Box::new(|state| {
      let (state, nam0) = parser::name1(state)?;
      let (state, nam1) = parser::name1(state)?;
      let (state, spk1) = parser::consume("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, skp2) = parser::text(";", state)?;
      let (state, body) = parse_term(state)?;
      return Ok((
        state,
        Box::new(Term::Dup {
          nam0,
          nam1,
          expr,
          body,
        }),
      ));
    }),
    state,
  );
}

pub fn parse_lam<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    parser::text_parser("λ"),
    Box::new(|state| {
      let (state, skp0) = parser::text("λ", state)?;
      let (state, name) = parser::name(state)?;
      let (state, body) = parse_term(state)?;
      return Ok((state, Box::new(Term::Lam { name, body })));
    }),
    state,
  );
}

pub fn parse_app<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    parser::text_parser("("),
    Box::new(|state| {
      return parser::list(
        parser::text_parser("("),
        parser::text_parser(""),
        parser::text_parser(")"),
        Box::new(|x| parse_term(x)),
        Box::new(|args| {
          if args.len() > 0 {
            return args
              .into_iter()
              .reduce(|a, b| Box::new(Term::App { func: a, argm: b }))
              .unwrap();
          } else {
            return Box::new(Term::U32 { numb: 0 });
          }
        }),
        state,
      );
    }),
    state,
  );
}

pub fn parse_ctr<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    Box::new(|state| {
      let (state, open) = parser::text("(", state)?;
      let (state, head) = parser::get_char(state)?;
      return Ok((state, (head >= 'A' && head <= 'Z') || head == '.'));
    }),
    Box::new(|state| {
      let (state, skp0) = parser::text("(", state)?;
      let (state, name) = parser::name1(state)?;
      let (state, args) =
        parser::until(parser::text_parser(")"), Box::new(|x| parse_term(x)), state)?;
      return Ok((state, Box::new(Term::Ctr { name, args })));
    }),
    state,
  );
}

pub fn parse_u32<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      return Ok((state, head >= '0' && head <= '9'));
    }),
    Box::new(|state| {
      let (state, numb) = parser::name1(state)?;
      if numb.len() > 0 {
        return Ok((
          state,
          Box::new(Term::U32 {
            numb: numb.parse::<u32>().unwrap(),
          }),
        ));
      } else {
        return Ok((state, Box::new(Term::U32 { numb: 0 })));
      }
    }),
    state,
  );
}

pub fn parse_op2<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  fn is_op_char(chr: char) -> bool {
    return chr == '+'
      || chr == '\\'
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
      || chr == '!';
  }
  fn parse_oper<'a>(state: parser::State<'a>) -> parser::Answer<'a, Oper> {
    fn op<'a>(symb: &'static str, oper: Oper) -> parser::Parser<'a, Option<Oper>> {
      return Box::new(move |state| {
        let (state, done) = parser::text(symb, state)?;
        return Ok((state, if done { Some(oper) } else { None }));
      });
    }
    return parser::grammar("Oper", &[op("+", Oper::ADD)], state);
  }
  return parser::guard(
    Box::new(|state| {
      let (state, open) = parser::text("(", state)?;
      let (state, head) = parser::get_char(state)?;
      return Ok((state, open && is_op_char(head)));
    }),
    Box::new(|state| {
      let (state, skp0) = parser::text("(", state)?;
      let (state, oper) = parse_oper(state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, skp1) = parser::text(")", state)?;
      return Ok((state, Box::new(Term::Op2 { oper, val0, val1 })));
    }),
    state,
  );
}

pub fn parse_var<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    Box::new(|state| {
      let (state, head) = parser::get_char(state)?;
      return Ok((state, head >= 'a' && head <= 'z' || head == '_'));
    }),
    Box::new(|state| {
      let (state, name) = parser::name(state)?;
      return Ok((state, Box::new(Term::Var { name })));
    }),
    state,
  );
}

pub fn parse_term<'a>(state: parser::State<'a>) -> parser::Answer<'a, BTerm> {
  return parser::grammar(
    "Term",
    &[
      Box::new(|state| parse_let(state)),
      Box::new(|state| parse_dup(state)),
      Box::new(|state| parse_lam(state)),
      Box::new(|state| parse_ctr(state)),
      Box::new(|state| parse_op2(state)),
      Box::new(|state| parse_app(state)),
      Box::new(|state| parse_u32(state)),
      Box::new(|state| parse_var(state)),
      Box::new(|state| Ok((state, Some(Box::new(Term::U32 { numb: 0 }))))),
    ],
    state,
  );
}

pub fn parse_rule<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<Rule>> {
  return parser::guard(
    parser::text_parser(""),
    Box::new(|state| {
      let (state, lhs) = parse_term(state)?;
      let (state, spk) = parser::consume("=", state)?;
      let (state, rhs) = parse_term(state)?;
      return Ok((state, Rule { lhs, rhs }));
    }),
    state,
  );
}

pub fn parse_file<'a>(state: parser::State<'a>) -> parser::Answer<'a, File> {
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
  return Ok((state, File { rules }));
}

pub fn read_file(code: &str) -> File {
  return parser::read(Box::new(|x| parse_file(x)), code);
}
