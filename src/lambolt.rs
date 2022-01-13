use crate::parser;
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
        .rules
        .iter()
        .map(|rule| format!("{}", rule))
        .collect::<Vec<String>>()
        .join("\n")
    )
  }
}

// Parser
// ======

pub fn parse_let<'a>(state: parser::State<'a>) -> parser::Answer<'a, Option<BTerm>> {
  return parser::guard(
    parser::text_parser("let "),
    Box::new(|state| {
      let (state, spk1) = parser::consume("let ", state)?;
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
      let (state, spk1) = parser::consume("dup ", state)?;
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
      let (state, args) = parser::until(parser::text_parser(")"), Box::new(|x| parse_term(x)), state)?;
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
      Box::new(|state| Ok((state, None))),
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

pub fn read_term(code: &str) -> Box<Term> {
  return parser::read(Box::new(|x| parse_term(x)), code);
}

pub fn read_file(code: &str) -> File {
  return parser::read(Box::new(|x| parse_file(x)), code);
}
