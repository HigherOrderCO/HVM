use crate::parser;
use std::fmt;
use std::{collections::HashMap, fmt::format, u64};

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

// Sanitize
// ========

pub struct SanitizeResult {
  pub rule: Rule,
  pub uses: HashMap<String, u64>,
}

// This big function sanitizes a rule. That is, it renames every variable in a
// rule, in order to make it unique. Moreover, it will also add a `.N` to the
// end of the name of each variable used in the right-hand side of the rule,
// where `N` stands for the number of times it was used. For example:
//   sanitize `(fn (cons head tail)) = (cons (pair head head) tail)`
//         ~> `(fn (cons x0   x1))   = (cons (pair x0.0 x0.1) x1.0)`
// It also returns the usage count of each variable.
pub fn sanitize(rule: &Rule) -> Result<SanitizeResult, String> {

  // Pass through the lhs of the function generating new names
  // for every variable found in the style described before with
  // the fresh function. Also checks if rule's left side is valid.
  type NameTable = HashMap<String, String>;
  fn create_fresh(rule: &Rule, fresh: &mut dyn FnMut() -> String) -> Result<NameTable, String> {
    let mut table: HashMap<String, String> = HashMap::new();

    let lhs = &rule.lhs;
    if let Term::Ctr { ref name, ref args } = **lhs {
      for arg in args {
        match &**arg {
          Term::Var { name, .. } => {
            table.insert(name.clone(), fresh());
          }
          Term::Ctr { args, .. } => {
            for arg in args {
              if let Term::Var { name } = &**arg {
                table.insert(name.clone(), fresh());
              } else {
                return Err("Invalid left-hand side".to_owned());
              }
            }
          }
          Term::U32 { .. } => {}
          _ => {
            return Err("Invalid left-hand side".to_owned());
          }
        }
      }
    } else {
      return Err("Invalid left-hand side".to_owned());
    }

    Ok(table)
  }

  struct CtxSanitizeTerm<'a> {
    uses: &'a mut HashMap<String, u64>,
    fresh: &'a mut dyn FnMut() -> String,
  }

  // Sanitize one term, following the described in main function
  fn sanitize_term(
    term: &Term,
    lhs: bool,
    tbl: &mut HashMap<String, String>,
    ctx: &mut CtxSanitizeTerm,
  ) -> Result<Box<Term>, String> {
    let term = match term {
      Term::Var { name } => {
        if lhs {
          // create a var with the name generated before
          let name = tbl.get(name).unwrap_or(name);
          Box::new(Term::Var { name: name.clone() })
        } else {
          // create a var with the name generated before
          // concatenated with '.{{times_used}}'
          let gen_name = tbl.get(name);
          if let Some(name) = gen_name {
            let used = {
              *ctx
                .uses
                .entry(name.clone())
                .and_modify(|x| *x += 1)
                .or_insert(1)
            };
            let name = format!("{}.{}", name, used - 1);
            Box::new(Term::Var { name })
          } else {
            return Err(format!("Error: unbound variable {}.", name));
          }
        }
      }
      Term::Dup { expr, body, nam0, nam1 } => {
        let new_nam0 = (ctx.fresh)();
        let new_nam1 = (ctx.fresh)();
        let expr = sanitize_term(expr, lhs, tbl, ctx)?;
        tbl.insert(nam0.clone(), new_nam0.clone());
        tbl.insert(nam1.clone(), new_nam1.clone());

        let body = sanitize_term(body, lhs, tbl, ctx)?;
        let nam0 = format!("{}.0", new_nam0.clone());
        let nam1 = format!("{}.0", new_nam1.clone());
        let term = Term::Dup {
          nam0,
          nam1,
          expr,
          body,
        };
        Box::new(term)
      }
      Term::Let { name, expr, body } => {
        let new_name = (ctx.fresh)();
        let expr = sanitize_term(expr, lhs, tbl, ctx)?;
        tbl.insert(name.clone(), new_name);

        let body = sanitize_term(body, lhs, tbl, ctx)?;
        let term = duplicator(&name, expr, body, ctx.uses);
        term
      }
      Term::Lam { name, body } => {
        let new_name = (ctx.fresh)();
        tbl.insert(name.clone(), new_name.clone());
        let body = {
          let body = sanitize_term(body, lhs, tbl, ctx)?;
          let expr = Box::new(Term::Var {
            name: new_name.clone(),
          });
          let body = duplicator(&name, expr, body, ctx.uses);
          body
        };
        let term = Term::Lam {
          name: new_name.clone(),
          body,
        };
        Box::new(term)
      }
      Term::App { func, argm } => {
        let func = sanitize_term(func, lhs, tbl, ctx)?;
        let argm = sanitize_term(argm, lhs, tbl, ctx)?;
        let term = Term::App { func, argm };
        Box::new(term)
      }
      Term::Ctr { name, args } => {
        let mut n_args = vec![];
        for arg in args {
          let arg = sanitize_term(arg, lhs, tbl, ctx)?;
          n_args.push(arg);
        }
        let term = Term::Ctr {
          name: name.clone(),
          args: n_args,
        };
        Box::new(term)
      }
      Term::Op2 { oper, val0, val1 } => {
        let val0 = sanitize_term(val0, lhs, tbl, ctx)?;
        let val1 = sanitize_term(val1, lhs, tbl, ctx)?;
        let term = Term::Op2 {
          oper: *oper,
          val0,
          val1,
        };
        Box::new(term)
      }
      Term::U32 { numb } => {
        let term = Term::U32 { numb: *numb };
        Box::new(term)
      }
    };

    Ok(term)
  }

  // Duplicates all variables that are used more than once.
  // The process is done generating auxiliary variables and
  // applying dup on them.
  fn duplicator(
    name: &String,
    expr: Box<Term>,
    body: Box<Term>,
    uses: &HashMap<String, u64>,
  ) -> Box<Term> {
    let amount = uses.get(name).map(|x| *x);
    // verify if variable is used more than once
    if amount > Some(1) {
      let amount = amount.unwrap(); // certainly is not None
      let duplicated_times = amount - 1; // times that name is duplicated
      let aux_qtt = amount - 2; // quantity of aux variables
      let mut vars = vec![];

      // generate name for duplicated variables
      for i in (aux_qtt..duplicated_times * 2).rev() {
        let i = i - aux_qtt; // moved to 0,1,..
        let key = format!("{}.{}", name, i);
        vars.push(key);
      }

      // generate name for aux variables
      for i in (0..aux_qtt).rev() {
        let key = format!("c.{}", i);
        vars.push(key);
      }

      // use aux variables to duplicate the variable
      let dup = Term::Dup {
        nam0: vars.pop().unwrap(),
        nam1: vars.pop().unwrap(),
        expr,
        body: duplicator_go(1, duplicated_times, body, &mut vars),
      };

      Box::new(dup)
    } else {
      // if not used more than once just make a let then
      let term = Term::Let {
        name: format!("{}.0", name),
        expr,
        body,
      };
      Box::new(term)
    }
  }

  // Recursive aux function to duplicate one varible
  // an amount of times
  fn duplicator_go(
    i: u64,
    duplicated_times: u64,
    body: Box<Term>,
    vars: &mut Vec<String>,
  ) -> Box<Term> {
    if i == duplicated_times {
      body
    } else {
      let nam0 = vars.pop().unwrap();
      let nam1 = vars.pop().unwrap();
      let exp0 = Box::new(Term::Var {
        name: format!("c.{}", i - 1),
      });
      Box::new(Term::Dup {
        nam0,
        nam1,
        expr: exp0,
        body: duplicator_go(i + 1, duplicated_times, body, vars),
      })
    }
  }

  let mut size = 0;
  let mut uses: HashMap<String, u64> = HashMap::new();

  // creates a new name for a variable
  // the first will receive x0, second x1, ...
  let mut fresh = || {
    let key = format!("x{}", size);
    size += 1;
    key
  };

  // generate table containing the new_names following
  // pattern described before
  let table = create_fresh(rule, &mut fresh)?;

  // create context for sanitize_term
  let mut ctx = CtxSanitizeTerm {
    uses: &mut uses,
    fresh: &mut fresh,
  };

  // sanitize left side
  let lhs = sanitize_term(&rule.lhs, true, &mut table.clone(), &mut ctx)?;
  // sanitize right side
  let mut rhs = sanitize_term(&rule.rhs, false, &mut table.clone(), &mut ctx)?;

  // duplicate right side variables that are used more than once
  for (key, value) in table {
    let expr = Box::new(Term::Var {
      name: value.clone(),
    });
    rhs = duplicator(&value, expr, rhs, &mut uses);
  }

  // forms the new rules
  let rule = Rule { lhs, rhs };
  Ok(SanitizeResult { rule, uses })
}


// Meta generators
// ---------------

// Generates a name table for a whole program. That table links constructor
// names (such as `cons` and `succ`) to small ids (such as `0` and `1`).
pub type IdTable = HashMap<String, u64>;
pub fn gen_name_table(file: &File) -> IdTable {
  fn find_ctrs(term: &Term, table: &mut IdTable, fresh: &mut u64) {
    match term {
      Term::Dup { expr, body, .. } => {
        find_ctrs(expr, table, fresh);
        find_ctrs(body, table, fresh);
      }
      Term::Let { expr, body, .. } => {
        find_ctrs(expr, table, fresh);
        find_ctrs(body, table, fresh);
      }
      Term::Lam { body, .. } => {
        find_ctrs(body, table, fresh);
      }
      Term::App { func, argm, .. } => {
        find_ctrs(func, table, fresh);
        find_ctrs(argm, table, fresh);
      }
      Term::Op2 { val0, val1, .. } => {
        find_ctrs(val0, table, fresh);
        find_ctrs(val1, table, fresh);
      }
      Term::Ctr { name, args } => {
        let id = table.get(name);
        if id.is_none() {
          let first_char = name.chars().next();
          if let Some(c) = first_char {
            if c == '.' {
              let id = &name[1..].parse::<u64>();
              if let Ok(id) = id {
                table.insert(name.clone(), *id);
              }
            } else {
              table.insert(name.clone(), *fresh);
              *fresh += 1;
            }
          }
          for arg in args {
            find_ctrs(arg, table, fresh);
          }
        }
      }
      _ => (),
    }
  }

  let mut table = HashMap::new();
  let mut fresh = 0;
  let rules = &file.rules;
  for rule in rules {
    find_ctrs(&rule.lhs, &mut table, &mut fresh);
    find_ctrs(&rule.rhs, &mut table, &mut fresh);
  }
  table
}

// Finds constructors that are used as functions.
pub type IsFunctionTable = HashMap<String, bool>;
pub fn gen_is_call(file: &File) -> IsFunctionTable {
  let mut is_call: IsFunctionTable = HashMap::new();
  let rules = &file.rules;
  for rule in rules {
    let term = &rule.lhs;
    if let Term::Ctr { ref name, .. } = **term {
      // FIXME: this looks wrong, will check later
      is_call.insert(name.clone(), true);
    }
  }
  is_call
}

// Groups rules by name. For example:
//   (add (succ a) (succ b)) = (succ (succ (add a b)))
//   (add (succ a) (zero)  ) = (succ a)
//   (add (zero)   (succ b)) = (succ b)
//   (add (zero)   (zero)  ) = (zero)
// This is a group of 4 rules starting with the "add" name.
pub type GroupTable<'a> = HashMap<String, (usize, Vec<&'a Rule>)>;
pub fn gen_groups(file: &File) -> GroupTable {
  let mut groups: GroupTable = HashMap::new();
  let rules = &file.rules;
  for rule in rules {
    let term = &rule.lhs;
    if let Term::Ctr { name, args } = &**term {
      // FIXME: this looks wrong, will check later
      let args_size = args.len();
      let group = groups.get_mut(name);
      match group {
        None => {
          groups.insert(name.clone(), (args_size, Vec::from([rule])));
        }
        Some(group) => {
          let (size, rules) = group;
          if *size == args_size {
            rules.push(rule);
          }
        }
      }
    }
  }
  groups
}
