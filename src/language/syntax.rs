use crate::runtime::data::f60;
use crate::runtime::data::u60;
use HOPA;

use std::collections::{BTreeMap, HashMap};

type NameTable = BTreeMap<String, String>;

use crate::runtime::Tag;

struct CtxSanitizeTerm<'a> {
  uses: &'a mut HashMap<String, u64>,
  fresh: &'a mut dyn FnMut() -> String,
}
// Recursive aux function to duplicate one variable
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
    let exp0 = Box::new(Term::Var { name: format!("c.{}", i - 1) });
    Box::new(Term::Dup {
      nam0,
      nam1,
      expr: exp0,
      body: duplicator_go(i + 1, duplicated_times, body, vars),
    })
  }
}
// Duplicates all variables that are used more than once.
// The process is done generating auxiliary variables and
// applying dup on them.
fn duplicator(
  name: &str,
  expr: Box<Term>,
  body: Box<Term>,
  uses: &HashMap<String, u64>,
) -> Box<Term> {
  let amount = uses.get(name).copied();

  match amount {
    // if not used nothing is done
    None => body,
    Some(x) => {
      match x.cmp(&1) {
        // if not used nothing is done
        std::cmp::Ordering::Less => body,
        // if used once just make a let then
        std::cmp::Ordering::Equal => {
          let term = Term::Let { name: format!("{}.0", name), expr, body };
          Box::new(term)
        }
        // if used more then once duplicate
        std::cmp::Ordering::Greater => {
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
        }
      }
    }
  }
}
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

impl Term {
  pub fn is_matchable(&self) -> bool {
    matches!(self, Self::Ctr { .. } | Self::U6O { .. } | Self::F6O { .. })
  }

  pub fn is_strict(&self) -> bool {
    matches!(self, Self::Ctr { .. } | Self::U6O { .. } | Self::F6O { .. })
  }

  // Checks if this rule has nested patterns, and must be splitted
  #[rustfmt::skip]
  pub fn must_split(&self) -> bool {
/**/if let Self::Ctr { ref args, .. } = *self {
/*  */for arg in args {
/*  H */if let Self::Ctr { args: ref arg_args, .. } = **arg {
/*   A  */for field in arg_args {
/*    D   */if field.is_matchable() {
/* ─=≡ΣO)   */return true;
/*    U   */}
/*   K  */}
/*  E */}
/* N*/}
/**/} false
  }

  pub fn subst(&mut self, sub_name: &str, value: &Self) {
    match self {
      Self::Var { name } => {
        if sub_name == name {
          *self = value.clone();
        }
      }
      Self::Dup { nam0, nam1, expr, body } => {
        expr.subst(sub_name, value);
        if nam0 != sub_name && nam1 != sub_name {
          body.subst(sub_name, value);
        }
      }
      Self::Sup { val0, val1 } => {
        val0.subst(sub_name, value);
        val1.subst(sub_name, value);
      }
      Self::Let { name, expr, body } => {
        expr.subst(sub_name, value);
        if name != sub_name {
          body.subst(sub_name, value);
        }
      }
      Self::Lam { name, body } => {
        if name != sub_name {
          body.subst(sub_name, value);
        }
      }
      Self::App { func, argm } => {
        func.subst(sub_name, value);
        argm.subst(sub_name, value);
      }
      Self::Ctr { args, .. } => {
        for arg in args {
          arg.subst(sub_name, value);
        }
      }
      Self::U6O { .. } => {}
      Self::F6O { .. } => {}
      Self::Op2 { val0, val1, .. } => {
        val0.subst(sub_name, value);
        val1.subst(sub_name, value);
      }
    }
  }

  // Sanitize one term, following the described in main function
  fn sanitize_term(
    &self,
    lhs: bool,
    tbl: &mut NameTable,
    ctx: &mut CtxSanitizeTerm,
  ) -> Result<Box<Self>, String> {
    fn rename_erased(name: &mut String, uses: &HashMap<String, u64>) {
      if !Tag::global_name_misc(name).is_some() && uses.get(name).copied() <= Some(0) {
        *name = "*".to_string();
      }
    }
    let term = match self {
      Term::Var { name } => {
        if lhs {
          let mut name = tbl.get(name).unwrap_or(name).clone();
          rename_erased(&mut name, ctx.uses);
          Box::new(Term::Var { name })
        } else if Tag::global_name_misc(name).is_some() {
          if tbl.get(name).is_some() {
            panic!("Using a global variable more than once isn't supported yet. Use an explicit 'let' to clone it. {} {:?}", name, tbl.get(name));
          } else {
            tbl.insert(name.clone(), String::new());
            Box::new(Term::Var { name: name.clone() })
          }
        } else {
          // create a var with the name generated before
          // concatenated with '.{{times_used}}'
          if let Some(name) = tbl.get(name) {
            let used = { *ctx.uses.entry(name.clone()).and_modify(|x| *x += 1).or_insert(1) };
            let name = format!("{}.{}", name, used - 1);
            Box::new(Term::Var { name })
          //} else if get_global_name_misc(&name) {
          // println!("Allowed unbound variable: {}", name);
          // Box::new(Term::Var { name: name.clone() })
          } else {
            return Err(format!("Unbound variable: `{}`.", name));
          }
        }
      }
      Self::Dup { expr, body, nam0, nam1 } => {
        let is_global_0 = Tag::global_name_misc(nam0).is_some();
        let is_global_1 = Tag::global_name_misc(nam1).is_some();
        if is_global_0 && Tag::global_name_misc(nam0) != Some(Tag::DP0) {
          panic!("The name of the global dup var '{}' must start with '$0'.", nam0);
        }
        if is_global_1 && Tag::global_name_misc(nam1) != Some(Tag::DP1) {
          panic!("The name of the global dup var '{}' must start with '$1'.", nam1);
        }
        if is_global_0 != is_global_1 {
          panic!("Both variables must be global: '{}' and '{}'.", nam0, nam1);
        }
        if is_global_0 && &nam0[2..] != &nam1[2..] {
          panic!("Global dup names must be identical: '{}' and '{}'.", nam0, nam1);
        }
        let new_nam0 = if is_global_0 { nam0.clone() } else { (ctx.fresh)() };
        let new_nam1 = if is_global_1 { nam1.clone() } else { (ctx.fresh)() };
        let expr = expr.sanitize_term(lhs, tbl, ctx)?;
        let got_nam0 = tbl.remove(nam0);
        let got_nam1 = tbl.remove(nam1);
        if !is_global_0 {
          tbl.insert(nam0.clone(), new_nam0.clone());
        }
        if !is_global_1 {
          tbl.insert(nam1.clone(), new_nam1.clone());
        }
        let body = body.sanitize_term(lhs, tbl, ctx)?;
        if !is_global_0 {
          tbl.remove(nam0);
        }
        if let Some(x) = got_nam0 {
          tbl.insert(nam0.clone(), x);
        }
        if !is_global_1 {
          tbl.remove(nam1);
        }
        if let Some(x) = got_nam1 {
          tbl.insert(nam1.clone(), x);
        }
        let nam0 = format!("{}{}", new_nam0, if !is_global_0 { ".0" } else { "" });
        let nam1 = format!("{}{}", new_nam1, if !is_global_0 { ".0" } else { "" });
        let term = Self::Dup { nam0, nam1, expr, body };
        Box::new(term)
      }
      Self::Sup { val0, val1 } => {
        let val0 = val0.sanitize_term(lhs, tbl, ctx)?;
        let val1 = val1.sanitize_term(lhs, tbl, ctx)?;
        let term = Self::Sup { val0, val1 };
        Box::new(term)
      }
      Self::Let { name, expr, body } => {
        if Tag::global_name_misc(name).is_some() {
          panic!("Global variable '{}' not allowed on let. Use dup instead.", name);
        }
        let new_name = (ctx.fresh)();
        let expr = expr.sanitize_term(lhs, tbl, ctx)?;
        let got_name = tbl.remove(name);
        tbl.insert(name.clone(), new_name.clone());
        let body = body.sanitize_term(lhs, tbl, ctx)?;
        tbl.remove(name);
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        duplicator(&new_name, expr, body, ctx.uses)
      }
      Self::Lam { name, body } => {
        let is_global = Tag::global_name_misc(name).is_some();
        let mut new_name = if is_global { name.clone() } else { (ctx.fresh)() };
        let got_name = tbl.remove(name);
        if !is_global {
          tbl.insert(name.clone(), new_name.clone());
        }
        let body = body.sanitize_term(lhs, tbl, ctx)?;
        if !is_global {
          tbl.remove(name);
        }
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        let expr = Box::new(Term::Var { name: new_name.clone() });
        let body = duplicator(&new_name, expr, body, ctx.uses);
        rename_erased(&mut new_name, ctx.uses);
        let term = Self::Lam { name: new_name, body };
        Box::new(term)
      }
      Self::App { func, argm } => {
        let func = func.sanitize_term(lhs, tbl, ctx)?;
        let argm = argm.sanitize_term(lhs, tbl, ctx)?;
        let term = Self::App { func, argm };
        Box::new(term)
      }
      Self::Ctr { name, args } => {
        let mut n_args = Vec::with_capacity(args.len());
        for arg in args {
          let arg = arg.sanitize_term(lhs, tbl, ctx)?;
          n_args.push(arg);
        }
        let term = Self::Ctr { name: name.clone(), args: n_args };
        Box::new(term)
      }
      Self::Op2 { oper, val0, val1 } => {
        let val0 = val0.sanitize_term(lhs, tbl, ctx)?;
        let val1 = val1.sanitize_term(lhs, tbl, ctx)?;
        let term = Self::Op2 { oper: *oper, val0, val1 };
        Box::new(term)
      }
      Self::U6O { numb } => {
        let term = Self::U6O { numb: *numb };
        Box::new(term)
      }
      Self::F6O { numb } => {
        let term = Self::F6O { numb: *numb };
        Box::new(term)
      }
    };

    Ok(term)
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
  pub lhs: Box<Term>,
  pub rhs: Box<Term>,
}

impl Rule {
  // Checks true if every time that `a` matches, `b` will match too
  pub fn matches_together(&self, b: &Self) -> (bool, bool) {
    let mut same_shape = true;
    if let (
      Term::Ctr { name: ref _a_name, args: ref a_args },
      Term::Ctr { name: ref _b_name, args: ref b_args },
    ) = (&*self.lhs, &*b.lhs)
    {
      for (a_arg, b_arg) in a_args.iter().zip(b_args) {
        match **a_arg {
          Term::Ctr { name: ref a_arg_name, args: ref a_arg_args } => match **b_arg {
            Term::Ctr { name: ref b_arg_name, args: ref b_arg_args } => {
              if a_arg_name != b_arg_name || a_arg_args.len() != b_arg_args.len() {
                return (false, false);
              }
            }
            Term::U6O { .. } => {
              return (false, false);
            }
            Term::F6O { .. } => {
              return (false, false);
            }
            Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          Term::U6O { numb: a_arg_numb } => match **b_arg {
            Term::U6O { numb: b_arg_numb } => {
              if a_arg_numb != b_arg_numb {
                return (false, false);
              }
            }
            Term::Ctr { .. } => {
              return (false, false);
            }
            Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          Term::F6O { numb: a_arg_numb } => match **b_arg {
            Term::F6O { numb: b_arg_numb } => {
              if a_arg_numb != b_arg_numb {
                return (false, false);
              }
            }
            Term::Ctr { .. } => {
              return (false, false);
            }
            Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          _ => {}
        }
      }
    }
    (true, same_shape)
  }

  // FIXME: right now, the sanitizer isn't able to identify if a scopeless lambda doesn't use its
  // bound variable, so it won't set the "eras" flag to "true" in this case, but it should.

  // This big function sanitizes a rule. That has the following effect:
  // - All variables are renamed to have a global unique name.
  // - All variables are linearized.
  //   - If they're used more than once, dups are inserted.
  //   - If they're used once, nothing changes.
  //   - If they're never used, their name is changed to "*"
  // Example:
  //   - sanitizing: `(Foo a b) = (+ a a)`
  //   - results in: `(Foo x0 *) = dup x0.0 x0.1 = x0; (+ x0.0 x0.1)`
  pub fn sanitize_rule(&self) -> Result<Self, String> {
    // Pass through the lhs of the function generating new names
    // for every variable found in the style described before with
    // the fresh function. Also checks if rule's left side is valid.
    // BTree is used here for determinism (HashMap does not maintain
    // order among executions)

    fn create_fresh(rule: &Rule, fresh: &mut dyn FnMut() -> String) -> Result<NameTable, String> {
      let mut table = BTreeMap::new();

      let lhs = &rule.lhs;
      if let Term::Ctr { name: _, ref args } = **lhs {
        for arg in args {
          match &**arg {
            Term::Var { name, .. } => {
              table.insert(name.clone(), fresh());
            }
            Term::Ctr { args, .. } => {
              for arg in args {
                if let Term::Var { name } = &**arg {
                  table.insert(name.clone(), fresh());
                }
              }
            }
            Term::U6O { .. } => {}
            Term::F6O { .. } => {}
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
    let table = create_fresh(&self, &mut fresh)?;

    // create context for sanitize_term
    let mut ctx = CtxSanitizeTerm { uses: &mut uses, fresh: &mut fresh };

    // sanitize left and right sides
    let mut rhs = self.rhs.sanitize_term(false, &mut table.clone(), &mut ctx)?;
    let lhs = self.lhs.sanitize_term(true, &mut table.clone(), &mut ctx)?;

    // duplicate right side variables that are used more than once
    for (_key, value) in table {
      let expr = Box::new(Term::Var { name: value.clone() });
      rhs = duplicator(&value, expr, rhs, &uses);
    }

    // returns the sanitized rule
    Ok(Rule { lhs, rhs })
  }
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
      Self::Dup { nam0, nam1, expr, body } => {
        write!(f, "dup {} {} = {}; {}", nam0, nam1, expr, body)
      }
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
        write!(
          f,
          "({} {})",
          expr,
          args.iter().map(|x| format!("{}", x)).collect::<Vec<String>>().join(" ")
        )
      }
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
    write!(
      f,
      "{}",
      self.rules.iter().map(|rule| format!("{}", rule)).collect::<Vec<String>>().join("\n")
    )
  }
}

// Parser
// ======

pub fn parse_let(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    HOPA::do_there_take_exact("let "),
    Box::new(|state| {
      let (state, _) = HOPA::force_there_take_exact("let ", state)?;
      let (state, name) = HOPA::there_nonempty_name(state)?;
      let (state, _) = HOPA::force_there_take_exact("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _) = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Let { name, expr, body })))
    }),
    state,
  )
}

pub fn parse_dup(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    HOPA::do_there_take_exact("dup "),
    Box::new(|state| {
      let (state, _) = HOPA::force_there_take_exact("dup ", state)?;
      let (state, nam0) = HOPA::there_nonempty_name(state)?;
      let (state, nam1) = HOPA::there_nonempty_name(state)?;
      let (state, _) = HOPA::force_there_take_exact("=", state)?;
      let (state, expr) = parse_term(state)?;
      let (state, _) = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Dup { nam0, nam1, expr, body })))
    }),
    state,
  )
}

pub fn parse_sup(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
    HOPA::do_there_take_exact("{"),
    Box::new(move |state| {
      let (state, _) = HOPA::force_there_take_exact("{", state)?;
      let (state, val0) = parse_term(state)?;
      let (state, val1) = parse_term(state)?;
      let (state, _) = HOPA::force_there_take_exact("}", state)?;
      Ok((state, Box::new(Term::Sup { val0, val1 })))
    }),
    state,
  )
}

pub fn parse_lam(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  let parse_symbol = |x| {
    return HOPA::any(&[HOPA::do_there_take_exact("λ"), HOPA::do_there_take_exact("@")], x);
  };
  HOPA::guard(
    Box::new(parse_symbol),
    Box::new(move |state| {
      let (state, _) = parse_symbol(state)?;
      let (state, name) = HOPA::there_name(state)?;
      let (state, body) = parse_term(state)?;
      Ok((state, Box::new(Term::Lam { name, body })))
    }),
    state,
  )
}

pub fn parse_app(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  HOPA::guard(
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
  )
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
        (state, vec![])
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
          Ok((
            state,
            Box::new(Term::U6O { numb: u60::new(u64::from_str_radix(&text[2..], 16).unwrap()) }),
          ))
        } else {
          if text.find(".").is_some() {
            Ok((state, Box::new(Term::F6O { numb: f60::new(text.parse::<f64>().unwrap()) })))
          } else {
            Ok((state, Box::new(Term::U6O { numb: u60::new(text.parse::<u64>().unwrap()) })))
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
    HOPA::attempt(
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
      let (state, _) = HOPA::there_take_exact("%", state)?;
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
      let (state, _) = HOPA::force_there_take_exact("ask ", state)?;
      let (state, name) = HOPA::there_nonempty_name(state)?;
      let (state, _) = HOPA::force_there_take_exact("=", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _) = HOPA::there_take_exact(";", state)?;
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
      let (state, _) = HOPA::force_there_take_exact("ask ", state)?;
      let (state, func) = parse_term(state)?;
      let (state, _) = HOPA::there_take_exact(";", state)?;
      let (state, body) = parse_term(state)?;
      Ok((
        state,
        Box::new(Term::App { func, argm: Box::new(Term::Lam { name: "*".to_string(), body }) }),
      ))
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
      let mut chars: Vec<char> = vec![];
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
      let empty = Term::Ctr { name: "String.nil".to_string(), args: vec![] };
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
      // let mut elems: Vec<Box<Term>> = vec![];
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
      let empty = Term::Ctr { name: "List.nil".to_string(), args: vec![] };
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
      let (state, _) = HOPA::force_there_take_exact("if ", state)?;
      let (state, cond) = parse_term(state)?;
      let (state, _) = HOPA::force_there_take_exact("{", state)?;
      let (state, if_t) = parse_term(state)?;
      let (state, _) = HOPA::force_there_take_exact("}", state)?;
      let (state, _) = HOPA::force_there_take_exact("else", state)?;
      let (state, _) = HOPA::force_there_take_exact("{", state)?;
      let (state, if_f) = parse_term(state)?;
      let (state, _) = HOPA::force_there_take_exact("}", state)?;
      Ok((state, Box::new(Term::Ctr { name: "U60.if".to_string(), args: vec![cond, if_t, if_f] })))
    }),
    state,
  );
}

pub fn parse_bng(state: HOPA::State) -> HOPA::Answer<Option<Box<Term>>> {
  return HOPA::guard(
    HOPA::do_there_take_exact("!"),
    Box::new(|state| {
      let (state, _) = HOPA::force_there_take_exact("!", state)?;
      let (state, term) = parse_term(state)?;
      Ok((state, term))
    }),
    state,
  );
}

pub fn parse_term(state: HOPA::State) -> HOPA::Answer<Box<Term>> {
  HOPA::attempt(
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
      Box::new(parse_if_sugar),
      Box::new(parse_bng),
      Box::new(parse_ask_sugar_named),
      Box::new(parse_ask_sugar_anon),
      Box::new(parse_var),
      Box::new(|state| Ok((state, None))),
    ],
    state,
  )
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
    let (state, _) = parse_term(state)?;
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
  let mut rules = vec![];
  let mut smaps = vec![];
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
