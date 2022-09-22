use crate::language as lang;
use crate::runtime as rt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::iter;
//use std::time::Instant;

// RuleBook
// ========

// A RuleBook is a file ready for compilation. It includes:
// - rule_group: sanitized rules grouped by function
// - id_to_name: maps ctr ids to names
// - name_to_id: maps ctr names to ids
// - ctr_is_cal: true if a ctr is used as a function
// A sanitized rule has all its variables renamed to have unique names.
// Variables that are never used are renamed to "*".
#[derive(Debug)]
pub struct RuleBook {
  pub rule_group: HashMap<String, RuleGroup>,
  pub name_count: u64,
  pub id_to_name: HashMap<u64, String>,
  pub name_to_id: HashMap<String, u64>,
  pub id_to_arit: HashMap<u64, u64>,
  pub ctr_is_cal: HashMap<String, bool>,
}

pub type RuleGroup = (usize, Vec<lang::Rule>);

// Creates an empty rulebook
pub fn new_rulebook() -> RuleBook {
  // Creates an empty book
  let mut book = RuleBook {
    rule_group: HashMap::new(),
    name_count: 0,
    name_to_id: HashMap::new(),
    id_to_name: HashMap::new(),
    id_to_arit: HashMap::new(),
    ctr_is_cal: HashMap::new(),
  };
  fn register_name(book: &mut RuleBook, name: &str, ctid: u64) {
    let name = name.to_string();
    book.name_count = book.name_count + 1;
    book.name_to_id.insert(name.clone(), ctid);
    book.id_to_name.insert(ctid, name.clone());
  }
  fn register(book: &mut RuleBook, name: &str, ctid: u64, arity: u64, is_fun: bool) {
    register_name(book, name, ctid);
    let name = name.to_string();
    book.id_to_arit.insert(ctid, arity);
    book.ctr_is_cal.insert(name.clone(), is_fun);
  }
  register(&mut book, "HVM.log", rt::HVM_LOG, 2, true); // HVM.log a b : b
  register(&mut book, "HVM.put", rt::HVM_PUT, 2, true); // HVM.put a b : b
  register(&mut book, "String.nil", rt::STRING_NIL, 0, false); // String.nil : String
  register(&mut book, "String.cons", rt::STRING_CONS, 2, false); // String.cons (head: U60) (tail: String) : String
  register(&mut book, "IO.done", rt::IO_DONE, 1, false); // IO.done a : (IO a)
  register(&mut book, "IO.do_input", rt::IO_DO_INPUT, 1, false); // IO.do_input (String -> IO a) : (IO a)
  register(&mut book, "IO.do_output", rt::IO_DO_OUTPUT, 2, false); // IO.do_output String (Num -> IO a) : (IO a)
  register(&mut book, "IO.do_fetch", rt::IO_DO_FETCH, 3, false); // IO.do_fetch String Options (String -> IO a) : (IO a)
  register(&mut book, "IO.do_store", rt::IO_DO_STORE, 3, false); // IO.do_store String String (Num -> IO a) : (IO a)
  register(&mut book, "IO.do_load", rt::IO_DO_LOAD, 2, false); // IO.do_load String (String -> IO a) : (IO a)
//GENERATED-BOOK-ENTRIES//
  register_name(&mut book, "Kind.Term.ct0", rt::HOAS_CT0);
  register_name(&mut book, "Kind.Term.ct1", rt::HOAS_CT1);
  register_name(&mut book, "Kind.Term.ct2", rt::HOAS_CT2);
  register_name(&mut book, "Kind.Term.ct3", rt::HOAS_CT3);
  register_name(&mut book, "Kind.Term.ct4", rt::HOAS_CT4);
  register_name(&mut book, "Kind.Term.ct5", rt::HOAS_CT5);
  register_name(&mut book, "Kind.Term.ct6", rt::HOAS_CT6);
  register_name(&mut book, "Kind.Term.ct7", rt::HOAS_CT7);
  register_name(&mut book, "Kind.Term.ct8", rt::HOAS_CT8);
  register_name(&mut book, "Kind.Term.ct9", rt::HOAS_CT9);
  register_name(&mut book, "Kind.Term.ctA", rt::HOAS_CTA);
  register_name(&mut book, "Kind.Term.ctB", rt::HOAS_CTB);
  register_name(&mut book, "Kind.Term.ctC", rt::HOAS_CTC);
  register_name(&mut book, "Kind.Term.ctD", rt::HOAS_CTD);
  register_name(&mut book, "Kind.Term.ctE", rt::HOAS_CTE);
  register_name(&mut book, "Kind.Term.ctF", rt::HOAS_CTF);
  register_name(&mut book, "Kind.Term.ctG", rt::HOAS_CTG);
  register_name(&mut book, "Kind.Term.num", rt::HOAS_NUM);
  return book;
}

// Adds a group to a rulebook
pub fn add_group(book: &mut RuleBook, name: &str, group: &RuleGroup) {
  fn register_names_and_arities(book: &mut RuleBook, term: &lang::Term) {
    match term {
      lang::Term::Dup { expr, body, .. } => {
        register_names_and_arities(book, expr);
        register_names_and_arities(book, body);
      }
      lang::Term::Let { expr, body, .. } => {
        register_names_and_arities(book, expr);
        register_names_and_arities(book, body);
      }
      lang::Term::Lam { body, .. } => {
        register_names_and_arities(book, body);
      }
      lang::Term::App { func, argm, .. } => {
        register_names_and_arities(book, func);
        register_names_and_arities(book, argm);
      }
      lang::Term::Op2 { val0, val1, .. } => {
        register_names_and_arities(book, val0);
        register_names_and_arities(book, val1);
      }
      term@lang::Term::Ctr { name, args } => {
        // Registers id
        let id = match book.name_to_id.get(name) {
          None => {
            let id = book.name_count;
            book.name_to_id.insert(name.clone(), id);
            book.id_to_name.insert(id, name.clone());
            book.name_count += 1;
            id
          }
          Some(id) => {
            *id
          }
        };
        // Registers arity
        if let Some(arit) = book.id_to_arit.get(&id) {
          if *arit != args.len() as u64 {
            panic!("Incorrect arity on {}.", term);
          }
        } else {
          book.id_to_arit.insert(id, args.len() as u64);
        }
        // Recurses
        for arg in args {
          register_names_and_arities(book, arg);
        }
      }
      _ => (),
    }
  }

  // Inserts the group on the book
  book.rule_group.insert(name.to_string(), group.clone());

  // Builds its metadata (name_to_id, id_to_name, ctr_is_cal)
  for rule in &group.1 {
    register_names_and_arities(book, &rule.lhs);
    register_names_and_arities(book, &rule.rhs);
    if let lang::Term::Ctr { ref name, .. } = *rule.lhs {
      book.ctr_is_cal.insert(name.clone(), true);
    }
  }
}

// Converts a file to a rulebook
pub fn gen_rulebook(file: &lang::File) -> RuleBook {
  // Creates an empty rulebook
  let mut book = new_rulebook();

  // Flattens, sanitizes and groups this file's rules
  let groups = group_rules(&sanitize_rules(&flatten(&file.rules)));

  // Adds each group
  for (name, group) in groups.iter() {
    add_group(&mut book, name, group);
  }

  book
}

// Groups rules by name. For example:
//   (add (succ a) (succ b)) = (succ (succ (add a b)))
//   (add (succ a) (zero)  ) = (succ a)
//   (add (zero)   (succ b)) = (succ b)
//   (add (zero)   (zero)  ) = (zero)
// This is a group of 4 rules starting with the "add" name.
pub fn group_rules(rules: &[lang::Rule]) -> HashMap<String, RuleGroup> {
  let mut groups: HashMap<String, RuleGroup> = HashMap::new();
  for rule in rules {
    if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
      let group = groups.get_mut(name);
      match group {
        None => {
          groups.insert(name.clone(), (args.len(), Vec::from([rule.clone()])));
        }
        Some((_arity, rules)) => {
          rules.push(rule.clone());
        }
      }
    }
  }
  groups
}

pub fn is_global_name(name: &str) -> bool {
  !name.is_empty() && name.starts_with(&"$")
}

// Sanitize
// ========

#[allow(dead_code)]
pub struct SanitizedRule {
  pub rule: lang::Rule,
  pub uses: HashMap<String, u64>,
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
pub fn sanitize_rule(rule: &lang::Rule) -> Result<lang::Rule, String> {
  // Pass through the lhs of the function generating new names
  // for every variable found in the style described before with
  // the fresh function. Also checks if rule's left side is valid.
  // BTree is used here for determinism (HashMap does not maintain
  // order among executions)
  type NameTable = BTreeMap<String, String>;
  fn create_fresh(
    rule: &lang::Rule,
    fresh: &mut dyn FnMut() -> String,
  ) -> Result<NameTable, String> {
    let mut table = BTreeMap::new();

    let lhs = &rule.lhs;
    if let lang::Term::Ctr { name: _, ref args } = **lhs {
      for arg in args {
        match &**arg {
          lang::Term::Var { name, .. } => {
            table.insert(name.clone(), fresh());
          }
          lang::Term::Ctr { args, .. } => {
            for arg in args {
              if let lang::Term::Var { name } = &**arg {
                table.insert(name.clone(), fresh());
              }
            }
          }
          lang::Term::Num { .. } => {}
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
    term: &lang::Term,
    lhs: bool,
    tbl: &mut NameTable,
    ctx: &mut CtxSanitizeTerm,
  ) -> Result<Box<lang::Term>, String> {
    fn rename_erased(name: &mut String, uses: &HashMap<String, u64>) {
      if !is_global_name(name) && uses.get(name).copied() <= Some(0) {
        *name = "*".to_string();
      }
    }
    let term = match term {
      lang::Term::Var { name } => {
        if lhs {
          let mut name = tbl.get(name).unwrap_or(name).clone();
          rename_erased(&mut name, ctx.uses);
          Box::new(lang::Term::Var { name })
        } else if is_global_name(name) {
          if tbl.get(name).is_some() {
            panic!("Using a global variable more than once isn't supported yet. Use an explicit 'let' to clone it. {} {:?}", name, tbl.get(name));
          } else {
            tbl.insert(name.clone(), String::new());
            Box::new(lang::Term::Var { name: name.clone() })
          }
        } else {
          // create a var with the name generated before
          // concatenated with '.{{times_used}}'
          if let Some(name) = tbl.get(name) {
            let used = { *ctx.uses.entry(name.clone()).and_modify(|x| *x += 1).or_insert(1) };
            let name = format!("{}.{}", name, used - 1);
            Box::new(lang::Term::Var { name })
          //} else if is_global_name(&name) {
          // println!("Allowed unbound variable: {}", name);
          // Box::new(lang::Term::Var { name: name.clone() })
          } else {
            return Err(format!("Unbound variable: `{}`.", name));
          }
        }
      }
      lang::Term::Dup { expr, body, nam0, nam1 } => {
        let new_nam0 = (ctx.fresh)();
        let new_nam1 = (ctx.fresh)();
        let expr = sanitize_term(expr, lhs, tbl, ctx)?;
        let got_nam0 = tbl.remove(nam0);
        let got_nam1 = tbl.remove(nam1);
        tbl.insert(nam0.clone(), new_nam0.clone());
        tbl.insert(nam1.clone(), new_nam1.clone());
        let body = sanitize_term(body, lhs, tbl, ctx)?;
        tbl.remove(nam0);
        if let Some(x) = got_nam0 {
          tbl.insert(nam0.clone(), x);
        }
        tbl.remove(nam1);
        if let Some(x) = got_nam1 {
          tbl.insert(nam1.clone(), x);
        }
        let nam0 = format!("{}.0", new_nam0);
        let nam1 = format!("{}.0", new_nam1);
        let term = lang::Term::Dup { nam0, nam1, expr, body };
        Box::new(term)
      }
      lang::Term::Let { name, expr, body } => {
        let new_name = (ctx.fresh)();
        let expr = sanitize_term(expr, lhs, tbl, ctx)?;
        let got_name = tbl.remove(name);
        tbl.insert(name.clone(), new_name.clone());
        let body = sanitize_term(body, lhs, tbl, ctx)?;
        tbl.remove(name);
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        duplicator(&new_name, expr, body, ctx.uses)
      }
      lang::Term::Lam { name, body } => {
        let mut new_name = if is_global_name(name) { name.clone() } else { (ctx.fresh)() };
        let got_name = tbl.remove(name);
        tbl.insert(name.clone(), new_name.clone());
        let body = sanitize_term(body, lhs, tbl, ctx)?;
        tbl.remove(name);
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        let expr = Box::new(lang::Term::Var { name: new_name.clone() });
        let body = duplicator(&new_name, expr, body, ctx.uses);
        rename_erased(&mut new_name, ctx.uses);
        let term = lang::Term::Lam { name: new_name, body };
        Box::new(term)
      }
      lang::Term::App { func, argm } => {
        let func = sanitize_term(func, lhs, tbl, ctx)?;
        let argm = sanitize_term(argm, lhs, tbl, ctx)?;
        let term = lang::Term::App { func, argm };
        Box::new(term)
      }
      lang::Term::Ctr { name, args } => {
        let mut n_args = Vec::with_capacity(args.len());
        for arg in args {
          let arg = sanitize_term(arg, lhs, tbl, ctx)?;
          n_args.push(arg);
        }
        let term = lang::Term::Ctr { name: name.clone(), args: n_args };
        Box::new(term)
      }
      lang::Term::Op2 { oper, val0, val1 } => {
        let val0 = sanitize_term(val0, lhs, tbl, ctx)?;
        let val1 = sanitize_term(val1, lhs, tbl, ctx)?;
        let term = lang::Term::Op2 { oper: *oper, val0, val1 };
        Box::new(term)
      }
      lang::Term::Num { numb } => {
        let term = lang::Term::Num { numb: *numb };
        Box::new(term)
      }
    };

    Ok(term)
  }

  // Duplicates all variables that are used more than once.
  // The process is done generating auxiliary variables and
  // applying dup on them.
  fn duplicator(
    name: &str,
    expr: Box<lang::Term>,
    body: Box<lang::Term>,
    uses: &HashMap<String, u64>,
  ) -> Box<lang::Term> {
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
            let term = lang::Term::Let { name: format!("{}.0", name), expr, body };
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
            let dup = lang::Term::Dup {
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

  // Recursive aux function to duplicate one variable
  // an amount of times
  fn duplicator_go(
    i: u64,
    duplicated_times: u64,
    body: Box<lang::Term>,
    vars: &mut Vec<String>,
  ) -> Box<lang::Term> {
    if i == duplicated_times {
      body
    } else {
      let nam0 = vars.pop().unwrap();
      let nam1 = vars.pop().unwrap();
      let exp0 = Box::new(lang::Term::Var { name: format!("c.{}", i - 1) });
      Box::new(lang::Term::Dup {
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
  let mut ctx = CtxSanitizeTerm { uses: &mut uses, fresh: &mut fresh };

  // sanitize left and right sides
  let mut rhs = sanitize_term(&rule.rhs, false, &mut table.clone(), &mut ctx)?;
  let lhs = sanitize_term(&rule.lhs, true, &mut table.clone(), &mut ctx)?;

  // duplicate right side variables that are used more than once
  for (_key, value) in table {
    let expr = Box::new(lang::Term::Var { name: value.clone() });
    rhs = duplicator(&value, expr, rhs, &uses);
  }

  // returns the sanitized rule
  Ok(lang::Rule { lhs, rhs })
}

// Sanitizes all rules in a vector
pub fn sanitize_rules(rules: &[lang::Rule]) -> Vec<lang::Rule> {
  rules
    .iter()
    .map(|rule| {
      match sanitize_rule(rule) {
        Ok(rule) => rule,
        Err(err) => {
          println!("{}", err);
          println!("On rule: `{}`.", rule);
          std::process::exit(0); // FIXME: avoid this, propagate this error upwards
        }
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use core::panic;

  use super::{gen_rulebook, sanitize_rule};
  use crate::language::{read_file, read_rule};

  #[test]
  fn test_sanitize_expected_code() {
    // code and expected code after sanitize
    let codes = [
      (
        "(Foo a b c) = (+c (+c (+b (+ b b))))",
        "(Foo * x1 x2) = dup x2.0 x2.1 = x2; dup c.0 x1.0 = x1; dup x1.1 x1.2 = c.0; (+ x2.0 (+ x2.1 (+ x1.0 (+ x1.1 x1.2))))",
      ),
      (
        "(Foo a b c d e f g h i j k l m n) = (+ (+ a a) i)",
        "(Foo x0 * * * * * * * x8 * * * * *) = let x8.0 = x8; dup x0.0 x0.1 = x0; (+ (+ x0.0 x0.1) x8.0)"
      ),
      (
        "(Double (Zero)) = (Zero)",
        "(Double (Zero)) = (Zero)"
      ),
      (
        "(Double (Succ a)) = (Double (Succ (Succ a)))",
        "(Double (Succ x0)) = let x0.0 = x0; (Double (Succ (Succ x0.0)))"
      )
    ];

    // test if after sanitize all are equal
    // to the expected
    for (code, expected) in codes {
      let rule = read_rule(code).unwrap();
      match rule {
        None => panic!("Rule not parsed"),
        Some(v) => {
          let result = sanitize_rule(&v);
          match result {
            Ok(rule) => assert_eq!(rule.to_string(), expected),
            Err(_) => panic!("Rule not sanitized"),
          }
        }
      }
    }
  }

  // FIXME: panicking
  #[test]
  fn test_sanitize_fail_code() {
    // code that has to fail
    const FAILS: [&str; 2] = [
      // more than one nesting in constructors
      "(Foo (Bar (Zaz x))) = (x)",
      // variable not declared in lhs
      "(Succ x) = (j)",
    ];

    for code in FAILS {
      let rule = read_rule(code).unwrap();
      match rule {
        None => panic!("Rule not parsed"),
        Some(v) => {
          let result = sanitize_rule(&v);
          assert!(matches!(result, Err(_)));
        }
      }
    }
  }

  #[test]
  fn test_rulebook_expected() {
    let file = "
      (Double (Zero)) = (Zero)
      (Double (Succ x)) = (Succ ( Succ (Double x)))
    ";

    let file = read_file(file).unwrap();
    let rulebook = gen_rulebook(&file);

    // rule_group testing
    // contains expected key
    assert!(rulebook.rule_group.contains_key("Double"));
    // contains expected number of keys
    assert_eq!(rulebook.rule_group.len(), 1);
    // key contains expected number of rules
    assert_eq!(rulebook.rule_group.get("Double").unwrap().1.len(), 2);
    // key contains expected arity
    assert_eq!(rulebook.rule_group.get("Double").unwrap().0, 1);

    // id_to_name e name_to_id testing
    // check expected length
    assert_eq!(rulebook.id_to_name.len(), 3);
    // check determinism and existence
    assert_eq!(rulebook.id_to_name.get(&0).unwrap(), "Double");
    assert_eq!(rulebook.id_to_name.get(&1).unwrap(), "Zero");
    assert_eq!(rulebook.id_to_name.get(&2).unwrap(), "Succ");
    // check cohesion
    let _size = rulebook.id_to_name.len();
    for (id, name) in rulebook.id_to_name {
      // assert name_to_id id will have same
      // id that generate name in id_to_name
      // also checks if the two maps have same length
      let id_to_compare = rulebook.name_to_id.get(&name).unwrap();
      assert_eq!(*id_to_compare, id);
    }

    // ctr_is_cal testing
    // expected key exist
    assert!(rulebook.ctr_is_cal.contains_key("Double"));
    // contains expected number of keys
    assert_eq!(rulebook.ctr_is_cal.len(), 1);
    // key contains expected value
    assert!(*rulebook.ctr_is_cal.get("Double").unwrap());
  }
}

pub fn subst(term: &mut lang::Term, sub_name: &str, value: &lang::Term) {
  match term {
    lang::Term::Var { name } => {
      if sub_name == name {
        *term = value.clone();
      }
    }
    lang::Term::Dup { nam0, nam1, expr, body } => {
      subst(&mut *expr, sub_name, value);
      if nam0 != sub_name && nam1 != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    lang::Term::Let { name, expr, body } => {
      subst(&mut *expr, sub_name, value);
      if name != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    lang::Term::Lam { name, body } => {
      if name != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    lang::Term::App { func, argm } => {
      subst(&mut *func, sub_name, value);
      subst(&mut *argm, sub_name, value);
    }
    lang::Term::Ctr { args, .. } => {
      for arg in args {
        subst(&mut *arg, sub_name, value);
      }
    }
    lang::Term::Num { .. } => {}
    lang::Term::Op2 { val0, val1, .. } => {
      subst(&mut *val0, sub_name, value);
      subst(&mut *val1, sub_name, value);
    }
  }
}

// Split rules that have nested cases, flattening them.
// I'm not proud of this code. Must improve considerably.
pub fn flatten(rules: &[lang::Rule]) -> Vec<lang::Rule> {
  // Unique name generator
  let mut name_count = 0;
  fn fresh(name_count: &mut u64) -> u64 {
    let name = *name_count;
    *name_count += 1;
    name
  }

  // Checks if this rule has nested patterns, and must be splitted
  #[rustfmt::skip]
  fn must_split(lhs: &lang::Term) -> bool {
/**/if let lang::Term::Ctr { ref args, .. } = *lhs {
/*  */for arg in args {
/*  H */if let lang::Term::Ctr { args: ref arg_args, .. } = **arg {
/*   A  */for field in arg_args {
/*    D   */if is_matchable(field) {
/* ─=≡ΣO)   */return true;
/*    U   */}
/*   K  */}
/*  E */}
/* N*/}
/**/} false
  }

  // Returns true if a rule is a global default case (all args are vars)
  //fn is_default(lhs: &lang::Term) -> bool {
///**/if let lang::Term::Ctr { ref args, .. } = *lhs {
///*  */for arg in args {
///*  H */if let lang::Term::Ctr { args: ref arg_args, .. } = **arg {
///*   A  */for field in arg_args {
///*    D   */if !is_variable(field) {
///* ─=≡ΣO)   */return false;
///*    U   */}
///*   K  */}
///*  E */}
///* N*/}
///**/} true
  //}

  fn is_matchable(term: &lang::Term) -> bool {
    matches!(term, lang::Term::Ctr { .. } | lang::Term::Num { .. })
  }

  //fn is_variable(term: &lang::Term) -> bool {
    //matches!(term, lang::Term::Var { .. })
  //}


  // Checks true if every time that `a` matches, `b` will match too
  fn matches_together(a: &lang::Rule, b: &lang::Rule) -> (bool, bool) {
    let mut same_shape = true;
    if let (
      lang::Term::Ctr { name: ref _a_name, args: ref a_args },
      lang::Term::Ctr { name: ref _b_name, args: ref b_args },
    ) = (&*a.lhs, &*b.lhs) {
      for (a_arg, b_arg) in a_args.iter().zip(b_args) {
        match **a_arg {
          lang::Term::Ctr { name: ref a_arg_name, args: ref a_arg_args } => match **b_arg {
            lang::Term::Ctr { name: ref b_arg_name, args: ref b_arg_args } => {
              if a_arg_name != b_arg_name || a_arg_args.len() != b_arg_args.len() {
                return (false, false);
              }
            }
            lang::Term::Num { .. } => {
              return (false, false);
            }
            lang::Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          lang::Term::Num { numb: a_arg_numb } => match **b_arg {
            lang::Term::Num { numb: b_arg_numb } => {
              if a_arg_numb != b_arg_numb {
                return (false, false);
              }
            }
            lang::Term::Ctr { .. } => {
              return (false, false);
            }
            lang::Term::Var { .. } => {
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

  fn split_group(rules: &[lang::Rule], name_count: &mut u64) -> Vec<lang::Rule> {
    // println!("\n[split_group]");
    // for rule in rules {
    //   println!("{}", rule);
    // }
    let mut skip: HashSet<usize> = HashSet::new();
    let mut new_rules: Vec<lang::Rule> = Vec::new();
    for i in 0..rules.len() {
      if !skip.contains(&i) {
        let rule = &rules[i];
        if must_split(&rule.lhs) {
          if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
            let mut new_group: Vec<lang::Rule> = Vec::new();
            let new_lhs_name: String = name.clone();
            let new_rhs_name: String = format!("{}.{}", name, fresh(name_count));
            let mut new_lhs_args: Vec<Box<lang::Term>> = Vec::new();
            let mut new_rhs_args: Vec<Box<lang::Term>> = Vec::new();
            for arg in args {
              match &**arg {
                lang::Term::Ctr { name: ref arg_name, args: ref arg_args } => {
                  let new_arg_name = arg_name.clone();
                  let mut new_arg_args = Vec::new();
                  for field in arg_args {
                    match &**field {
                      lang::Term::Ctr { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                      }
                      lang::Term::Num { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                      }
                      lang::Term::Var { .. } => {
                        new_arg_args.push(field.clone());
                        new_rhs_args.push(field.clone());
                      }
                      _ => {
                        panic!("?");
                      }
                    }
                  }
                  new_lhs_args.push(Box::new(lang::Term::Ctr { name: new_arg_name, args: new_arg_args }));
                }
                lang::Term::Var { .. } => {
                  new_lhs_args.push(Box::new(*arg.clone()));
                  new_rhs_args.push(Box::new(*arg.clone()));
                }
                _ => {}
              }
            }
            //(Foo Tic (Bar a b) (Haz c d)) = A
            //(Foo Tic x         y)         = B
            //---------------------------------
            //(Foo Tic (Bar a b) (Haz c d)) = B[x <- (Bar a b), y <- (Haz c d)]
            //
            //(Foo.0 a b c d) = ...
            let new_lhs = Box::new(lang::Term::Ctr { name: new_lhs_name, args: new_lhs_args.clone() });
            let new_rhs = Box::new(lang::Term::Ctr { name: new_rhs_name.clone(), args: new_rhs_args });
            let new_rule = lang::Rule { lhs: new_lhs, rhs: new_rhs };
            new_group.push(new_rule);
            for (j, other) in rules.iter().enumerate().skip(i) {
              let (compatible, same_shape) = matches_together(&rule, &other);
              if compatible {
                if let (
                  lang::Term::Ctr { name: ref _rule_name, args: ref rule_args },
                  lang::Term::Ctr { name: ref _other_name, args: ref other_args }, 
                ) = (&*rule.lhs, &*other.lhs) {
                  // (Foo a     (B x P) (C y0 y1)) = F
                  // (Foo (A k) (B x Q) y        ) = G
                  // -----------------------------
                  // (Foo a (B x u) (C y0 y1)) = (Foo.0 a x u y0 y1)
                  //   (Foo.0 a     x P y0 y1) = F
                  //   (Foo.0 (A k) x Q f0 f1) = G [y <- (C f0 f1)] // f0 and f1 are fresh
                  if same_shape {
                    skip.insert(j); // avoids identical, duplicated clauses
                  }
                  let other_new_lhs_name = new_rhs_name.clone();
                  let mut other_new_lhs_args = Vec::new();
                  let mut other_new_rhs = other.rhs.clone();
                  for (rule_arg, other_arg) in rule_args.iter().zip(other_args) {
                    match &**rule_arg {
                      lang::Term::Ctr { name: ref rule_arg_name, args: ref rule_arg_args } => {
                        match &**other_arg {
                          lang::Term::Ctr { name: ref _other_arg_name, args: ref other_arg_args } => {
                            for other_field in other_arg_args {
                              other_new_lhs_args.push(other_field.clone());
                            }
                          }
                          lang::Term::Var { name: ref other_arg_name } => {
                            let mut new_ctr_args = vec![];
                            for _ in 0 .. rule_arg_args.len() {
                              let new_arg = lang::Term::Var { name: format!(".{}", fresh(name_count)) };
                              new_ctr_args.push(Box::new(new_arg.clone()));
                              other_new_lhs_args.push(Box::new(new_arg));
                            }
                            let new_ctr = lang::Term::Ctr { name: rule_arg_name.clone(), args: new_ctr_args };
                            subst(&mut other_new_rhs, other_arg_name, &new_ctr);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      lang::Term::Var { .. } => {
                        other_new_lhs_args.push(other_arg.clone());
                      }
                      lang::Term::Num { numb: rule_arg_numb } => {
                        match &**other_arg {
                          lang::Term::Num { numb: other_arg_numb } => {
                            if rule_arg_numb == other_arg_numb {
                              other_new_lhs_args.push(Box::new(*other_arg.clone()));
                            } else {
                              panic!("Internal error. Please report."); // not possible since it matches
                            }
                          }
                          lang::Term::Var { name: ref other_arg_name } => {
                            subst(&mut other_new_rhs, other_arg_name, &rule_arg);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      _ => {
                        panic!("Internal error. Please report."); // not possible since it matches
                      }
                    }
                  }
                  let other_new_lhs = Box::new(lang::Term::Ctr {
                    name: other_new_lhs_name,
                    args: other_new_lhs_args,
                  });
                  let new_rule = lang::Rule { lhs: other_new_lhs, rhs: other_new_rhs };
                  new_group.push(new_rule);
                }
              }
            }
            for rule in split_group(&new_group, name_count) {
              new_rules.push(rule);
            }
          } else {
            panic!("Invalid left-hand side.");
          }
        } else {
          new_rules.push(rules[i].clone());
        }
      }
    }
    new_rules
  }

  // Groups rules by function name
  let mut groups: HashMap<String, Vec<lang::Rule>> = HashMap::new();
  for rule in rules {
    if let lang::Term::Ctr { ref name, .. } = *rule.lhs {
      if let Some(group) = groups.get_mut(name) {
        group.push(rule.clone());
      } else {
        groups.insert(name.clone(), vec![rule.clone()]);
      }
    }
  }

  // For each group, split its internal rules
  let mut new_rules = Vec::new();
  for (_name, rules) in &groups {
    for rule in split_group(rules, &mut name_count) {
      new_rules.push(rule);
    }
  }

  // println!("\nresult:");
  // for rule in &new_rules {
  //   println!("{}", rule);
  // }

  new_rules
}

// todo: "dups" still needs to be moved out on `alloc_body` etc.
pub fn build_function(book: &RuleBook, fn_name: &str, rules: &[lang::Rule]) -> rt::Function {
  let mut is_strict = if let lang::Term::Ctr { name: _, ref args } = *rules[0].lhs {
    vec![false; args.len()]
  } else {
    panic!("invalid left-hand side: {}", rules[0].lhs);
  };
  let hoas = fn_name.starts_with("f$");
  let dynrules = rules.iter().filter_map(|rule| {
    if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
      let mut cond = Vec::new();
      let mut vars = Vec::new();
      let mut inps = Vec::new();
      let mut free = Vec::new();
      if args.len() != is_strict.len() {
        panic!("inconsistent length of left-hand side on equation for '{}'.", name);
      }
      for ((i, arg), is_strict) in args.iter().enumerate().zip(is_strict.iter_mut()) {
        match &**arg {
          lang::Term::Ctr { name, args } => {
            *is_strict = true;
            cond.push(rt::Ctr(args.len() as u64, *book.name_to_id.get(&*name).unwrap_or(&0), 0));
            free.push((i as u64, args.len() as u64));
            for (j, arg) in args.iter().enumerate() {
              if let lang::Term::Var { ref name } = **arg {
                vars.push(rt::RuleVar { param: i as u64, field: Some(j as u64), erase: name == "*" });
                inps.push(name.clone());
              } else {
                panic!("sorry, left-hand sides can't have nested constructors yet.");
              }
            }
          }
          lang::Term::Num { numb } => {
            *is_strict = true;
            cond.push(rt::Num(*numb as u64));
          }
          lang::Term::Var { name } => {
            cond.push(rt::Var(0));
            vars.push(rt::RuleVar { param: i as u64, field: None, erase: name == "*" });
            inps.push(name.clone());
          }
          _ => {
            panic!("invalid left-hand side.");
          }
        }
      }

      let term = term_to_dynterm(book, &rule.rhs, &inps);
      let body = build_body(&term, vars.len() as u64);

      Some(rt::Rule { hoas, cond, vars, term, body, free })
    } else {
      None
    }
  }).collect();

  let arity = is_strict.len() as u64;
  let mut stricts = Vec::new();
  for (i, is_strict) in is_strict.iter().enumerate() {
    if *is_strict {
      stricts.push(i as u64);
    }
  }

  rt::Function {
    arity,
    is_strict: is_strict,
    stricts: stricts,
    rules: dynrules,
  }
}

pub fn hash<T: Hash>(t: &T) -> u64 {
  let mut s = DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

pub fn build_dynamic_arities(rb: &RuleBook) -> Vec<rt::Arity> {
  // todo: checking arity consistency would be nice, but where?
  fn find_arities(rb: &RuleBook, aris: &mut Vec<rt::Arity>, term: &lang::Term) {
    match term {
      lang::Term::Var { .. } => {}
      lang::Term::Dup { expr, body, .. } => {
        find_arities(rb, aris, expr);
        find_arities(rb, aris, body);
      }
      lang::Term::Lam { body, .. } => {
        find_arities(rb, aris, body);
      }
      lang::Term::Let { expr, body, .. } => {
        find_arities(rb, aris, expr);
        find_arities(rb, aris, body);
      }
      lang::Term::App { func, argm } => {
        find_arities(rb, aris, func);
        find_arities(rb, aris, argm);
      }
      lang::Term::Ctr { name, args } => {
        if let Some(id) = rb.name_to_id.get(name) {
          aris[*id as usize] = rt::Arity(args.len() as u64);
          for arg in args {
            find_arities(rb, aris, arg);
          }
        }
      }
      lang::Term::Num { .. } => {}
      lang::Term::Op2 { val0, val1, .. } => {
        find_arities(rb, aris, val0);
        find_arities(rb, aris, val1);
      }
    }
  }
  let mut aris: Vec<rt::Arity> = iter::repeat_with(|| rt::Arity(0)).take(65535).collect(); // fixme: hardcoded limit
  for rules_info in rb.rule_group.values() {
    for rule in &rules_info.1 {
      find_arities(rb, &mut aris, &rule.lhs);
      find_arities(rb, &mut aris, &rule.rhs);
    }
  }
  return aris;
}

pub fn build_dynamic_functions(book: &RuleBook) -> Vec<Option<rt::Function>> {
  let mut funs: Vec<Option<rt::Function>> = iter::repeat_with(|| None).take(65535).collect(); // fixme: hardcoded limit
  for (name, rules_info) in &book.rule_group {
    let fnid = book.name_to_id.get(name).unwrap_or(&0);
    let func = build_function(book, &name, &rules_info.1);
    funs[*fnid as usize] = Some(func);
  }
  funs
}

/// converts a language term to a runtime term
pub fn term_to_dynterm(book: &RuleBook, term: &lang::Term, inps: &[String]) -> rt::Term {
  fn convert_oper(oper: &lang::Oper) -> u64 {
    match oper {
      lang::Oper::Add => rt::ADD,
      lang::Oper::Sub => rt::SUB,
      lang::Oper::Mul => rt::MUL,
      lang::Oper::Div => rt::DIV,
      lang::Oper::Mod => rt::MOD,
      lang::Oper::And => rt::AND,
      lang::Oper::Or  => rt::OR,
      lang::Oper::Xor => rt::XOR,
      lang::Oper::Shl => rt::SHL,
      lang::Oper::Shr => rt::SHR,
      lang::Oper::Ltn => rt::LTN,
      lang::Oper::Lte => rt::LTE,
      lang::Oper::Eql => rt::EQL,
      lang::Oper::Gte => rt::GTE,
      lang::Oper::Gtn => rt::GTN,
      lang::Oper::Neq => rt::NEQ,
    }
  }

  #[allow(clippy::identity_op)]
  fn convert_term(
    term: &lang::Term,
    book: &RuleBook,
    depth: u64,
    vars: &mut Vec<String>,
  ) -> rt::Term {
    match term {
      lang::Term::Var { name } => {
        if let Some((idx, _)) = vars.iter().enumerate().rev().find(|(_, var)| var == &name) {
          rt::Term::Var { bidx: idx as u64 }
        } else {
          rt::Term::Glo { glob: hash(name) }
        }
      }
      lang::Term::Dup { nam0, nam1, expr, body } => {
        let eras = (nam0 == "*", nam1 == "*");
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(nam0.clone());
        vars.push(nam1.clone());
        let body = Box::new(convert_term(body, book, depth + 2, vars));
        vars.pop();
        vars.pop();
        rt::Term::Dup { eras, expr, body }
      }
      lang::Term::Lam { name, body } => {
        let glob = if is_global_name(name) { hash(name) } else { 0 };
        let eras = name == "*";
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        rt::Term::Lam { eras, glob, body }
      }
      lang::Term::Let { name, expr, body } => {
        let expr = Box::new(convert_term(expr, book, depth + 0, vars));
        vars.push(name.clone());
        let body = Box::new(convert_term(body, book, depth + 1, vars));
        vars.pop();
        rt::Term::Let { expr, body }
      }
      lang::Term::App { func, argm } => {
        let func = Box::new(convert_term(func, book, depth + 0, vars));
        let argm = Box::new(convert_term(argm, book, depth + 0, vars));
        rt::Term::App { func, argm }
      }
      lang::Term::Ctr { name, args } => {
        let term_func = *book.name_to_id.get(name).unwrap_or_else(|| panic!("unbound symbol: {}", name));
        let term_args = args.iter().map(|arg| convert_term(arg, book, depth + 0, vars)).collect();
        if *book.ctr_is_cal.get(name).unwrap_or(&false) {
          rt::Term::Fun { func: term_func, args: term_args }
        } else {
          rt::Term::Ctr { func: term_func, args: term_args }
        }
      }
      lang::Term::Num { numb } => rt::Term::Num { numb: *numb },
      lang::Term::Op2 { oper, val0, val1 } => {
        let oper = convert_oper(oper);
        let val0 = Box::new(convert_term(val0, book, depth + 0, vars));
        let val1 = Box::new(convert_term(val1, book, depth + 1, vars));
        rt::Term::Op2 { oper, val0, val1 }
      }
    }
  }

  let mut vars = inps.to_vec();
  convert_term(term, book, 0, &mut vars)
}

pub fn build_body(term: &rt::Term, free_vars: u64) -> rt::RuleBody {
  fn link(nodes: &mut [rt::RuleBodyNode], targ: u64, slot: u64, elem: rt::RuleBodyCell) {
    nodes[targ as usize][slot as usize] = elem;
    if let rt::RuleBodyCell::Loc { value, targ: var_targ, slot: var_slot } = elem {
      let tag = rt::get_tag(value);
      if tag <= rt::VAR {
        nodes[var_targ as usize][(var_slot + (tag & 0x01)) as usize] =
          rt::RuleBodyCell::Loc { value: rt::Arg(0), targ, slot };
      }
    }
  }
  fn alloc_lam(globs: &mut HashMap<u64, u64>, nodes: &mut Vec<rt::RuleBodyNode>, glob: u64) -> u64 {
    if let Some(targ) = globs.get(&glob) {
      *targ
    } else {
      let targ = nodes.len() as u64;
      nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; 2]);
      link(nodes, targ, 0, rt::RuleBodyCell::Fix { value: rt::Era() });
      if glob != 0 {
        globs.insert(glob, targ);
      }
      targ
    }
  }
  fn gen_elems(
    term: &rt::Term,
    dupk: &mut u64,
    vars: &mut Vec<rt::RuleBodyCell>,
    globs: &mut HashMap<u64, u64>,
    nodes: &mut Vec<rt::RuleBodyNode>,
    links: &mut Vec<(u64, u64, rt::RuleBodyCell)>,
  ) -> rt::RuleBodyCell {
    match term {
      rt::Term::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize]
        } else {
          panic!("unbound variable.");
        }
      }
      rt::Term::Glo { glob } => {
        let targ = alloc_lam(globs, nodes, *glob);
        rt::RuleBodyCell::Loc { value: rt::Var(0), targ, slot: 0 }
      }
      rt::Term::Dup { eras: _, expr, body } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; 3]);
        //let dupk = dups_count.next();
        links.push((targ, 0, rt::RuleBodyCell::Fix { value: rt::Era() }));
        links.push((targ, 1, rt::RuleBodyCell::Fix { value: rt::Era() }));
        let expr = gen_elems(expr, dupk, vars, globs, nodes, links);
        links.push((targ, 2, expr));
        vars.push(rt::RuleBodyCell::Loc { value: rt::Dp0(*dupk, 0), targ, slot: 0 });
        vars.push(rt::RuleBodyCell::Loc { value: rt::Dp1(*dupk, 0), targ, slot: 0 });
        *dupk = *dupk + 1;
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        vars.pop();
        vars.pop();
        body
      }
      rt::Term::Let { expr, body } => {
        let expr = gen_elems(expr, dupk, vars, globs, nodes, links);
        vars.push(expr);
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        vars.pop();
        body
      }
      rt::Term::Lam { eras: _, glob, body } => {
        let targ = alloc_lam(globs, nodes, *glob);
        let var = rt::RuleBodyCell::Loc { value: rt::Var(0), targ, slot: 0 };
        vars.push(var);
        let body = gen_elems(body, dupk, vars, globs, nodes, links);
        links.push((targ, 1, body));
        vars.pop();
        rt::RuleBodyCell::Loc { value: rt::Lam(0), targ, slot: 0 }
      }
      rt::Term::App { func, argm } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; 2]);
        let func = gen_elems(func, dupk, vars, globs, nodes, links);
        links.push((targ, 0, func));
        let argm = gen_elems(argm, dupk, vars, globs, nodes, links);
        links.push((targ, 1, argm));
        rt::RuleBodyCell::Loc { value: rt::App(0), targ, slot: 0 }
      }
      rt::Term::Fun { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, globs, nodes, links);
            links.push((targ, i as u64, arg));
          }
          rt::RuleBodyCell::Loc { value: rt::Fun(args.len() as u64, *func, 0), targ, slot: 0 }
        } else {
          rt::RuleBodyCell::Fix { value: rt::Fun(0, *func, 0) }
        }
      }
      rt::Term::Ctr { func, args } => {
        if !args.is_empty() {
          let targ = nodes.len() as u64;
          nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; args.len() as usize]);
          for (i, arg) in args.iter().enumerate() {
            let arg = gen_elems(arg, dupk, vars, globs, nodes, links);
            links.push((targ, i as u64, arg));
          }
          rt::RuleBodyCell::Loc { value: rt::Ctr(args.len() as u64, *func, 0), targ, slot: 0 }
        } else {
          rt::RuleBodyCell::Fix { value: rt::Ctr(0, *func, 0) }
        }
      }
      rt::Term::Num { numb } => rt::RuleBodyCell::Fix { value: rt::Num(*numb as u64) },
      rt::Term::Op2 { oper, val0, val1 } => {
        let targ = nodes.len() as u64;
        nodes.push(vec![rt::RuleBodyCell::Fix { value: 0 }; 2]);
        let val0 = gen_elems(val0, dupk, vars, globs, nodes, links);
        links.push((targ, 0, val0));
        let val1 = gen_elems(val1, dupk, vars, globs, nodes, links);
        links.push((targ, 1, val1));
        rt::RuleBodyCell::Loc { value: rt::Op2(*oper, 0), targ, slot: 0 }
      }
    }
  }

  let mut links: Vec<(u64, u64, rt::RuleBodyCell)> = Vec::new();
  let mut nodes: Vec<rt::RuleBodyNode> = Vec::new();
  let mut globs: HashMap<u64, u64> = HashMap::new();
  let mut vars: Vec<rt::RuleBodyCell> = (0..free_vars).map(|i| rt::RuleBodyCell::Ext { index: i }).collect();
  let mut dupk: u64 = 0;

  let elem = gen_elems(term, &mut dupk, &mut vars, &mut globs, &mut nodes, &mut links);
  for (targ, slot, elem) in links {
    link(&mut nodes, targ, slot, elem);
  }

  (elem, nodes, dupk)
}

pub fn alloc_closed_dynterm(heap: &mut rt::Heap, stat: &mut rt::Stat, term: &rt::Term) -> u64 {
  let host = rt::alloc(stat, 1);
  let body = build_body(term, 0);
  let term = rt::alloc_body(heap, stat, 0, &[], &body);
  rt::link(heap, host, term);
  host
}

pub fn alloc_term(heap: &mut rt::Heap, stat: &mut rt::Stat, book: &RuleBook, term: &lang::Term) -> u64 {
  alloc_closed_dynterm(heap, stat, &term_to_dynterm(book, term, &vec![]))
}

// Evaluates a HVM term to normal form
pub fn eval_code(
  call: &lang::Term,
  code: &str,
  debug: bool,
  size: usize,
) -> Result<(String, u64, u64, u64), String> {
  let mut heap = rt::new_heap();
  let mut stat = rt::new_stats();
  let mut info = rt::new_info();

  // Parses and reads the input file
  let file = lang::read_file(code)?;

  // Converts the HVM "file" to a Rulebook
  let book = gen_rulebook(&file);

  // Builds functions
  info.funs = build_dynamic_functions(&book);

  // Builds arities
  info.aris = build_dynamic_arities(&book);

  // builds names
  info.nams = book.id_to_name.clone();
  
  // Allocates the main term
  let host = alloc_term(&mut heap, &mut stat[0], &book, call); // FIXME tid?

  // Normalizes it
  let init = instant::Instant::now();
  #[cfg(not(target_arch = "wasm32"))]
  rt::normalize(&mut heap, &mut stat, &mut info, host, true);
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a string
  let code = format!("{}", crate::readback::as_term(&heap, &stat, &info, host));

  // Returns the normal form and the gas cost
  Ok((code, rt::get_cost(&stat), rt::get_used(&stat), time))
}

// notes
// -----

// hoas_opt: this is an internal optimization that allows us to simplify kind2's hoas generator.
// it will cause the default patterns of functions with a name starting with "f$" to only match
// productive hoas constructors (ct0, ct1, ..., ctg, num), as well as native numbers and
// constructors with 0-arity, which are used by kind2's hoas functions, unless it is the last
// (default) clause, which kind2 uses to quote a call back to low-order. this is an internal
// feature that won't affect programs other than kind2. we can remove this in a future, but that
// would require kind2 to replicate hvm's flattener algorithm, so we just use it instead.
