use crate::language as language;
use crate::runtime as runtime;
use std::collections::{BTreeMap, HashMap, HashSet};

// RuleBook
// ========

// A RuleBook is a file ready for compilation. It includes:
// - rule_group: sanitized rules grouped by function
// - id_to_name: maps ctr ids to names
// - name_to_id: maps ctr names to ids
// - ctr_is_fun: true if a ctr is used as a function
// A sanitized rule has all its variables renamed to have unique names.
// Variables that are never used are renamed to "*".
#[derive(Clone, Debug)]
pub struct RuleBook {
  pub rule_group: HashMap<String, RuleGroup>,
  pub name_count: u64,
  pub name_to_id: HashMap<String, u64>,
  pub id_to_smap: HashMap<u64, Vec<bool>>,
  pub id_to_name: HashMap<u64, String>,
  pub ctr_is_fun: HashMap<String, bool>,
}

pub type RuleGroup = (usize, Vec<language::syntax::Rule>);

// Creates an empty rulebook
pub fn new_rulebook() -> RuleBook {
  let mut book = RuleBook {
    rule_group: HashMap::new(),
    name_count: 0,
    name_to_id: HashMap::new(),
    id_to_smap: HashMap::new(),
    id_to_name: HashMap::new(),
    ctr_is_fun: HashMap::new(),
  };
  for precomp in runtime::PRECOMP {
    book.name_count += 1;
    book.name_to_id.insert(precomp.name.to_string(), precomp.id);
    book.id_to_name.insert(precomp.id, precomp.name.to_string());
    book.id_to_smap.insert(precomp.id, precomp.smap.to_vec());
    book.ctr_is_fun.insert(precomp.name.to_string(), precomp.funs.is_some());
  }
  book
}

// Adds a group to a rulebook
pub fn add_group(book: &mut RuleBook, name: &str, group: &RuleGroup) {
  fn register(book: &mut RuleBook, term: &language::syntax::Term, lhs_top: bool) {
    match term {
      language::syntax::Term::Dup { expr, body, .. } => {
        register(book, expr, false);
        register(book, body, false);
      }
      language::syntax::Term::Sup { val0, val1 } => {
        register(book, val0, false);
        register(book, val1, false);
      }
      language::syntax::Term::Let { expr, body, .. } => {
        register(book, expr, false);
        register(book, body, false);
      }
      language::syntax::Term::Lam { body, .. } => {
        register(book, body, false);
      }
      language::syntax::Term::App { func, argm, .. } => {
        register(book, func, false);
        register(book, argm, false);
      }
      language::syntax::Term::Op2 { val0, val1, .. } => {
        register(book, val0, false);
        register(book, val1, false);
      }
      term@language::syntax::Term::Ctr { name, args } => {
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
        // Registers smap
        match book.id_to_smap.get(&id) {
          None => {
            book.id_to_smap.insert(id, vec![false; args.len()]);
          }
          Some(smap) => {
            if smap.len() != args.len() {
              panic!("inconsistent arity on: '{}'", term);
            }
          }
        }
        // Force strictness when pattern-matching
        if lhs_top {
          (0 .. args.len()).for_each(|i| {
            let is_strict = matches!(*args[i], language::syntax::Term::Ctr { .. } | language::syntax::Term::U6O { .. } | language::syntax::Term::F6O { .. });
            if is_strict {
              book.id_to_smap.get_mut(&id).unwrap()[i] = true;
            }
          });
        }
        // Recurses
        for arg in args {
          register(book, arg, false);
        }
      }
      _ => (),
    }
  }

  // Inserts the group on the book
  book.rule_group.insert(name.to_string(), group.clone());

  // Builds its metadata (name_to_id, id_to_name, ctr_is_fun)
  for rule in &group.1 {
    register(book, &rule.lhs, true);
    register(book, &rule.rhs, false);
    if let language::syntax::Term::Ctr { ref name, .. } = *rule.lhs {
      book.ctr_is_fun.insert(name.clone(), true);
    }
  }
}

// Converts a file to a rulebook
pub fn gen_rulebook(file: &language::syntax::File) -> RuleBook {
  // Creates an empty rulebook
  let mut book = new_rulebook();

  // Flattens, sanitizes and groups this file's rules
  let groups = group_rules(&sanitize_rules(&flatten(&file.rules)));

  // Adds each group
  for (name, group) in groups.iter() {
    if book.name_to_id.get(name).unwrap_or(&u64::MAX) >= &runtime::PRECOMP_COUNT {
      add_group(&mut book, name, group);
    }
  }

  // Includes SMaps
  for (rule_name, rule_smap) in &file.smaps {
    let id = book.name_to_id.get(rule_name).unwrap();
    if book.id_to_smap.get(id).is_none() {
      book.id_to_smap.insert(*id, vec![false; rule_smap.len()]);
    }
    let smap = book.id_to_smap.get_mut(id).unwrap();
    for i in 0 .. smap.len() {
      if rule_smap[i] {
        smap[i] = true;
      }
    }
  }

  book
}

// Groups rules by name. For example:
//   (add (succ a) (succ b)) = (succ (succ (add a b)))
//   (add (succ a) (zero)  ) = (succ a)
//   (add (zero)   (succ b)) = (succ b)
//   (add (zero)   (zero)  ) = (zero)
// This is a group of 4 rules starting with the "add" name.
pub fn group_rules(rules: &[language::syntax::Rule]) -> HashMap<String, RuleGroup> {
  let mut groups: HashMap<String, RuleGroup> = HashMap::new();
  for rule in rules {
    if let language::syntax::Term::Ctr { ref name, ref args } = *rule.lhs {
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

// Sanitize
// ========

#[allow(dead_code)]
pub struct SanitizedRule {
  pub rule: language::syntax::Rule,
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
pub fn sanitize_rule(rule: &language::syntax::Rule) -> Result<language::syntax::Rule, String> {
  // Pass through the lhs of the function generating new names
  // for every variable found in the style described before with
  // the fresh function. Also checks if rule's left side is valid.
  // BTree is used here for determinism (HashMap does not maintain
  // order among executions)
  type NameTable = BTreeMap<String, String>;
  fn create_fresh(
    rule: &language::syntax::Rule,
    fresh: &mut dyn FnMut() -> String,
  ) -> Result<NameTable, String> {
    let mut table = BTreeMap::new();

    let lhs = &rule.lhs;
    if let language::syntax::Term::Ctr { name: _, ref args } = **lhs {
      for arg in args {
        match &**arg {
          language::syntax::Term::Var { name, .. } => {
            table.insert(name.clone(), fresh());
          }
          language::syntax::Term::Ctr { args, .. } => {
            for arg in args {
              if let language::syntax::Term::Var { name } = &**arg {
                table.insert(name.clone(), fresh());
              }
            }
          }
          language::syntax::Term::U6O { .. } => {}
          language::syntax::Term::F6O { .. } => {}
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
    term: &language::syntax::Term,
    lhs: bool,
    tbl: &mut NameTable,
    ctx: &mut CtxSanitizeTerm,
  ) -> Result<Box<language::syntax::Term>, String> {
    fn rename_erased(name: &mut String, uses: &HashMap<String, u64>) {
      if runtime::get_global_name_misc(name).is_none() && uses.get(name).copied() <= Some(0) {
        *name = "*".to_string();
      }
    }
    let term = match term {
      language::syntax::Term::Var { name } => {
        if lhs {
          let mut name = tbl.get(name).unwrap_or(name).clone();
          rename_erased(&mut name, ctx.uses);
          Box::new(language::syntax::Term::Var { name })
        } else if runtime::get_global_name_misc(name).is_some() {
          if tbl.get(name).is_some() {
            panic!("Using a global variable more than once isn't supported yet. Use an explicit 'let' to clone it. {} {:?}", name, tbl.get(name));
          } else {
            tbl.insert(name.clone(), String::new());
            Box::new(language::syntax::Term::Var { name: name.clone() })
          }
        } else {
          // create a var with the name generated before
          // concatenated with '.{{times_used}}'
          if let Some(name) = tbl.get(name) {
            let used = { *ctx.uses.entry(name.clone()).and_modify(|x| *x += 1).or_insert(1) };
            let name = format!("{}.{}", name, used - 1);
            Box::new(language::syntax::Term::Var { name })
          //} else if get_global_name_misc(&name) {
          // println!("Allowed unbound variable: {}", name);
          // Box::new(language::syntax::Term::Var { name: name.clone() })
          } else {
            return Err(format!("Unbound variable: `{}`.", name));
          }
        }
      }
      language::syntax::Term::Dup { expr, body, nam0, nam1 } => {
        let is_global_0 = runtime::get_global_name_misc(nam0).is_some();
        let is_global_1 = runtime::get_global_name_misc(nam1).is_some();
        if is_global_0 && runtime::get_global_name_misc(nam0) != Some(runtime::DP0) {
          panic!("The name of the global dup var '{}' must start with '$0'.", nam0);
        }
        if is_global_1 && runtime::get_global_name_misc(nam1) != Some(runtime::DP1) {
          panic!("The name of the global dup var '{}' must start with '$1'.", nam1);
        }
        if is_global_0 != is_global_1 {
          panic!("Both variables must be global: '{}' and '{}'.", nam0, nam1);
        }
        if is_global_0 && nam0[2..] != nam1[2..] {
          panic!("Global dup names must be identical: '{}' and '{}'.", nam0, nam1);
        }
        let new_nam0 = if is_global_0 { nam0.clone() } else { (ctx.fresh)() };
        let new_nam1 = if is_global_1 { nam1.clone() } else { (ctx.fresh)() };
        let expr = sanitize_term(expr, lhs, tbl, ctx)?;
        let got_nam0 = tbl.remove(nam0);
        let got_nam1 = tbl.remove(nam1);
        if !is_global_0 {
          tbl.insert(nam0.clone(), new_nam0.clone());
        }
        if !is_global_1 {
          tbl.insert(nam1.clone(), new_nam1.clone());
        }
        let body = sanitize_term(body, lhs, tbl, ctx)?;
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
        let term = language::syntax::Term::Dup { nam0, nam1, expr, body };
        Box::new(term)
      }
      language::syntax::Term::Sup { val0, val1 } => {
        let val0 = sanitize_term(val0, lhs, tbl, ctx)?;
        let val1 = sanitize_term(val1, lhs, tbl, ctx)?;
        let term = language::syntax::Term::Sup { val0, val1 };
        Box::new(term)
      }
      language::syntax::Term::Let { name, expr, body } => {
        if runtime::get_global_name_misc(name).is_some() {
          panic!("Global variable '{}' not allowed on let. Use dup instead.", name);
        }
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
      language::syntax::Term::Lam { name, body } => {
        let is_global = runtime::get_global_name_misc(name).is_some();
        let mut new_name = if is_global { name.clone() } else { (ctx.fresh)() };
        let got_name = tbl.remove(name);
        if !is_global {
          tbl.insert(name.clone(), new_name.clone());
        }
        let body = sanitize_term(body, lhs, tbl, ctx)?;
        if !is_global {
          tbl.remove(name);
        }
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        let expr = Box::new(language::syntax::Term::Var { name: new_name.clone() });
        let body = duplicator(&new_name, expr, body, ctx.uses);
        rename_erased(&mut new_name, ctx.uses);
        let term = language::syntax::Term::Lam { name: new_name, body };
        Box::new(term)
      }
      language::syntax::Term::App { func, argm } => {
        let func = sanitize_term(func, lhs, tbl, ctx)?;
        let argm = sanitize_term(argm, lhs, tbl, ctx)?;
        let term = language::syntax::Term::App { func, argm };
        Box::new(term)
      }
      language::syntax::Term::Ctr { name, args } => {
        let mut n_args = Vec::with_capacity(args.len());
        for arg in args {
          let arg = sanitize_term(arg, lhs, tbl, ctx)?;
          n_args.push(arg);
        }
        let term = language::syntax::Term::Ctr { name: name.clone(), args: n_args };
        Box::new(term)
      }
      language::syntax::Term::Op2 { oper, val0, val1 } => {
        let val0 = sanitize_term(val0, lhs, tbl, ctx)?;
        let val1 = sanitize_term(val1, lhs, tbl, ctx)?;
        let term = language::syntax::Term::Op2 { oper: *oper, val0, val1 };
        Box::new(term)
      }
      language::syntax::Term::U6O { numb } => {
        let term = language::syntax::Term::U6O { numb: *numb };
        Box::new(term)
      }
      language::syntax::Term::F6O { numb } => {
        let term = language::syntax::Term::F6O { numb: *numb };
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
    expr: Box<language::syntax::Term>,
    body: Box<language::syntax::Term>,
    uses: &HashMap<String, u64>,
  ) -> Box<language::syntax::Term> {
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
            let term = language::syntax::Term::Let { name: format!("{}.0", name), expr, body };
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
            let dup = language::syntax::Term::Dup {
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
    body: Box<language::syntax::Term>,
    vars: &mut Vec<String>,
  ) -> Box<language::syntax::Term> {
    if i == duplicated_times {
      body
    } else {
      let nam0 = vars.pop().unwrap();
      let nam1 = vars.pop().unwrap();
      let exp0 = Box::new(language::syntax::Term::Var { name: format!("c.{}", i - 1) });
      Box::new(language::syntax::Term::Dup {
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
    let expr = Box::new(language::syntax::Term::Var { name: value.clone() });
    rhs = duplicator(&value, expr, rhs, &uses);
  }

  // returns the sanitized rule
  Ok(language::syntax::Rule { lhs, rhs })
}

// Sanitizes all rules in a vector
pub fn sanitize_rules(rules: &[language::syntax::Rule]) -> Vec<language::syntax::Rule> {
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
  use crate::language::syntax::{read_file, read_rule};

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

    // ctr_is_fun testing
    // expected key exist
    assert!(rulebook.ctr_is_fun.contains_key("Double"));
    // contains expected number of keys
    assert_eq!(rulebook.ctr_is_fun.len(), 1);
    // key contains expected value
    assert!(*rulebook.ctr_is_fun.get("Double").unwrap());
  }
}

pub fn subst(term: &mut language::syntax::Term, sub_name: &str, value: &language::syntax::Term) {
  match term {
    language::syntax::Term::Var { name } => {
      if sub_name == name {
        *term = value.clone();
      }
    }
    language::syntax::Term::Dup { nam0, nam1, expr, body } => {
      subst(&mut *expr, sub_name, value);
      if nam0 != sub_name && nam1 != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    language::syntax::Term::Sup { val0, val1 } => {
      subst(&mut *val0, sub_name, value);
      subst(&mut *val1, sub_name, value);
    }
    language::syntax::Term::Let { name, expr, body } => {
      subst(&mut *expr, sub_name, value);
      if name != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    language::syntax::Term::Lam { name, body } => {
      if name != sub_name {
        subst(&mut *body, sub_name, value);
      }
    }
    language::syntax::Term::App { func, argm } => {
      subst(&mut *func, sub_name, value);
      subst(&mut *argm, sub_name, value);
    }
    language::syntax::Term::Ctr { args, .. } => {
      for arg in args {
        subst(&mut *arg, sub_name, value);
      }
    }
    language::syntax::Term::U6O { .. } => {}
    language::syntax::Term::F6O { .. } => {}
    language::syntax::Term::Op2 { val0, val1, .. } => {
      subst(&mut *val0, sub_name, value);
      subst(&mut *val1, sub_name, value);
    }
  }
}

// Split rules that have nested cases, flattening them.
// I'm not proud of this code. Must improve considerably.
pub fn flatten(rules: &[language::syntax::Rule]) -> Vec<language::syntax::Rule> {
  // Unique name generator
  let mut name_count = 0;
  fn fresh(name_count: &mut u64) -> u64 {
    let name = *name_count;
    *name_count += 1;
    name
  }

  // Checks if this rule has nested patterns, and must be splitted
  #[rustfmt::skip]
  fn must_split(lhs: &language::syntax::Term) -> bool {
/**/if let language::syntax::Term::Ctr { ref args, .. } = *lhs {
/*  */for arg in args {
/*  H */if let language::syntax::Term::Ctr { args: ref arg_args, .. } = **arg {
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
  //fn is_default(lhs: &language::syntax::Term) -> bool {
///**/if let language::syntax::Term::Ctr { ref args, .. } = *lhs {
///*  */for arg in args {
///*  H */if let language::syntax::Term::Ctr { args: ref arg_args, .. } = **arg {
///*   A  */for field in arg_args {
///*    D   */if !is_variable(field) {
///* ─=≡ΣO)   */return false;
///*    U   */}
///*   K  */}
///*  E */}
///* N*/}
///**/} true
  //}

  fn is_matchable(term: &language::syntax::Term) -> bool {
    matches!(term,
        language::syntax::Term::Ctr { .. }
      | language::syntax::Term::U6O { .. }
      | language::syntax::Term::F6O { .. })
  }

  //fn is_variable(term: &language::syntax::Term) -> bool {
    //matches!(term, language::syntax::Term::Var { .. })
  //}


  // Checks true if every time that `a` matches, `b` will match too
  fn matches_together(a: &language::syntax::Rule, b: &language::syntax::Rule) -> (bool, bool) {
    let mut same_shape = true;
    if let (
      language::syntax::Term::Ctr { name: ref _a_name, args: ref a_args },
      language::syntax::Term::Ctr { name: ref _b_name, args: ref b_args },
    ) = (&*a.lhs, &*b.lhs) {
      for (a_arg, b_arg) in a_args.iter().zip(b_args) {
        match **a_arg {
          language::syntax::Term::Ctr { name: ref a_arg_name, args: ref a_arg_args } => match **b_arg {
            language::syntax::Term::Ctr { name: ref b_arg_name, args: ref b_arg_args } => {
              if a_arg_name != b_arg_name || a_arg_args.len() != b_arg_args.len() {
                return (false, false);
              }
            }
            language::syntax::Term::U6O { .. } => {
              return (false, false);
            }
            language::syntax::Term::F6O { .. } => {
              return (false, false);
            }
            language::syntax::Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          language::syntax::Term::U6O { numb: a_arg_numb } => match **b_arg {
            language::syntax::Term::U6O { numb: b_arg_numb } => {
              if a_arg_numb != b_arg_numb {
                return (false, false);
              }
            }
            language::syntax::Term::Ctr { .. } => {
              return (false, false);
            }
            language::syntax::Term::Var { .. } => {
              same_shape = false;
            }
            _ => {}
          },
          language::syntax::Term::F6O { numb: a_arg_numb } => match **b_arg {
            language::syntax::Term::F6O { numb: b_arg_numb } => {
              if a_arg_numb != b_arg_numb {
                return (false, false);
              }
            }
            language::syntax::Term::Ctr { .. } => {
              return (false, false);
            }
            language::syntax::Term::Var { .. } => {
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

  fn split_group(rules: &[language::syntax::Rule], name_count: &mut u64) -> Vec<language::syntax::Rule> {
    // println!("\n[split_group]");
    // for rule in rules {
    //   println!("{}", rule);
    // }
    let mut skip: HashSet<usize> = HashSet::new();
    let mut new_rules: Vec<language::syntax::Rule> = Vec::new();
    for i in 0..rules.len() {
      if !skip.contains(&i) {
        let rule = &rules[i];
        if must_split(&rule.lhs) {
          if let language::syntax::Term::Ctr { ref name, ref args } = *rule.lhs {
            let mut new_group: Vec<language::syntax::Rule> = Vec::new();
            let new_lhs_name: String = name.clone();
            let new_rhs_name: String = format!("{}.{}", name, fresh(name_count));
            let mut new_lhs_args: Vec<Box<language::syntax::Term>> = Vec::new();
            let mut new_rhs_args: Vec<Box<language::syntax::Term>> = Vec::new();
            for arg in args {
              match &**arg {
                language::syntax::Term::Ctr { name: ref arg_name, args: ref arg_args } => {
                  let new_arg_name = arg_name.clone();
                  let mut new_arg_args = Vec::new();
                  for field in arg_args {
                    match &**field {
                      language::syntax::Term::Ctr { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                      }
                      language::syntax::Term::U6O { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                      }
                      language::syntax::Term::F6O { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(language::syntax::Term::Var { name: var_name.clone() }));
                      }
                      language::syntax::Term::Var { .. } => {
                        new_arg_args.push(field.clone());
                        new_rhs_args.push(field.clone());
                      }
                      _ => {
                        panic!("?");
                      }
                    }
                  }
                  new_lhs_args.push(Box::new(language::syntax::Term::Ctr { name: new_arg_name, args: new_arg_args }));
                }
                language::syntax::Term::Var { .. } => {
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
            let new_lhs = Box::new(language::syntax::Term::Ctr { name: new_lhs_name, args: new_lhs_args.clone() });
            let new_rhs = Box::new(language::syntax::Term::Ctr { name: new_rhs_name.clone(), args: new_rhs_args });
            let new_rule = language::syntax::Rule { lhs: new_lhs, rhs: new_rhs };
            new_group.push(new_rule);
            for (j, other) in rules.iter().enumerate().skip(i) {
              let (compatible, same_shape) = matches_together(rule, other);
              if compatible {
                if let (
                  language::syntax::Term::Ctr { name: ref _rule_name, args: ref rule_args },
                  language::syntax::Term::Ctr { name: ref _other_name, args: ref other_args }, 
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
                      language::syntax::Term::Ctr { name: ref rule_arg_name, args: ref rule_arg_args } => {
                        match &**other_arg {
                          language::syntax::Term::Ctr { name: ref _other_arg_name, args: ref other_arg_args } => {
                            for other_field in other_arg_args {
                              other_new_lhs_args.push(other_field.clone());
                            }
                          }
                          language::syntax::Term::Var { name: ref other_arg_name } => {
                            let mut new_ctr_args = vec![];
                            for _ in 0 .. rule_arg_args.len() {
                              let new_arg = language::syntax::Term::Var { name: format!(".{}", fresh(name_count)) };
                              new_ctr_args.push(Box::new(new_arg.clone()));
                              other_new_lhs_args.push(Box::new(new_arg));
                            }
                            let new_ctr = language::syntax::Term::Ctr { name: rule_arg_name.clone(), args: new_ctr_args };
                            subst(&mut other_new_rhs, other_arg_name, &new_ctr);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      language::syntax::Term::Var { .. } => {
                        other_new_lhs_args.push(other_arg.clone());
                      }
                      language::syntax::Term::U6O { numb: rule_arg_numb } => {
                        match &**other_arg {
                          language::syntax::Term::U6O { numb: other_arg_numb } => {
                            if rule_arg_numb == other_arg_numb {
                              other_new_lhs_args.push(Box::new(*other_arg.clone()));
                            } else {
                              panic!("Internal error. Please report."); // not possible since it matches
                            }
                          }
                          language::syntax::Term::Var { name: ref other_arg_name } => {
                            subst(&mut other_new_rhs, other_arg_name, rule_arg);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      language::syntax::Term::F6O { numb: rule_arg_numb } => {
                        match &**other_arg {
                          language::syntax::Term::F6O { numb: other_arg_numb } => {
                            if rule_arg_numb == other_arg_numb {
                              other_new_lhs_args.push(Box::new(*other_arg.clone()));
                            } else {
                              panic!("Internal error. Please report."); // not possible since it matches
                            }
                          }
                          language::syntax::Term::Var { name: ref other_arg_name } => {
                            subst(&mut other_new_rhs, other_arg_name, rule_arg);
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
                  let other_new_lhs = Box::new(language::syntax::Term::Ctr {
                    name: other_new_lhs_name,
                    args: other_new_lhs_args,
                  });
                  let new_rule = language::syntax::Rule { lhs: other_new_lhs, rhs: other_new_rhs };
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
  let mut groups: HashMap<String, Vec<language::syntax::Rule>> = HashMap::new();
  for rule in rules {
    if let language::syntax::Term::Ctr { ref name, .. } = *rule.lhs {
      if let Some(group) = groups.get_mut(name) {
        group.push(rule.clone());
      } else {
        groups.insert(name.clone(), vec![rule.clone()]);
      }
    }
  }

  // For each group, split its internal rules
  let mut new_rules = Vec::new();
  for (_name, rules) in groups {
    for rule in split_group(&rules, &mut name_count) {
      new_rules.push(rule);
    }
  }

  // println!("\nresult:");
  // for rule in &new_rules {
  //   println!("{}", rule);
  // }

  new_rules
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
