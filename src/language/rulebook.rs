use crate::language;
use crate::prelude::*;
use crate::prelude::*;
use crate::runtime;
use std::collections::{HashMap, HashSet};

// RuleBook
// ========

// A RuleBook is a file ready for compilation. It includes:
// - rule_group: sanitized rules grouped by function
// - id_to_name: maps ctr ids to names
// - name_to_id: maps ctr names to ids
// - ctr_is_fun: true if a ctr is used as a function
// A sanitized rule has all its variables renamed to have unique names.
// Variables that are never used are renamed to "*".
#[derive(Clone, Debug, Default)]
pub struct RuleBook {
  pub rule_group: HashMap<String, RuleGroup>,
  pub name_count: u64,
  pub name_to_id: HashMap<String, u64>,
  pub id_to_smap: HashMap<u64, Vec<bool>>,
  pub id_to_name: HashMap<u64, String>,
  pub ctr_is_fun: HashMap<String, bool>,
}

pub type RuleGroup = (usize, Vec<Rule>);

impl RuleBook {
  // Creates an empty rulebook
  pub fn new() -> Self {
    let mut book = Self::default();

    for precomp in runtime::PRECOMP {
      book.name_count += 1;
      book.name_to_id.insert(precomp.name.to_string(), precomp.id);
      book.id_to_name.insert(precomp.id, precomp.name.to_string());
      book.id_to_smap.insert(precomp.id, precomp.smap.to_vec());
      book.ctr_is_fun.insert(precomp.name.to_string(), precomp.funs.is_some());
    }
    book
  }

  fn register(&mut self, term: &Term, lhs_top: bool) {
    match term {
      Term::Dup { expr, body, .. } => {
        self.register(expr, false);
        self.register(body, false);
      }
      Term::Sup { val0, val1 } => {
        self.register(val0, false);
        self.register(val1, false);
      }
      Term::Let { expr, body, .. } => {
        self.register(expr, false);
        self.register(body, false);
      }
      Term::Lam { body, .. } => {
        self.register(body, false);
      }
      Term::App { func, argm, .. } => {
        self.register(func, false);
        self.register(argm, false);
      }
      Term::Op2 { val0, val1, .. } => {
        self.register(val0, false);
        self.register(val1, false);
      }
      term @ Term::Ctr { name, args } => {
        // Registers id
        let id = match self.name_to_id.get(name) {
          None => {
            let id = self.name_count;
            self.name_to_id.insert(name.clone(), id);
            self.id_to_name.insert(id, name.clone());
            self.name_count += 1;
            id
          }
          Some(id) => *id,
        };
        // Registers smap
        match self.id_to_smap.get(&id) {
          None => {
            self.id_to_smap.insert(id, vec![false; args.len()]);
          }
          Some(smap) => {
            if smap.len() != args.len() {
              panic!("inconsistent arity on: '{}'", term);
            }
          }
        }
        // Force strictness when pattern-matching
        if lhs_top {
          for (i, arg) in args.iter().enumerate() {
            if arg.is_strict() {
              self.id_to_smap.get_mut(&id).unwrap()[i] = true;
            }
          }
        }
        // Recurses
        for arg in args {
          self.register(arg, false);
        }
      }
      _ => (),
    }
  }
  // Adds a group to a rulebook
  pub fn add_group(&mut self, name: &str, group: &RuleGroup) {
    // Inserts the group on the book
    self.rule_group.insert(name.to_string(), group.clone());

    // Builds its metadata (name_to_id, id_to_name, ctr_is_fun)
    for rule in &group.1 {
      self.register(&rule.lhs, true);
      self.register(&rule.rhs, false);
      if let Term::Ctr { ref name, .. } = *rule.lhs {
        self.ctr_is_fun.insert(name.clone(), true);
      }
    }
  }
}

impl From<&language::syntax::File> for RuleBook {
  fn from(file: &language::syntax::File) -> Self {
    // Creates an empty rulebook
    let mut book = Self::new();

    // Flattens, sanitizes and groups this file's rules
    let groups = group_rules(&sanitize_rules(&flatten(&file.rules)));

    // Adds each group
    for (name, group) in groups.iter() {
      if book.name_to_id.get(name).unwrap_or(&u64::MAX) >= &runtime::PRECOMP_COUNT {
        book.add_group(name, group);
      }
    }

    // Includes SMaps
    for (rule_name, rule_smap) in &file.smaps {
      let id = book.name_to_id.get(rule_name).unwrap();
      if book.id_to_smap.get(id).is_none() {
        book.id_to_smap.insert(*id, vec![false; rule_smap.len()]);
      }
      let smap = book.id_to_smap.get_mut(id).unwrap();
      for i in 0..smap.len() {
        if rule_smap[i] {
          smap[i] = true;
        }
      }
    }

    book
  }
}

// Groups rules by name. For example:
//   (add (succ a) (succ b)) = (succ (succ (add a b)))
//   (add (succ a) (zero)  ) = (succ a)
//   (add (zero)   (succ b)) = (succ b)
//   (add (zero)   (zero)  ) = (zero)
// This is a group of 4 rules starting with the "add" name.
pub fn group_rules(rules: &[Rule]) -> HashMap<String, RuleGroup> {
  let mut groups: HashMap<String, RuleGroup> = HashMap::new();
  for rule in rules {
    if let Term::Ctr { ref name, ref args } = *rule.lhs {
      let group = groups.get_mut(name);
      match group {
        None => {
          groups.insert(name.clone(), (args.len(), vec![rule.clone()]));
        }
        Some((_, rules)) => {
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
  pub rule: Rule,
  pub uses: HashMap<String, u64>,
}

// Sanitizes all rules in a vector
pub fn sanitize_rules(rules: &[Rule]) -> Vec<Rule> {
  rules
    .iter()
    .map(|rule| {
      match rule.sanitize_rule() {
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

  use super::RuleBook;
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
          let result = v.sanitize_rule();
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
          let result = v.sanitize_rule();
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
    let rulebook: RuleBook = (&file).into();

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

// Split rules that have nested cases, flattening them.
// I'm not proud of this code. Must improve considerably.
pub fn flatten(rules: &[Rule]) -> Vec<Rule> {
  // Unique name generator
  let mut name_count = 0;
  fn fresh(name_count: &mut u64) -> u64 {
    let name = *name_count;
    *name_count += 1;
    name
  }

  // Returns true if a rule is a global default case (all args are vars)
  //fn is_default(lhs: &Term) -> bool {
  ///**/if let Term::Ctr { ref args, .. } = *lhs {
  ///*  */for arg in args {
  ///*  H */if let Term::Ctr { args: ref arg_args, .. } = **arg {
  ///*   A  */for field in arg_args {
  ///*    D   */if !is_variable(field) {
  ///* ─=≡ΣO)   */return false;
  ///*    U   */}
  ///*   K  */}
  ///*  E */}
  ///* N*/}
  ///**/} true
  //}

  //fn is_variable(term: &Term) -> bool {
  //matches!(term, Term::Var { .. })
  //}

  fn split_group(rules: &[Rule], name_count: &mut u64) -> Vec<Rule> {
    // println!("\n[split_group]");
    // for rule in rules {
    //   println!("{}", rule);
    // }
    let mut skip: HashSet<usize> = HashSet::new();
    let mut new_rules: Vec<Rule> = vec![];
    for i in 0..rules.len() {
      if !skip.contains(&i) {
        let rule = &rules[i];
        if rule.lhs.must_split() {
          if let Term::Ctr { ref name, ref args } = *rule.lhs {
            let mut new_group: Vec<Rule> = vec![];
            let new_lhs_name: String = name.clone();
            let new_rhs_name: String = format!("{}.{}", name, fresh(name_count));
            let mut new_lhs_args: Vec<Box<Term>> = vec![];
            let mut new_rhs_args: Vec<Box<Term>> = vec![];
            for arg in args {
              match &**arg {
                Term::Ctr { name: ref arg_name, args: ref arg_args } => {
                  let new_arg_name = arg_name.clone();
                  let mut new_arg_args = vec![];
                  for field in arg_args {
                    match &**field {
                      Term::Ctr { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(Term::Var { name: var_name.clone() }));
                      }
                      Term::U6O { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(Term::Var { name: var_name.clone() }));
                      }
                      Term::F6O { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(Term::Var { name: var_name.clone() }));
                      }
                      Term::Var { .. } => {
                        new_arg_args.push(field.clone());
                        new_rhs_args.push(field.clone());
                      }
                      _ => {
                        panic!("?");
                      }
                    }
                  }
                  new_lhs_args.push(Box::new(Term::Ctr { name: new_arg_name, args: new_arg_args }));
                }
                Term::Var { .. } => {
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
            let new_lhs = Box::new(Term::Ctr { name: new_lhs_name, args: new_lhs_args.clone() });
            let new_rhs = Box::new(Term::Ctr { name: new_rhs_name.clone(), args: new_rhs_args });
            let new_rule = Rule { lhs: new_lhs, rhs: new_rhs };
            new_group.push(new_rule);
            for (j, other) in rules.iter().enumerate().skip(i) {
              let (compatible, same_shape) = rule.matches_together(&other);
              if compatible {
                if let (
                  Term::Ctr { name: ref _rule_name, args: ref rule_args },
                  Term::Ctr { name: ref _other_name, args: ref other_args },
                ) = (&*rule.lhs, &*other.lhs)
                {
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
                  let mut other_new_lhs_args = vec![];
                  let mut other_new_rhs = other.rhs.clone();
                  for (rule_arg, other_arg) in rule_args.iter().zip(other_args) {
                    match &**rule_arg {
                      Term::Ctr { name: ref rule_arg_name, args: ref rule_arg_args } => {
                        match &**other_arg {
                          Term::Ctr { name: ref _other_arg_name, args: ref other_arg_args } => {
                            for other_field in other_arg_args {
                              other_new_lhs_args.push(other_field.clone());
                            }
                          }
                          Term::Var { name: ref other_arg_name } => {
                            let mut new_ctr_args = vec![];
                            for _ in 0..rule_arg_args.len() {
                              let new_arg = Term::Var { name: format!(".{}", fresh(name_count)) };
                              new_ctr_args.push(Box::new(new_arg.clone()));
                              other_new_lhs_args.push(Box::new(new_arg));
                            }
                            let new_ctr =
                              Term::Ctr { name: rule_arg_name.clone(), args: new_ctr_args };
                            other_new_rhs.subst(other_arg_name, &new_ctr);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      Term::Var { .. } => {
                        other_new_lhs_args.push(other_arg.clone());
                      }
                      Term::U6O { numb: rule_arg_numb } => {
                        match &**other_arg {
                          Term::U6O { numb: other_arg_numb } => {
                            if rule_arg_numb == other_arg_numb {
                              other_new_lhs_args.push(Box::new(*other_arg.clone()));
                            } else {
                              panic!("Internal error. Please report."); // not possible since it matches
                            }
                          }
                          Term::Var { name: ref other_arg_name } => {
                            other_new_rhs.subst(other_arg_name, &rule_arg);
                          }
                          _ => {
                            panic!("Internal error. Please report."); // not possible since it matches
                          }
                        }
                      }
                      Term::F6O { numb: rule_arg_numb } => {
                        match &**other_arg {
                          Term::F6O { numb: other_arg_numb } => {
                            if rule_arg_numb == other_arg_numb {
                              other_new_lhs_args.push(Box::new(*other_arg.clone()));
                            } else {
                              panic!("Internal error. Please report."); // not possible since it matches
                            }
                          }
                          Term::Var { name: ref other_arg_name } => {
                            other_new_rhs.subst(other_arg_name, &rule_arg);
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
                  let other_new_lhs =
                    Box::new(Term::Ctr { name: other_new_lhs_name, args: other_new_lhs_args });
                  let new_rule = Rule { lhs: other_new_lhs, rhs: other_new_rhs };
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
  let mut groups: HashMap<String, Vec<Rule>> = HashMap::new();
  for rule in rules {
    if let Term::Ctr { ref name, .. } = *rule.lhs {
      if let Some(group) = groups.get_mut(name) {
        group.push(rule.clone());
      } else {
        groups.insert(name.clone(), vec![rule.clone()]);
      }
    }
  }

  // For each group, split its internal rules
  let mut new_rules = vec![];
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

// notes
// -----

// hoas_opt: this is an internal optimization that allows us to simplify kind2's hoas generator.
// it will cause the default patterns of functions with a name starting with "f$" to only match
// productive hoas constructors (ct0, ct1, ..., ctg, num), as well as native numbers and
// constructors with 0-arity, which are used by kind2's hoas functions, unless it is the last
// (default) clause, which kind2 uses to quote a call back to low-order. this is an internal
// feature that won't affect programs other than kind2. we can remove this in a future, but that
// would require kind2 to replicate hvm's flattener algorithm, so we just use it instead.
