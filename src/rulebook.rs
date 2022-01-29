use crate::language as lang;
use std::collections::{BTreeMap, HashMap, HashSet};

// RuleBook
// ========

// A RuleBook is a file ready for compilation. It includes:
// - func_rules: sanitized rules grouped by function
// - id_to_name: maps ctr ids to names
// - name_to_id: maps ctr names to ids
// - ctr_is_cal: true if a ctr is used as a function
// A sanitized rule has all its variables renamed to have unique names.
// Variables that are never used are renamed to "*".
#[derive(Debug)]
pub struct RuleBook {
  pub func_rules: HashMap<String, (usize, Vec<lang::Rule>)>,
  pub id_to_name: HashMap<u64, String>,
  pub name_to_id: HashMap<String, u64>,
  pub ctr_is_cal: HashMap<String, bool>,
}

pub fn gen_rulebook(file: &lang::File) -> RuleBook {

  // Generates a name table for a whole program. That table links constructor
  // names (such as `cons` and `succ`) to small ids (such as `0` and `1`).
  pub type NameToId = HashMap<String, u64>;
  pub type IdToName = HashMap<u64, String>;
  pub fn gen_name_to_id(rules: &[lang::Rule]) -> NameToId {
    fn find_ctrs(term: &lang::Term, table: &mut NameToId, fresh: &mut u64) {
      match term {
        lang::Term::Dup { expr, body, .. } => {
          find_ctrs(expr, table, fresh);
          find_ctrs(body, table, fresh);
        }
        lang::Term::Let { expr, body, .. } => {
          find_ctrs(expr, table, fresh);
          find_ctrs(body, table, fresh);
        }
        lang::Term::Lam { body, .. } => {
          find_ctrs(body, table, fresh);
        }
        lang::Term::App { func, argm, .. } => {
          find_ctrs(func, table, fresh);
          find_ctrs(argm, table, fresh);
        }
        lang::Term::Op2 { val0, val1, .. } => {
          find_ctrs(val0, table, fresh);
          find_ctrs(val1, table, fresh);
        }
        lang::Term::Ctr { name, args } => {
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
          }
          for arg in args {
            find_ctrs(arg, table, fresh);
          }
        }
        _ => (),
      }
    }
    let mut table = HashMap::new();
    let mut fresh = 0;
    for rule in rules {
      find_ctrs(&rule.lhs, &mut table, &mut fresh);
      find_ctrs(&rule.rhs, &mut table, &mut fresh);
    }
    table
  }
  pub fn invert(name_to_id: &NameToId) -> IdToName {
    let mut id_to_name: IdToName = HashMap::new();
    for (name, id) in name_to_id {
      id_to_name.insert(*id, name.clone());
    }
    id_to_name
  }

  // Finds constructors that are used as functions.
  pub type IsFunctionTable = HashMap<String, bool>;
  pub fn gen_ctr_is_cal(rules: &[lang::Rule]) -> IsFunctionTable {
    let mut is_call: IsFunctionTable = HashMap::new();
    for rule in rules {
      let term = &rule.lhs;
      if let lang::Term::Ctr { ref name, .. } = **term {
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
  pub type FuncRules = HashMap<String, (usize, Vec<lang::Rule>)>;
  pub fn gen_func_rules(rules: &[lang::Rule]) -> FuncRules {
    let mut groups: FuncRules = HashMap::new();
    for rule in rules {
      if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
        let group = groups.get_mut(name);
        let rule = sanitize_rule(rule).unwrap();
        match group {
          None => {
            groups.insert(name.clone(), (args.len(), Vec::from([rule])));
          }
          Some((arity, rules)) => {
            rules.push(rule);
          }
        }
      }
    }
    groups
  }

  let flat_rules = flatten(&file.rules);
  let func_rules = gen_func_rules(&flat_rules);
  let name_to_id = gen_name_to_id(&flat_rules);
  let id_to_name = invert(&name_to_id);
  let ctr_is_cal = gen_ctr_is_cal(&flat_rules);
  RuleBook { func_rules, name_to_id, id_to_name, ctr_is_cal }
}

// Sanitize
// ========

pub struct SanitizedRule {
  pub rule: lang::Rule,
  pub uses: HashMap<String, u64>,
}

// This big function sanitizes a rule. That has the following effect:
// - All variables are renamed to have a global unique name.
// - All variables are linearized.
//   - If they're used more than once, dups are inserted.
//   - If they're used once, nothing changes.
//   - If they're never used, their name is changed to "*".
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
    if let lang::Term::Ctr { ref name, ref args } = **lhs {
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
          lang::Term::U32 { .. } => {}
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
      if uses.get(name).copied() <= Some(0) {
        *name = "*".to_string();
      }
    }
    let term = match term {
      lang::Term::Var { name } => {
        if lhs {
          let mut name = tbl.get(name).unwrap_or(name).clone();
          rename_erased(&mut name, ctx.uses);
          Box::new(lang::Term::Var { name })
        } else {
          // create a var with the name generated before
          // concatenated with '.{{times_used}}'
          let gen_name = tbl.get(name);
          if let Some(name) = gen_name {
            let used = { *ctx.uses.entry(name.clone()).and_modify(|x| *x += 1).or_insert(1) };
            let name = format!("{}.{}", name, used - 1);
            Box::new(lang::Term::Var { name })
          } else {
            return Err(format!("Error: unbound variable {}.", name));
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
        if let Some(x) = got_nam0 {
          tbl.insert(nam0.clone(), x);
        }
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
        if let Some(x) = got_name {
          tbl.insert(name.clone(), x);
        }
        duplicator(&new_name, expr, body, ctx.uses)
      }
      lang::Term::Lam { name, body } => {
        let mut new_name = (ctx.fresh)();
        let got_name = tbl.remove(name);
        tbl.insert(name.clone(), new_name.clone());
        let body = sanitize_term(body, lhs, tbl, ctx)?;
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
      lang::Term::U32 { numb } => {
        let term = lang::Term::U32 { numb: *numb };
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
  let mut lhs = sanitize_term(&rule.lhs, true, &mut table.clone(), &mut ctx)?;

  // duplicate right side variables that are used more than once
  for (key, value) in table {
    let expr = Box::new(lang::Term::Var { name: value.clone() });
    rhs = duplicator(&value, expr, rhs, &uses);
  }

  // returns the sanitized rule
  Ok(lang::Rule { lhs, rhs })
}

#[cfg(test)]
mod tests {
  use core::panic;
  use cranelift::codegen::timing::compile;

  use super::{gen_rulebook, sanitize_rule};
  use crate::language::{read_file, read_rule, Rule};

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
      let rule = read_rule(code);
      match rule {
        None => panic!("Rule not parsed"),
        Some(v) => {
          let result = sanitize_rule(&v);
          match result {
            Ok(rule) => assert_eq!(rule.to_string(), expected),
            Err(err) => panic!("Rule not sanitized"),
          }
        }
      }
    }
  }

  #[test]
  fn test_sanitize_fail_code() {
    // code that has to fail
    let FAILS = [
      // more than one nesting in constructors
      "(Foo (Bar (Zaz x))) = (x)",
      // variable not declared in lhs
      "(Succ x) = (j)",
    ];

    for code in FAILS {
      let rule = read_rule(code);
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

    let file = read_file(file);
    let rulebook = gen_rulebook(&file);

    // func_rules testing
    // contains expected key
    assert!(rulebook.func_rules.contains_key("Double"));
    // contains expected number of keys
    assert_eq!(rulebook.func_rules.len(), 1);
    // key contains expected number of rules
    assert_eq!(rulebook.func_rules.get("Double").unwrap().1.len(), 2);
    // key contains expected arity
    assert_eq!(rulebook.func_rules.get("Double").unwrap().0, 1);

    // id_to_name e name_to_id testing
    // check expected length
    assert_eq!(rulebook.id_to_name.len(), 3);
    // check determinism and existence
    assert_eq!(rulebook.id_to_name.get(&0).unwrap(), "Double");
    assert_eq!(rulebook.id_to_name.get(&1).unwrap(), "Zero");
    assert_eq!(rulebook.id_to_name.get(&2).unwrap(), "Succ");
    // check cohesion
    let size = rulebook.id_to_name.len();
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

// Split rules that have nested cases, flattening them.
// I'm not proud of this code. Must improve considerably.
pub fn flatten(rules: &Vec<lang::Rule>) -> Vec<lang::Rule> {

  // Unique name generator
  let mut name_count = 0;
  fn fresh(name_count: &mut u64) -> u64 {
    let name = *name_count;
    *name_count += 1;
    return name;
  }
  
  // Checks if this rule has nested patterns, and must be splitted
  fn must_split(lhs: &lang::Term) -> bool {
/**/if let lang::Term::Ctr { ref args, .. } = *lhs {
/*  */for arg in args {
/*  H */if let lang::Term::Ctr { args: ref arg_args, .. } = **arg {
/*   A  */for field in arg_args {
/*    D   */if is_tested(&field) {
/* ─=≡ΣO)   */return true;
/*    U   */}
/*   K  */}
/*  E */}
/* N*/}
/**/} return false;
  }

  fn is_tested(term: &lang::Term) -> bool {
    match term {
      lang::Term::Ctr { .. } => true,
      lang::Term::U32 { .. } => true,
      _                      => false,
    }
  }

  // Checks true if every time that `a` matches, `b` will match too
  fn matches_together(a: &lang::Rule, b: &lang::Rule) -> bool {
    if let (
      lang::Term::Ctr { name: ref a_name, args: ref a_args },
      lang::Term::Ctr { name: ref b_name, args: ref b_args }
    ) = (&*a.lhs, &*b.lhs) {
      for (a_arg, b_arg) in a_args.iter().zip(b_args) {
        match **a_arg {
          lang::Term::Ctr { name: ref a_arg_name, args: ref a_arg_args } => {
            match **b_arg {
              lang::Term::Ctr { name: ref b_arg_name, args: ref b_arg_args } => {
                if a_arg_name != b_arg_name || a_arg_args.len() != b_arg_args.len() {
                  return false;
                }
              }
              lang::Term::U32 { .. } => {
                return false;
              }
              lang::Term::Var { .. } => {
                println!("Sorry! HVM can't flatten this nested case:");
                println!();
                println!("  {}", a);
                println!();
                println!("Because of the argument '{}', in:", b_arg);
                println!();
                println!("  {}", b);
                println!();
                println!("This is a HVM limitation, and will be fixed in a future.");
                println!();
                std::process::exit(1);
              }
              _ => {}
            }
          }
          lang::Term::U32 { numb: a_arg_numb } => {
            match **b_arg {
              lang::Term::U32 { numb: b_arg_numb } => {
                if a_arg_numb != b_arg_numb {
                  return false;
                }
              }
              lang::Term::Ctr { .. } => {
                return false;
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
    }
    return true;
  }

  fn split_group(rules: &Vec<lang::Rule>, name_count: &mut u64) -> Vec<lang::Rule> {
    let mut skip : HashSet<usize> = HashSet::new();
    let mut new_rules : Vec<lang::Rule> = Vec::new();
    for i in 0 .. rules.len() {
      if !skip.contains(&i) {
        let rule = &rules[i];
        //println!("- {}", rule);
        if must_split(&rule.lhs) {
          if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
            let mut new_group : Vec<lang::Rule> = Vec::new();
            let new_lhs_name : String = name.clone();
            let new_rhs_name : String = format!("{}.{}", name, fresh(name_count));
            let mut new_lhs_args : Vec<Box<lang::Term>> = Vec::new();
            let mut new_rhs_args : Vec<Box<lang::Term>> = Vec::new();
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
                      lang::Term::U32 { .. } => {
                        let var_name = format!(".{}", fresh(name_count));
                        new_arg_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                        new_rhs_args.push(Box::new(lang::Term::Var { name: var_name.clone() }));
                      }
                      lang::Term::Var { .. } => {
                        new_arg_args.push(field.clone());
                        new_rhs_args.push(field.clone());
                      }
                      _ => { panic!("?"); }
                    }
                  }
                  new_lhs_args.push(Box::new(lang::Term::Ctr { name: new_arg_name, args: new_arg_args}));
                }
                lang::Term::Var { .. } => {
                  new_lhs_args.push(Box::new(*arg.clone()));
                  new_rhs_args.push(Box::new(*arg.clone()));
                }
                _ => {}
              }
            }
            let new_lhs = Box::new(lang::Term::Ctr {
              name: new_lhs_name,
              args: new_lhs_args,
            });
            let new_rhs = Box::new(lang::Term::Ctr {
              name: new_rhs_name.clone(),
              args: new_rhs_args,
            });
            new_group.push(lang::Rule {
              lhs: new_lhs,
              rhs: new_rhs,
            });

            let new_rule_name = format!(".{}", fresh(name_count));

            for j in i .. rules.len() {
              let other = &rules[j];
              if matches_together(&rule, &other) {
                skip.insert(j);
                if let lang::Term::Ctr { name: ref other_name, args: ref other_args } = &*other.lhs {
                  let mut other_new_lhs_name = new_rhs_name.clone();
                  let mut other_new_lhs_args = Vec::new();
                  for other_arg in other_args {
                    match &**other_arg {
                      lang::Term::Ctr { name: ref other_arg_name, args: ref other_arg_args } => {
                        for other_field in other_arg_args {
                          other_new_lhs_args.push(other_field.clone());
                        }
                      }
                      lang::Term::U32 { .. } => {
                      }
                      lang::Term::Var { .. } => {
                        other_new_lhs_args.push(other_arg.clone());
                      }
                      _ => {}
                    }
                  }
                  let other_new_lhs = Box::new(lang::Term::Ctr {
                    name: other_new_lhs_name,
                    args: other_new_lhs_args,
                  });
                  let other_new_rhs = other.rhs.clone();
                  //println!("~~ {} = {}", other_new_lhs, other_new_rhs);
                  new_group.push(lang::Rule {
                    lhs: other_new_lhs,
                    rhs: other_new_rhs,
                  });
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
    return new_rules;
  }

  // Groups rules by function name
  let mut groups : HashMap<String, Vec<lang::Rule>> = HashMap::new();
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
  for (name, rules) in groups {
    for rule in split_group(&rules, &mut name_count) {
      new_rules.push(rule);
    }
  }

  return new_rules;

}
