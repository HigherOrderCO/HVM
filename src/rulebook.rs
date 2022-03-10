use crate::language as lang;
use std::collections::{BTreeMap, HashMap};

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
  pub ctr_is_cal: HashMap<String, bool>,
}

pub type RuleGroup = (usize, Vec<lang::Rule>);

// Creates an empty rulebook
pub fn new_rulebook() -> RuleBook {
  RuleBook {
    rule_group: HashMap::new(),
    name_count: 0,
    name_to_id: HashMap::new(),
    id_to_name: HashMap::new(),
    ctr_is_cal: HashMap::new(),
  }
}

// Adds a group to a rulebook
pub fn add_group(book: &mut RuleBook, name: &String, group: &RuleGroup) {
  fn register_names(book: &mut RuleBook, term: &lang::Term) {
    match term {
      lang::Term::Dup { expr, body, .. } => {
        register_names(book, expr);
        register_names(book, body);
      }
      lang::Term::Let { expr, body, .. } => {
        register_names(book, expr);
        register_names(book, body);
      }
      lang::Term::Lam { body, .. } => {
        register_names(book, body);
      }
      lang::Term::App { func, argm, .. } => {
        register_names(book, func);
        register_names(book, argm);
      }
      lang::Term::Op2 { val0, val1, .. } => {
        register_names(book, val0);
        register_names(book, val1);
      }
      lang::Term::Ctr { name, args } => {
        let id = book.name_to_id.get(name);
        if id.is_none() {
          book.name_to_id.insert(name.clone(), book.name_count);
          book.id_to_name.insert(book.name_count, name.clone());
          book.name_count += 1;
        }
        for arg in args {
          register_names(book, arg);
        }
      }
      _ => (),
    }
  }

  // Inserts the group on the book
  book.rule_group.insert(name.clone(), group.clone());

  // Builds its metadata (name_to_id, id_to_name, ctr_is_cal)
  for rule in &group.1 {
    register_names(book, &rule.lhs);
    register_names(book, &rule.rhs);
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
//   (Add (Succ a) (Succ b)) = (Succ (Succ (Add a b)))
//   (Add (Succ a) (Zero)  ) = (Succ a)
//   (Add (Zero)   (Succ b)) = (Succ b)
//   (Add (Zero)   (Zero)  ) = (Zero)
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
  name.len() > 0 && name.starts_with(&"$")
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
        } else {
          if is_global_name(&name) {
            if let Some(_) = tbl.get(name) {
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
            //println!("Allowed unbound variable: {}", name);
            //Box::new(lang::Term::Var { name: name.clone() })
            } else {
              panic!("Unbound variable: {}", name);
            }
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
        let mut new_name = if is_global_name(&name) { name.clone() } else { (ctx.fresh)() };
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
  rules.iter().map(|rule| sanitize_rule(rule).unwrap()).collect()
}

// Split rules that have nested cases, flattening them.
// I'm not proud of this code. Must improve considerably.
pub fn flatten(rules: &[lang::Rule]) -> Vec<lang::Rule> {
  // Checks if this rule has nested patterns, and must be splitted
  #[rustfmt::skip]
  fn nested_pattern(rule: &lang::Rule) -> bool {
/**/if let lang::Term::Ctr { ref args, .. } = *rule.lhs {
/*  */for arg in args {
/*  H */if let lang::Term::Ctr { args: ref arg_args, .. } = **arg {
/*   A  */for field in arg_args {
/*    D   */if is_tested(field) {
/* ─=≡ΣO)   */return true;
/*    U   */}
/*   K  */}
/*  E */}
/* N*/}
/**/} false
  }

  fn is_tested(term: &lang::Term) -> bool {
    matches!(term, lang::Term::Ctr { .. } | lang::Term::U32 { .. })
  }
  fn show_var(i: usize) -> String {
    put_suffix("x", i)
  }
  fn put_suffix(prefix: &str, i: usize) -> String {
    prefix.to_owned() + "." + &i.to_string()
  }

  // this function takes a rule that may have nested patterns and returns
  // a rule that only matches one layer of constructors and then calls an
  // auxiliary def.
  // look at "first_layer" tests for examples
  // i bet this could be shorter and clearer, but I'm still getting used to rust
  fn first_layer(rule: &lang::Rule, n: usize) -> lang::Rule {
    if let lang::Term::Ctr { ref name, ref args } = *rule.lhs {
      let mut i = 0;
      let mut lhs_args: Vec<Box<lang::Term>> = Vec::new();
      let mut rhs_args: Vec<Box<lang::Term>> = Vec::new();
      let mut ok = true;
      for arg in args {
        match **arg {
          lang::Term::Ctr { name: ref arg_name, args: ref arg_args } => {
            let mut new_arg_args: Vec<Box<lang::Term>> = Vec::new();
            for _ in arg_args {
              let var_name = show_var(i);
              i += 1;
              let var = Box::new(lang::Term::Var { name: var_name });
              new_arg_args.push(var.clone());
              rhs_args.push(var);
            }
            let new_arg = Box::new(lang::Term::Ctr { name: arg_name.clone(), args: new_arg_args });
            lhs_args.push(new_arg);
          }
          lang::Term::Var { .. } => {
            let var_name = show_var(i);
            i += 1;
            let new_var = Box::new(lang::Term::Var { name: var_name });
            rhs_args.push(new_var.clone());
            lhs_args.push(new_var);
          }
          lang::Term::U32 { numb } => {
            lhs_args.push(Box::new(lang::Term::U32 { numb }));
          }
          _ => {
            ok = false;
            break;
          }
        }
      }
      if ok {
        let lhs = Box::new(lang::Term::Ctr { name: name.to_string(), args: lhs_args });
        let rhs = Box::new(lang::Term::Ctr { name: put_suffix(name, n), args: rhs_args });
        lang::Rule { lhs, rhs }
      } else {
        panic!("Tried to extract first layer of invalid pattern: {}", rule)
      }
    } else {
      panic!("Tried to extract first layer of invalid pattern: {}", rule)
    }
  }

  // takes a pattern, a subpattern and defines the auxiliary function that's on the right hand side
  // of the output of first_layer(rule, n)
  //
  // preconditions:
  //  P0: subpattern(pattern, subpattern) == true
  //  P1: both lhs' should be Ctr
  fn denest_with_pattern(pattern: &lang::Rule, subpattern: &lang::Rule, n: usize) -> lang::Rule {
    if let (
      lang::Term::Ctr { ref name, ref args },
      lang::Term::Ctr { name: ref _sub_name, args: ref sub_args },
    ) = (&*pattern.lhs, &*subpattern.lhs)
    {
      // P0.0: name == _sub_name
      // P0.1: args.len() == sub_args.len()
      // P0.2: for (arg, sub_args) in args.iter().zip(sub_args) {
      //         subpattern_aux(arg, sub_arg) == true
      //       }
      let mut new_args: Vec<Box<lang::Term>> = Vec::new();
      for (arg, sub_arg) in args.iter().zip(sub_args) {
        match (&**arg, &**sub_arg) {
          (lang::Term::Ctr { .. }, lang::Term::Ctr { args, .. }) => {
            new_args.append(&mut args.clone());
          }
          (lang::Term::U32 { .. }, _) => (),
          _ => {
            new_args.push(sub_arg.clone());
          }
        }
      }
      let lhs = Box::new(lang::Term::Ctr { name: put_suffix(name, n), args: new_args });
      let rhs = subpattern.rhs.clone();
      lang::Rule { lhs, rhs }
    } else {
      // absurd, contradicts P1
      panic!("Tried to denest invalid patterns:\n{}\n{}", pattern, subpattern);
    }
  }

  // checks that if a matches, then b matches, for a valid pair of rules
  // preconditions:
  // - valid_pattern(a)
  // - valid_pattern(b)
  fn subpattern(a: &lang::Rule, b: &lang::Rule) -> bool {
    subpattern_aux(&*a.lhs, &*b.lhs)
  }
  // i'm actually proud of this code
  // preconditions:
  // - valid_pattern(a)
  // - valid_pattern(b)
  fn subpattern_aux(a: &lang::Term, b: &lang::Term) -> bool {
    match (a, b) {
      (lang::Term::Var { .. }, lang::Term::Var { .. }) => true,
      (lang::Term::Ctr { .. }, lang::Term::Var { .. }) => true,
      (lang::Term::U32 { .. }, lang::Term::Var { .. }) => true,
      (
        lang::Term::Ctr { name: a_name, args: a_args },
        lang::Term::Ctr { name: b_name, args: b_args },
      ) => {
        let mut compatible = true;
        for (a_arg, b_arg) in a_args.iter().zip(b_args) {
          compatible = compatible && subpattern_aux(&a_arg, &b_arg);
        }
        (a_name == b_name) && (a_args.len() == b_args.len()) && compatible
      }
      (lang::Term::U32 { numb: a_numb }, lang::Term::U32 { numb: b_numb }) => a_numb == b_numb,
      _ => false,
    }
  }

  // specializes rule_left to a subpattern of rule_right, if that's possible
  // all variables will receive fresh names to avoid name collisions
  fn specialize_left(left: &lang::Rule, right: &lang::Rule) -> Option<lang::Rule> {
    let (lhs, rhs, _i) = specialize_left_aux(*left.lhs.clone(), *left.rhs.clone(), &right.lhs, 0)?;
    let (lhs, rhs) = (Box::new(lhs), Box::new(rhs));
    Some(lang::Rule { lhs, rhs })
  }
  // preconditions:
  // P0: valid_pattern(rule_left_lhs)
  // P1: valid_pattern(rule_right_lhs)
  fn specialize_left_aux(
    rule_left_lhs: lang::Term,
    rule_left_rhs: lang::Term,
    rule_right_lhs: &lang::Term,
    i: usize,
  ) -> Option<(lang::Term, lang::Term, usize)> {
    match (rule_left_lhs, rule_right_lhs) {
      // nothing happens
      (lang::Term::Var { name }, lang::Term::Var { .. }) => {
        let fresh_var = lang::Term::Var { name: show_var(i) };
        let rule_left_rhs = replace(&name, &fresh_var, rule_left_rhs);
        Some((fresh_var, rule_left_rhs, 1 + i))
      }
      (lang::Term::Ctr { name, args }, lang::Term::Var { .. }) => {
        let mut j = i;
        let mut new_args = Vec::new();
        let mut subs: Vec<(String, Box<lang::Term>)> = Vec::new();
        for arg in args {
          let (new_arg, temp_subs, i) = freshen_vars(*arg, subs, j);
          subs = temp_subs;
          j = i;
          new_args.push(Box::new(new_arg));
        }
        let rule_left_rhs = multi_replace(&subs, rule_left_rhs);
        Some((lang::Term::Ctr { name, args: new_args }, rule_left_rhs, j))
      }
      (lang::Term::U32 { numb }, lang::Term::Var { .. }) => {
        Some((lang::Term::U32 { numb }, rule_left_rhs, i))
      }

      // var is replaced
      (lang::Term::Var { name: ref var_name }, lang::Term::Ctr { .. }) => {
        let (rule_right_lhs, _subs, i) = freshen_vars(rule_right_lhs.clone(), Vec::new(), i);
        Some((rule_right_lhs.clone(), replace(var_name, &rule_right_lhs, rule_left_rhs), i))
      }
      (lang::Term::Var { name: ref var_name }, lang::Term::U32 { .. }) => {
        // freshen is unecessary because rule_left_rhs doesn't have subterms
        Some((rule_right_lhs.clone(), replace(var_name, rule_right_lhs, rule_left_rhs), i))
      }

      // match same Ctr and recurse
      (
        lang::Term::Ctr { name: left_name, args: left_args },
        lang::Term::Ctr { name: right_name, args: right_args },
      ) => {
        if &left_name == right_name && left_args.len() == right_args.len() {
          let mut new_left_args: Vec<Box<lang::Term>> = Vec::new();
          let mut rule_left_rhs = rule_left_rhs;
          let mut j = i;
          for (left_arg, right_arg) in left_args.into_iter().zip(right_args) {
            let (new_arg, temp_rule_left_rhs, temp_j) =
              specialize_left_aux(*left_arg, rule_left_rhs, right_arg, j)?;
            rule_left_rhs = temp_rule_left_rhs;
            j = temp_j;
            new_left_args.push(Box::new(new_arg));
          }
          Some((lang::Term::Ctr { name: left_name, args: new_left_args }, rule_left_rhs, j))
        } else {
          None
        }
      }

      // match same U32
      (lang::Term::U32 { numb: left_numb }, lang::Term::U32 { numb: right_numb }) => {
        if left_numb == *right_numb {
          Some((lang::Term::U32 { numb: left_numb }, rule_left_rhs, i))
        } else {
          None
        }
      }

      // error
      _ => None,
    }
  }

  // renames the nth variable in left first depth first order to "x.n"
  // along with a vector of all substitutions made
  // preconditions:
  // - valid_pattern(pattern)
  // example:
  // freshen_vars((Pair (Pair a b) 2 (Succ n)), [], 0)
  //   == (
  //     (Pair (Pair x.0 x.1) 2 (Succ x.2)),
  //     vec![("a", x.0), ("b", x.1), ("n", x.2)],
  //     3
  //   )
  fn freshen_vars(
    pattern: lang::Term,
    mut subs: Vec<(String, Box<lang::Term>)>,
    i: usize,
  ) -> (lang::Term, Vec<(String, Box<lang::Term>)>, usize) {
    match pattern {
      lang::Term::Var { name } => {
        subs.push((name.clone(), Box::new(lang::Term::Var { name: show_var(i) })));
        (lang::Term::Var { name: show_var(i) }, subs, 1 + i)
      }
      lang::Term::Ctr { name, args } => {
        let mut j = i;
        let mut new_args = Vec::new();
        for arg in args {
          let (arg, temp_subs, temp_j) = freshen_vars(*arg, subs, j);
          subs = temp_subs;
          j = temp_j;
          new_args.push(Box::new(arg));
        }
        (lang::Term::Ctr { name, args: new_args }, subs, j)
      }
      lang::Term::U32 { numb } => (lang::Term::U32 { numb }, subs, i),
      _ => panic!("Tried to call freshen_vars on invalid_pattern."),
    }
  }

  // replaces every var named `from` with term `to` inside `here`
  fn replace(from: &String, to: &lang::Term, here: lang::Term) -> lang::Term {
    multi_replace(&vec![(from.clone(), Box::new(to.clone()))], here)
  }
  fn multi_replace(from_to: &Vec<(String, Box<lang::Term>)>, here: lang::Term) -> lang::Term {
    match here {
      lang::Term::Var { ref name } => {
        for (from, to) in from_to {
          if name == from {
            return *to.clone();
          }
        }
        return here;
      }
      // TODO should shadowing stop replaces?
      // I think it should but currently it doesn't
      lang::Term::Dup { nam0, nam1, expr, body } => lang::Term::Dup {
        nam0: nam0,
        nam1: nam1,
        expr: Box::new(multi_replace(from_to, *expr)),
        body: Box::new(multi_replace(from_to, *body)),
      },
      lang::Term::Let { name, expr, body } => lang::Term::Let {
        name: name,
        expr: Box::new(multi_replace(from_to, *expr)),
        body: Box::new(multi_replace(from_to, *body)),
      },
      lang::Term::Lam { name, body } => {
        lang::Term::Lam { name: name, body: Box::new(multi_replace(from_to, *body)) }
      }

      lang::Term::App { func, argm } => lang::Term::App {
        func: Box::new(multi_replace(from_to, *func)),
        argm: Box::new(multi_replace(from_to, *argm)),
      },
      lang::Term::Ctr { name, args } => {
        let mut new_args = Vec::new();
        for arg in args {
          let new_arg = Box::new(multi_replace(from_to, *arg));
          new_args.push(new_arg);
        }
        lang::Term::Ctr { name: name, args: new_args }
      }
      lang::Term::U32 { numb } => lang::Term::U32 { numb: numb },
      lang::Term::Op2 { oper, val0, val1 } => lang::Term::Op2 {
        oper,
        val0: Box::new(multi_replace(from_to, *val0)),
        val1: Box::new(multi_replace(from_to, *val1)),
      },
    }
  }
  pub fn flatten_aux(rules: &[lang::Rule]) -> Vec<lang::Rule> {
    // holds denested rules to be returned in reverse order
    let mut ret: Vec<lang::Rule> = Vec::new();
    // holds rules that could need to be denested
    let mut remaining: Vec<lang::Rule> = Vec::from(rules);
    remaining.reverse();
    let mut i = 0;
    while let Some(reference_pattern) = remaining.pop() {
      if nested_pattern(&reference_pattern) {
        // denested rules for recursive call
        let mut recurse: Vec<lang::Rule> = Vec::new();
        // rules that will be returned `remaining` after iteration ends
        let mut return_to_remaining: Vec<lang::Rule> = Vec::new();
        let first_layer_reference_pattern = first_layer(&reference_pattern, i);
        ret.push(first_layer_reference_pattern.clone());
        recurse.push(denest_with_pattern(&first_layer_reference_pattern, &reference_pattern, i));
        for pattern in remaining.iter().rev() {
          let specialized_pattern = specialize_left(&pattern, &first_layer_reference_pattern);
          if let Some(specialized_pattern) = specialized_pattern {
            recurse.push(denest_with_pattern(
              &first_layer_reference_pattern,
              &specialized_pattern,
              i,
            ));
            if !subpattern(&pattern, &first_layer_reference_pattern) {
              return_to_remaining.push(pattern.clone());
            }
          } else {
            return_to_remaining.push(pattern.clone());
          }
        }
        ret.append(&mut flatten_aux(&recurse));
        return_to_remaining.reverse();
        remaining = return_to_remaining;
        i += 1;
      } else {
        ret.push(reference_pattern);
      }
    }
    ret
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
  for (_name, rules) in groups {
    new_rules.append(&mut flatten_aux(&rules));
  }

  new_rules
}

#[cfg(test)]
mod tests {
  use core::panic;

  use super::{gen_rulebook, sanitize_rule};
  use crate::language as lang;
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
  // NEW FLATTEN
  // TODO
  //fn valid_pattern(pattern: &lang::Term) -> bool { false }

  //  TODO fix tests
  //  // examples for flattening algorithm
  //  const EQ0: &str = "(Half (Succ (Succ x))) = (Succ (Half x))";
  //  const EQ0_FIRST_LAYER: &str = "(Half (Succ x.0)) = (Half.0 x.0)";
  //  const EQ0_AUX_DEF: &str = "(Half.0 (Succ x)) = (Succ (Half x))";
  //
  //  const EQ1: &str = "(Foo (A (B x0)) 2 (B (C x2 x3) x4)) = (Bar 2 (Baz (C x4 x3) x2))";
  //  const EQ1_FIRST_LAYER: &str = "(Foo (A x.0) 2 (B x.1 x.2)) = (Foo.0 x.0 x.1 x.2)";
  //  const EQ1_AUX_DEF: &str = "(Foo.0 (B x0) (C x2 x3) x4) = (Bar 2 (Baz (C x4 x3) x2))";
  //
  //  const EQ2: &str = "(Foo abacate (A (C banana cereja)) (B d e)) = (Bar Zero (Baz cereja d))";
  //  const EQ2_FIRST_LAYER: &str = "(Foo x.0 (A x.1) (B x.2 x.3)) = (Foo.0 x.0 x.1 x.2 x.3)";
  //  const EQ2_AUX_DEF: &str = "(Foo.0 abacate (C banana cereja) d e) = (Bar Zero (Baz cereja d))";
  //
  //  //  const EQ1_EQ2_AUX_DEF: &str = "(Foo.0 x0 (A (C x1 x2)) x.3 x.4) = (Bar Zero (Baz x.2 x.3))";
  //  //  const EQ2_EQ1_AUX_DEF: &str = "(Foo (A (B x.0)) x.1 (C x.2 x.3) x.4) = (Bar (A x.1) (Baz (C x.4 x.3) x.2)";
  //
  //  const TERM_VAR: &str = "x";
  //  const TERM_DUP: &str = "dup a b = x; x";
  //  const TERM_LET: &str = "let a = x; x";
  //  const TERM_LAM: &str = "λa x";
  //  const TERM_APP: &str = "(x x)";
  //  const TERM_CTR: &str = "(Pair x y x)";
  //  const TERM_U32: &str = "2";
  //  const TERM_OP2: &str = "(+ x x)";
  //
  //  const TERM_TO: &str = "(Succ Zero)";
  //
  //  const REPLACED_TERM_VAR: &str = "(Succ Zero)";
  //  const REPLACED_TERM_DUP: &str = "dup a b = (Succ Zero); (Succ Zero)";
  //  const REPLACED_TERM_LET: &str = "let a = (Succ Zero); (Succ Zero)";
  //  const REPLACED_TERM_LAM: &str = "λa (Succ Zero)";
  //  const REPLACED_TERM_APP: &str = "((Succ Zero) (Succ Zero))";
  //  const REPLACED_TERM_CTR: &str = "(Pair (Succ Zero) y (Succ Zero))";
  //  const REPLACED_TERM_U32: &str = "2";
  //  const REPLACED_TERM_OP2: &str = "(+ (Succ Zero) (Succ Zero))";
  //
  //  const FILE_0: &str = "
  //    (Half (Succ (Succ x))) = (Succ x)
  //    (Half n) = n
  //  ";
  //  const FILE_1: &str = "
  //    (Third (Succ (Succ (Succ x)))) = (Succ x)
  //    (Third n) = n
  //  ";
  //  const FILE_2: &str = "
  //    (IntHalf (Succ (Succ a)) b               acc_left acc_right) = (IntHalf a b (Succ acc_left) acc_right)
  //    (IntHalf a               (Succ (Succ b)) acc_left acc_right) = (IntHalf a b acc_left (Succ acc_right))
  //    (IntHalf Zero            Zero            acc_left acc_right) = (Int acc_left acc_right)
  //  ";
  const FILE_3: &str = "
    (RedBlack.balance
      Color.black
      (RedBlack.tie
        Color.red
        (RedBlack.tie Color.red child0 key0 value0 child1)
        key1
        value1
        child2
      )
      key2
      value2
      child3
    ) =
      (RedBlack.tie
        Color.red
        (RedBlack.tie
          Color.black
          child0
          key0
          value0
          child1
        )
        key1
        value1
        (RedBlack.tie
          Color.black
          child2
          key2
          value2
          child3
        )
      )

    (RedBlack.balance
      Color.black
      (RedBlack.tie
        Color.red
        child0
        key0
        value0
        (RedBlack.tie Color.red child1 key1 value1 child2)
      )
      key2
      value2
      child3
    ) =
      (RedBlack.tie
        Color.red
        (RedBlack.tie
          Color.black
          child0
          key0
          value0
          child1
        )
        key1
        value1
        (RedBlack.tie
          Color.black
          child2
          key2
          value2
          child3
        )
      )

    (RedBlack.balance
      Color.black
      child0
      key0
      value0
      (RedBlack.tie
        Color.red
        (RedBlack.tie Color.red child1 key1 value1 child2)
        key2
        value2
        child3
      )
    ) =
      (RedBlack.tie
        Color.red
        (RedBlack.tie
          Color.black
          child0
          key0
          value0
          child1
        )
        key1
        value1
        (RedBlack.tie
          Color.black
          child2
          key2
          value2
          child3
        )
      )

    (RedBlack.balance
      Color.black
      child0
      key0
      value0
      (RedBlack.tie
        Color.red
        child1
        key1
        value1
        (RedBlack.tie Color.red child2 key2 value2 child3)
      )
    ) =
      (RedBlack.tie
        Color.red
        (RedBlack.tie
          Color.black
          child0
          key0
          value0
          child1
        )
        key1
        value1
        (RedBlack.tie
          Color.black
          child2
          key2
          value2
          child3
        )
      )

    (RedBlack.balance color child0 key0 value0 child1) = (RedBlack.tie Color.red child0 key0 value0 child1)
  ";
  //  const FILE_4: &str = "
  //    (Pattern (Succ 2) (Succ (Succ (Succ a)))) = a
  //    (Pattern (Succ 1) (Succ (Succ (Succ a)))) = a
  //    (Pattern (Succ 0) (Succ (Succ (Succ b)))) = b
  //    (Pattern k n) = n
  //  ";
  //  const FILE_5: &str = "
  //    (Pattern (Succ (Pair a        (Succ a)) c)) = 0
  //    (Pattern (Succ c (Pair (Succ b) b       ))) = 1
  //    (Pattern n) = n
  //  ";
  //  const FILE_6: &str = "
  //    (Pattern x (Succ (Succ k))) = (Ctor k)
  //    (Pattern (Succ x.1) y) = 0
  //  ";
  const FILE_7: &str = "
  // (Compare U32 U32) : Comparison
  (Compare a b) = (CompareGo (< a b) (> a b))
    (CompareGo 1 0) = LT
    (CompareGo 0 1) = GT
    (CompareGo a b) = EQ

  // (Insert a (Map a)) : (Map a)
  (Insert x s) = (MakeBlack (InsertGo x s))
    (MakeBlack (T c a x b))       = (T B a x b)
    (InsertGo x E)                = (T R E x E)
    (InsertGo x (T col a y b))    = (InsertGoTest (Compare x y) x col a y b)
      (InsertGoTest LT x col a y b) = (Balance (T col (InsertGo x a) y b))
      (InsertGoTest EQ x col a y b) = (T col a y b)
      (InsertGoTest GT x col a y b) = (Balance (T col a y (InsertGo x b)))

  // (Balance (Map a)) : (Map a)
  (Balance (T B (T R (T R a x b) y c) z d)) = (T R (T B a x b) y (T B c z d))
  (Balance (T B (T R a x (T R b y c)) z d)) = (T R (T B a x b) y (T B c z d))
  (Balance (T B a x (T R (T R b y c) z d))) = (T R (T B a x b) y (T B c z d))
  (Balance (T B a x (T R b y (T R c z d)))) = (T R (T B a x b) y (T B c z d))
  (Balance tree)                            = tree

  (Main n) =
    let map = E
    let map = (Insert 90 map)
    let map = (Insert 10 map)
    map
  ";
  //  #[test]
  //  fn replace_0() {
  //    let term_from = "x";
  //    let term_to = lang::read_term(TERM_TO).unwrap();
  //
  //    let term_var = lang::read_term(TERM_VAR).unwrap();
  //    let term_dup = lang::read_term(TERM_DUP).unwrap();
  //    let term_let = lang::read_term(TERM_LET).unwrap();
  //    let term_lam = lang::read_term(TERM_LAM).unwrap();
  //    let term_app = lang::read_term(TERM_APP).unwrap();
  //    let term_ctr = lang::read_term(TERM_CTR).unwrap();
  //    let term_u32 = lang::read_term(TERM_U32).unwrap();
  //    let term_op2 = lang::read_term(TERM_OP2).unwrap();
  //
  //    let replaced_term_var = lang::read_term(REPLACED_TERM_VAR).unwrap();
  //    let replaced_term_dup = lang::read_term(REPLACED_TERM_DUP).unwrap();
  //    let replaced_term_let = lang::read_term(REPLACED_TERM_LET).unwrap();
  //    let replaced_term_lam = lang::read_term(REPLACED_TERM_LAM).unwrap();
  //    let replaced_term_app = lang::read_term(REPLACED_TERM_APP).unwrap();
  //    let replaced_term_ctr = lang::read_term(REPLACED_TERM_CTR).unwrap();
  //    let replaced_term_u32 = lang::read_term(REPLACED_TERM_U32).unwrap();
  //    let replaced_term_op2 = lang::read_term(REPLACED_TERM_OP2).unwrap();
  //
  //    assert_eq!(replace(term_from, &term_to, *term_var), *replaced_term_var);
  //    assert_eq!(replace(term_from, &term_to, *term_dup), *replaced_term_dup);
  //    assert_eq!(replace(term_from, &term_to, *term_let), *replaced_term_let);
  //    assert_eq!(replace(term_from, &term_to, *term_lam), *replaced_term_lam);
  //    assert_eq!(replace(term_from, &term_to, *term_app), *replaced_term_app);
  //    assert_eq!(replace(term_from, &term_to, *term_ctr), *replaced_term_ctr);
  //    assert_eq!(replace(term_from, &term_to, *term_u32), *replaced_term_u32);
  //    assert_eq!(replace(term_from, &term_to, *term_op2), *replaced_term_op2);
  //  }
  //
  //  // TODO add tests that return None
  //  #[test]
  //  fn flatten_first_layer_0() {
  //    let nested: lang::Rule = lang::read_rule(EQ0).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ0_FIRST_LAYER).unwrap().unwrap();
  //    assert_eq!(first_layer(&nested, 0), expected_first_layer);
  //  }
  //
  //  #[test]
  //  fn flatten_first_layer_1() {
  //    let nested: lang::Rule = lang::read_rule(EQ1).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ1_FIRST_LAYER).unwrap().unwrap();
  //    assert_eq!(first_layer(&nested, 0), expected_first_layer);
  //  }
  //
  //  #[test]
  //  fn flatten_first_layer_2() {
  //    let nested: lang::Rule = lang::read_rule(EQ2).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ2_FIRST_LAYER).unwrap().unwrap();
  //    assert_eq!(first_layer(&nested, 0), expected_first_layer);
  //  }
  //
  //  #[test]
  //  fn flatten_denest_with_pattern_0() {
  //    let nested: lang::Rule = lang::read_rule(EQ0).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ0_AUX_DEF).unwrap().unwrap();
  //    assert_eq!(denest_with_pattern(&nested, &nested, 0), expected_first_layer);
  //  }
  //
  //  #[test]
  //  fn flatten_denest_with_pattern_1() {
  //    let nested: lang::Rule = lang::read_rule(EQ1).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ1_AUX_DEF).unwrap().unwrap();
  //    assert_eq!(denest_with_pattern(&nested, &nested, 0), expected_first_layer);
  //  }
  //
  //  #[test]
  //  fn flatten_denest_with_pattern_2() {
  //    let nested: lang::Rule = lang::read_rule(EQ2).unwrap().unwrap();
  //    let expected_first_layer: lang::Rule = lang::read_rule(EQ2_AUX_DEF).unwrap().unwrap();
  //    assert_eq!(denest_with_pattern(&nested, &nested, 0), expected_first_layer);
  //  }

  #[test]
  fn flatten_0() {
    let nested: Vec<lang::Rule> = lang::read_file(FILE_7).unwrap().rules;
    //    let first_layer_rule = first_layer(&nested[0], 0);
    //    let denested = denest_with_pattern(&first_layer_rule, &nested[1], 0);
    //    let denested_first_layer = first_layer(&denested, 0);
    let denested = super::flatten(&nested);
    println!("");
    for rule in denested {
      println!("{}", rule);
    }
  }
  //  #[test]
  //  fn freshen_0() {
  //    let unfresh: lang::Term = *lang::read_term("(Pair (Pair a b) 2 (Succ n))").unwrap();
  //    let (fresh, subs, i) = freshen_vars(unfresh, Vec::new(), 0);
  //    assert_eq!(fresh, *lang::read_term("(Pair (Pair x.0 x.1) 2 (Succ x.2))").unwrap());
  //    assert_eq!(subs, vec![("a".to_string(), Box::new(lang::Term::Var { name: "x.0".to_string() })), ("b".to_string(), Box::new(lang::Term::Var { name: "x.1".to_string() })), ("n".to_string(), Box::new(lang::Term::Var { name: "x.2".to_string() }))]);
  //    assert_eq!(i, 3);
  //  }
  //  #[test]
  //  fn subpattern_thing() {
  //    let reference = lang::read_rule("(Pattern A) = 0").unwrap().unwrap();
  //    let catchall = lang::read_rule("(Pattern n) = n").unwrap().unwrap();
  //    if subpattern(&reference, &catchall) {
  //      println!("if");
  //    } else {
  //      println!("else");
  //    }
  //  }
}
