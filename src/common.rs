use crate::lambolt::{File, Rule, Term};
use std::{collections::HashMap, u128};

pub type IdTable = HashMap<String, u64>;

// Generates a name table for a whole program. That table links constructor
// names (such as `cons` and `succ`) to small ids (such as `0` and `1`).
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

pub type IsFunctionTable = HashMap<String, bool>;

// Finds constructors that are used as functions.
pub fn gen_is_call(file: &File) -> IsFunctionTable {
  let mut is_call: IsFunctionTable = HashMap::new();
  let rules = &file.rules;
  for rule in rules {
    let term = &rule.lhs;
    if let Term::Ctr { name, .. } = &**term {
      // FIXME: this looks wrong, will check later
      is_call.insert(name.clone(), true);
    }
  }
  is_call
}

pub type GroupTable<'a, 'b> = HashMap<String, (usize, Vec<&'a Rule>)>;
// Groups rules by name. For example:
//   (add (succ a) (succ b)) = (succ (succ (add a b)))
//   (add (succ a) (zero)  ) = (succ a)
//   (add (zero)   (succ b)) = (succ b)
//   (add (zero)   (zero)  ) = (zero)
// This is a group of 4 rules starting with the "add" name.
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
