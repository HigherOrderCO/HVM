use crate::runtime::{*};

#[inline(always)]
pub fn apply(heap: &Heap, prog: &Program, tid: usize, host: u64, term: Ptr, fid: u64, arity: u64, visit: &VisitObj, apply: &ApplyObj) -> bool {
  // Reduces function superpositions
  for (n, is_strict) in visit.strict_map.iter().enumerate() {
    let n = n as u64;
    if *is_strict && get_tag(load_arg(heap, term, n)) == SUP {
      fun_sup::apply(heap, &prog.arit, tid, host, term, load_arg(heap, term, n), n);
      return true;
    }
  }

  // For each rule condition vector
  let mut matched;
  for (r, rule) in apply.rules.iter().enumerate() {
    // Check if the rule matches
    matched = true;
    
    // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
    for (i, cond) in rule.cond.iter().enumerate() {
      let i = i as u64;
      match get_tag(*cond) {
        NUM => {
          let same_tag = get_tag(load_arg(heap, term, i)) == NUM;
          let same_val = get_num(load_arg(heap, term, i)) == get_num(*cond);
          matched = matched && same_tag && same_val;
        }
        CTR => {
          let same_tag = get_tag(load_arg(heap, term, i)) == CTR;
          let same_ext = get_ext(load_arg(heap, term, i)) == get_ext(*cond);
          matched = matched && same_tag && same_ext;
        }
        VAR => {
          // If this is a strict argument, then we're in a default variable
          if unsafe { *visit.strict_map.get_unchecked(i as usize) } {

            // This is a Kind2-specific optimization. Check 'KIND_TERM_OPT'.
            if rule.hoas && r != apply.rules.len() - 1 {

              // Matches number literals
              let is_num
                = get_tag(load_arg(heap, term, i)) == NUM;

              // Matches constructor labels
              let is_ctr
                =  get_tag(load_arg(heap, term, i)) == CTR
                && arity_of(&prog.arit, load_arg(heap, term, i)) == 0;

              // Matches KIND_TERM numbers and constructors
              let is_hoas_ctr_num
                =  get_tag(load_arg(heap, term, i)) == CTR
                && get_ext(load_arg(heap, term, i)) >= KIND_TERM_CT0
                && get_ext(load_arg(heap, term, i)) <= KIND_TERM_NUM;

              matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

            // Only match default variables on CTRs and NUMs
            } else {
              let is_ctr = get_tag(load_arg(heap, term, i)) == CTR;
              let is_num = get_tag(load_arg(heap, term, i)) == NUM;
              matched = matched && (is_ctr || is_num);
            }
          }
        }
        _ => {}
      }
    }

    // If all conditions are satisfied, the rule matched, so we must apply it
    if matched {
      // Increments the gas count
      inc_cost(heap, tid);

      // Builds the right-hand side term
      let done = alloc_body(heap, tid, term, &rule.vars, &rule.body);

      // Links the host location to it
      link(heap, host, done);

      // Collects unused variables
      for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
        if *erase {
          collect(heap, &prog.arit, tid, get_var(heap, term, var));
        }
      }

      // free the matched ctrs
      for (i, arity) in &rule.free {
        free(heap, tid, get_loc(load_arg(heap, term, *i as u64), 0), *arity);
      }
      free(heap, tid, get_loc(term, 0), arity);

      return true;
    }
  }

  return false;
}
