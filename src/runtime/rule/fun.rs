use crate::runtime::{*};

#[inline(always)]
pub fn visit(ctx: ReduceCtx, sidxs: &[u64]) -> bool {
  let len = sidxs.len() as u64;
  if len == 0 {
    return false;
  } else {
    let mut count = 0;
    for (i, sidx) in sidxs.iter().enumerate() {
      if !is_whnf(load_arg(ctx.heap, ctx.term, *sidx)) {
        count += 1;
      }
    }
    if count == 0 {
      return false;
    } else {
      let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, count));
      *ctx.cont = goup;
      *ctx.host = u64::MAX;
      for (i, sidx) in sidxs.iter().enumerate() {
        if !is_whnf(load_arg(ctx.heap, ctx.term, *sidx)) {
          if *ctx.host != u64::MAX {
            ctx.visit.push(new_visit(goup, *ctx.host));
          }
          *ctx.host = get_loc(ctx.term, *sidx);
        }
      }
      return true;
    }
  }
  //OLD_VISITER:
  //let len = fn_visit.strict_idx.len() as u64;
  //if len == 0 {
    //break 'visit;
  //} else {
    //let goup = redex.insert(tid, new_redex(host, cont, fn_visit.strict_idx.len() as u64));
    //for (i, arg_idx) in fn_visit.strict_idx.iter().enumerate() {
      //if i < fn_visit.strict_idx.len() - 1 {
        //visit.push(new_visit(get_loc(term, *arg_idx), goup));
      //} else {
        //cont = goup;
        //host = get_loc(term, *arg_idx);
        //continue 'visit;
      //}
    //}
  //}
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx, fid: u64, arity: u64, visit: &VisitObj, apply: &ApplyObj) -> bool {
  // Reduces function superpositions
  for (n, is_strict) in visit.strict_map.iter().enumerate() {
    let n = n as u64;
    if *is_strict && get_tag(load_arg(ctx.heap, ctx.term, n)) == SUP {
      superpose(ctx.heap, &ctx.prog.arit, ctx.tid, *ctx.host, ctx.term, load_arg(ctx.heap, ctx.term, n), n);
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
          let same_tag = get_tag(load_arg(ctx.heap, ctx.term, i)) == NUM;
          let same_val = get_num(load_arg(ctx.heap, ctx.term, i)) == get_num(*cond);
          matched = matched && same_tag && same_val;
        }
        CTR => {
          let same_tag = get_tag(load_arg(ctx.heap, ctx.term, i)) == CTR;
          let same_ext = get_ext(load_arg(ctx.heap, ctx.term, i)) == get_ext(*cond);
          matched = matched && same_tag && same_ext;
        }
        VAR => {
          // If this is a strict argument, then we're in a default variable
          if unsafe { *visit.strict_map.get_unchecked(i as usize) } {

            // This is a Kind2-specific optimization. Check 'KIND_ctx.term_OPT'.
            if rule.hoas && r != apply.rules.len() - 1 {

              // Matches number literals
              let is_num
                = get_tag(load_arg(ctx.heap, ctx.term, i)) == NUM;

              // Matches constructor labels
              let is_ctr
                =  get_tag(load_arg(ctx.heap, ctx.term, i)) == CTR
                && arity_of(&ctx.prog.arit, load_arg(ctx.heap, ctx.term, i)) == 0;

              // Matches KIND_ctx.term numbers and constructors
              let is_hoas_ctr_num
                =  get_tag(load_arg(ctx.heap, ctx.term, i)) == CTR
                && get_ext(load_arg(ctx.heap, ctx.term, i)) >= KIND_TERM_CT0
                && get_ext(load_arg(ctx.heap, ctx.term, i)) <= KIND_TERM_NUM;

              matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

            // Only match default variables on CTRs and NUMs
            } else {
              let is_ctr = get_tag(load_arg(ctx.heap, ctx.term, i)) == CTR;
              let is_num = get_tag(load_arg(ctx.heap, ctx.term, i)) == NUM;
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
      inc_cost(ctx.heap, ctx.tid);

      // Builds the right-hand side ctx.term
      let done = alloc_body(ctx.heap, ctx.tid, ctx.term, &rule.vars, &rule.body);

      // Links the *ctx.host location to it
      link(ctx.heap, *ctx.host, done);

      // Collects unused variables
      for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
        if *erase {
          collect(ctx.heap, &ctx.prog.arit, ctx.tid, get_var(ctx.heap, ctx.term, var));
        }
      }

      // free the matched ctrs
      for (i, arity) in &rule.free {
        free(ctx.heap, ctx.tid, get_loc(load_arg(ctx.heap, ctx.term, *i as u64), 0), *arity);
      }
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), arity);

      return true;
    }
  }

  return false;
}

#[inline(always)]
pub fn superpose(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, argn: Ptr, n: u64) -> Ptr {
  inc_cost(heap, tid);
  let arit = arity_of(arit, term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(heap, tid, arit);
  let par0 = get_loc(argn, 0);
  for i in 0 .. arit {
    if i != n {
      let leti = alloc(heap, tid, 3);
      let argi = take_arg(heap, term, i);
      link(heap, fun0 + i, Dp0(get_ext(argn), leti));
      link(heap, fun1 + i, Dp1(get_ext(argn), leti));
      link(heap, leti + 2, argi);
    } else {
      link(heap, fun0 + i, take_arg(heap, argn, 0));
      link(heap, fun1 + i, take_arg(heap, argn, 1));
    }
  }
  link(heap, par0 + 0, Fun(func, fun0));
  link(heap, par0 + 1, Fun(func, fun1));
  let done = Sup(get_ext(argn), par0);
  link(heap, host, done);
  done
}
