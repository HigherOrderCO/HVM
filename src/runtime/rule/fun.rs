use crate::runtime::*;
use std::sync::atomic::Ordering;

#[inline(always)]
pub fn visit(ctx: ReduceCtx, sidxs: &[u64]) -> bool {
  let len = sidxs.len() as u64;
  if len == 0 {
    false
  } else {
    let mut vlen = 0;
    let vbuf = unsafe { ctx.heap.vbuf.get_unchecked(ctx.tid) };
    for sidx in sidxs {
      if !is_whnf(ctx.heap.load_arg(ctx.term, *sidx)) {
        unsafe { vbuf.get_unchecked(vlen) }.store(get_loc(ctx.term, *sidx), Ordering::Relaxed);
        vlen += 1;
      }
    }
    if vlen == 0 {
      false
    } else {
      let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, vlen as u64));
      for i in 0..vlen - 1 {
        ctx.visit.push(new_visit(
          unsafe { vbuf.get_unchecked(i).load(Ordering::Relaxed) },
          ctx.hold,
          goup,
        ));
      }
      *ctx.cont = goup;
      *ctx.host = unsafe { vbuf.get_unchecked(vlen - 1).load(Ordering::Relaxed) };
      true
    }
  }
  //OLD_VISITER:
  //let len = sidxs.len() as u64;
  //if len == 0 {
  //return false;
  //} else {
  //let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, sidxs.len() as u64));
  //for (i, arg_idx) in sidxs.iter().enumerate() {
  //if i < sidxs.len() - 1 {
  //ctx.visit.push(new_visit(get_loc(ctx.term, *arg_idx), goup));
  //} else {
  //*ctx.cont = goup;
  //*ctx.host = get_loc(ctx.term, *arg_idx);
  //return true;
  //}
  //}
  //return true;
  //}
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx, fid: u64, visit: &VisitObj, apply: &ApplyObj) -> bool {
  // Reduces function superpositions
  for (n, is_strict) in visit.strict_map.iter().enumerate() {
    let n = n as u64;
    if *is_strict && get_tag(ctx.heap.load_arg(ctx.term, n)) == SUP {
      superpose(
        ctx.heap,
        &ctx.prog.aris,
        ctx.tid,
        *ctx.host,
        ctx.term,
        ctx.heap.load_arg(ctx.term, n),
        n,
      );
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
        U60 => {
          let same_tag = get_tag(ctx.heap.load_arg(ctx.term, i)) == U60;
          let same_val = get_num(ctx.heap.load_arg(ctx.term, i)) == get_num(*cond);
          matched = matched && same_tag && same_val;
        }
        F60 => {
          let same_tag = get_tag(ctx.heap.load_arg(ctx.term, i)) == F60;
          let same_val = get_num(ctx.heap.load_arg(ctx.term, i)) == get_num(*cond);
          matched = matched && same_tag && same_val;
        }
        CTR => {
          let same_tag = get_tag(ctx.heap.load_arg(ctx.term, i)) == CTR;
          let same_ext = get_ext(ctx.heap.load_arg(ctx.term, i)) == get_ext(*cond);
          matched = matched && same_tag && same_ext;
        }
        VAR => {
          // If this is a strict argument, then we're in a default variable
          if unsafe { *visit.strict_map.get_unchecked(i as usize) } {
            // This is a Kind2-specific optimization.
            if rule.hoas && r != apply.rules.len() - 1 {
              // Matches number literals
              let is_num = get_tag(ctx.heap.load_arg(ctx.term, i)) == U60
                || get_tag(ctx.heap.load_arg(ctx.term, i)) == F60;

              // Matches constructor labels
              let is_ctr = get_tag(ctx.heap.load_arg(ctx.term, i)) == CTR
                && arity_of(&ctx.prog.aris, ctx.heap.load_arg(ctx.term, i)) == 0;

              // Matches HOAS numbers and constructors
              let is_hoas_ctr_num = get_tag(ctx.heap.load_arg(ctx.term, i)) == CTR
                && get_ext(ctx.heap.load_arg(ctx.term, i)) >= KIND_TERM_CT0
                && get_ext(ctx.heap.load_arg(ctx.term, i)) <= KIND_TERM_F60;

              matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

            // Only match default variables on CTRs and NUMs
            } else {
              let is_ctr = get_tag(ctx.heap.load_arg(ctx.term, i)) == CTR;
              let is_u60 = get_tag(ctx.heap.load_arg(ctx.term, i)) == U60;
              let is_f60 = get_tag(ctx.heap.load_arg(ctx.term, i)) == F60;
              matched = matched && (is_ctr || is_u60 || is_f60);
            }
          }
        }
        _ => {}
      }
    }

    // If all conditions are satisfied, the rule matched, so we must apply it
    if matched {
      // Increments the gas count
      ctx.heap.inc_cost(ctx.tid);

      // Builds the right-hand side ctx.term
      let done = alloc_body(ctx.heap, ctx.prog, ctx.tid, ctx.term, &rule.vars, &rule.body);

      // Links the *ctx.host location to it
      link(ctx.heap, *ctx.host, done);

      // Collects unused variables
      for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
        if *erase {
          collect(ctx.heap, &ctx.prog.aris, ctx.tid, get_var(ctx.heap, ctx.term, var));
        }
      }

      // free the matched ctrs
      for (i, arity) in &rule.free {
        free(ctx.heap, ctx.tid, get_loc(ctx.heap.load_arg(ctx.term, *i), 0), *arity);
      }
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), arity_of(&ctx.prog.aris, fid));

      return true;
    }
  }

  false
}

#[inline(always)]
pub fn superpose(
  heap: &Heap,
  aris: &Aris,
  tid: usize,
  host: u64,
  term: Ptr,
  argn: Ptr,
  n: u64,
) -> Ptr {
  heap.inc_cost(tid);
  let arit = arity_of(aris, term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(heap, tid, arit);
  let par0 = get_loc(argn, 0);
  for i in 0..arit {
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
