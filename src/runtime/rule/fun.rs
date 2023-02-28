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
        unsafe { vbuf.get_unchecked(vlen) }.store(ctx.term.loc(*sidx), Ordering::Relaxed);
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
  //ctx.visit.push(new_visit(ctx.term.loc(*arg_idx), goup));
  //} else {
  //*ctx.cont = goup;
  //*ctx.host = ctx.term.loc(*arg_idx);
  //return true;
  //}
  //}
  //return true;
  //}
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx, fid: Ptr, visit: &VisitObj, apply: &ApplyObj) -> bool {
  // Reduces function superpositions
  for (n, is_strict) in visit.strict_map.iter().enumerate() {
    let n = n as u64;
    if *is_strict && ctx.heap.load_arg(ctx.term, n).tag() == Tag::SUP {
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

    // Tests each rule condition (ex: `args[0].tag() == SUCC`)
    for (i, cond) in rule.cond.iter().enumerate() {
      let ptr = ctx.heap.load_arg(ctx.term, i as _);
      match cond.tag() {
        Tag::U60 => {
          let same_tag = ptr.tag() == Tag::U60;
          let same_val = ptr.num() == cond.num();
          matched = matched && same_tag && same_val;
        }
        Tag::F60 => {
          let same_tag = ptr.tag() == Tag::F60;
          let same_val = ptr.num() == cond.num();
          matched = matched && same_tag && same_val;
        }
        Tag::CTR => {
          let same_tag = ptr.tag() == Tag::CTR;
          let same_ext = ptr.ext() == cond.ext();
          matched = matched && same_tag && same_ext;
        }
        Tag::VAR => {
          // If this is a strict argument, then we're in a default variable
          if unsafe { *visit.strict_map.get_unchecked(i) } {
            // This is a Kind2-specific optimization.
            if rule.hoas && r != apply.rules.len() - 1 {
              // Matches number literals
              let is_num = ptr.tag().is_numeric();

              // Matches constructor labels
              let is_ctr = ptr.tag() == Tag::CTR && ctx.prog.aris.arity_of(ptr) == 0;

              // Matches HOAS numbers and constructors
              let is_hoas_ctr_num =
                ptr.tag() == Tag::CTR && ptr.ext() >= KIND_TERM_CT0 && ptr.ext() <= KIND_TERM_F60;

              matched = matched && (is_num || is_ctr || is_hoas_ctr_num);

            // Only match default variables on CTRs and NUMs
            } else {
              let is_ctr = ptr.tag() == Tag::CTR;
              let is_num = ptr.tag().is_numeric();
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
      ctx.heap.inc_cost(ctx.tid);

      // Builds the right-hand side ctx.term
      let done = ctx.heap.alloc_body(ctx.prog, ctx.tid, ctx.term, &rule.vars, &rule.body);

      // Links the *ctx.host location to it
      ctx.heap.link(*ctx.host, done);

      // Collects unused variables
      for var @ RuleVar { param: _, field: _, erase } in rule.vars.iter() {
        if *erase {
          ctx.heap.collect(&ctx.prog.aris, ctx.tid, ctx.heap.get_var(ctx.term, var));
        }
      }

      // free the matched ctrs
      for (i, arity) in &rule.free {
        let t = ctx.heap.load_arg(ctx.term.loc(*i).into(), 0);
        ctx.heap.free(ctx.tid, t.into(), *arity);
      }
      ctx.heap.free(ctx.tid, ctx.term.loc(0), ctx.prog.aris.arity_of(fid));

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
  let arit = aris.arity_of(term);
  let func = term.ext();
  let fun0 = term.loc(0);
  let fun1 = heap.alloc(tid, arit);
  let par0 = argn.loc(0);
  for i in 0..arit {
    if i != n {
      let leti = heap.alloc(tid, 3);
      let argi = heap.take_arg(term, i);
      heap.link(fun0 + i, Dp0(argn.ext(), leti));
      heap.link(fun1 + i, Dp1(argn.ext(), leti));
      heap.link(leti + 2, argi);
    } else {
      heap.link(fun0 + i, heap.take_arg(argn, 0));
      heap.link(fun1 + i, heap.take_arg(argn, 1));
    }
  }
  heap.link(par0 + 0, Fun(func, fun0));
  heap.link(par0 + 1, Fun(func, fun1));
  let done = Sup(argn.ext(), par0);
  heap.link(host, done);
  done
}
