use crate::runtime::{*};

// dup r s = λx(f)
// --------------- DUP-LAM
// dup f0 f1 = f
// r <- λx0(f0)
// s <- λx1(f1)
// x <- {x0 x1}
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  let lam0 = alloc(heap, tid, 2);
  let lam1 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, arg0, 1));
  link(heap, par0 + 1, Var(lam1));
  link(heap, par0 + 0, Var(lam0));
  link(heap, lam0 + 1, Dp0(get_ext(term), let0));
  link(heap, lam1 + 1, Dp1(get_ext(term), let0));
  atomic_subst(heap, arit, tid, Var(get_loc(arg0, 0)), Sup(get_ext(term), par0));
  atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), Lam(lam0));
  atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), Lam(lam1));
  let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
  link(heap, host, done);
  free(heap, tid, get_loc(term, 0), 3);
  free(heap, tid, get_loc(arg0, 0), 2);
}
