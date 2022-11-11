use crate::runtime::{*};

// dup x y = {a b}
// --------------- DUP-SUP
// x <- {xA xB}
// y <- {yA yB}
// dup xA yA = a
// dup xB yB = b
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let par0 = alloc(heap, tid, 2);
  let let0 = alloc(heap, tid, 3);
  let par1 = get_loc(arg0, 0);
  let let1 = alloc(heap, tid, 3);
  link(heap, let0 + 2, take_arg(heap, arg0, 0));
  link(heap, let1 + 2, take_arg(heap, arg0, 1));
  link(heap, par1 + 0, Dp1(tcol, let0));
  link(heap, par1 + 1, Dp1(tcol, let1));
  link(heap, par0 + 0, Dp0(tcol, let0));
  link(heap, par0 + 1, Dp0(tcol, let1));
  atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par0));
  atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par1));
  free(heap, tid, get_loc(term, 0), 3);
}
