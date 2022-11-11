use crate::runtime::{*};

// dup x y = {a b}
// --------------- DUP-DUP
// x <- a
// y <- b
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), take_arg(heap, arg0, 0));
  atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), take_arg(heap, arg0, 1));
  free(heap, tid, get_loc(term, 0), 3);
  free(heap, tid, get_loc(arg0, 0), 2);
}
