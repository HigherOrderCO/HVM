use crate::runtime::{*};

// dup x y = *
// ----------- DUP-ERA
// x <- *
// y <- *
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), Era());
  atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), Era());
  link(heap, host, Era());
  free(heap, tid, get_loc(term, 0), 3);
}
