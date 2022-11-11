use crate::runtime::{*};

// (λx(body) a)
// ------------ APP-LAM
// x <- a
// body
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr) {
  //println!("app-lam");
  inc_cost(heap, tid);
  atomic_subst(heap, arit, tid, Var(get_loc(arg0, 0)), take_arg(heap, term, 1));
  link(heap, host, take_arg(heap, arg0, 1));
  free(heap, tid, get_loc(term, 0), 2);
  free(heap, tid, get_loc(arg0, 0), 2);
}
