use crate::runtime::{*};

// ({a b} c)
// --------------- APP-SUP
// dup x0 x1 = c
// {(a x0) (b x1)}
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr) {
  inc_cost(heap, tid);
  let app0 = get_loc(term, 0);
  let app1 = get_loc(arg0, 0);
  let let0 = alloc(heap, tid, 3);
  let par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, term, 1));
  link(heap, app0 + 1, Dp0(get_ext(arg0), let0));
  link(heap, app0 + 0, take_arg(heap, arg0, 0));
  link(heap, app1 + 0, take_arg(heap, arg0, 1));
  link(heap, app1 + 1, Dp1(get_ext(arg0), let0));
  link(heap, par0 + 0, App(app0));
  link(heap, par0 + 1, App(app1));
  let done = Sup(get_ext(arg0), par0);
  link(heap, host, done);
}
