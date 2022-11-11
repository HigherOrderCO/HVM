use crate::runtime::{*};

#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, argn: Ptr, n: u64) -> Ptr {
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

