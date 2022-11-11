use crate::runtime::{*};

// dup x y = (K a b c ...)
// ----------------------- DUP-CTR
// dup a0 a1 = a
// dup b0 b1 = b
// dup c0 c1 = c
// ...
// x <- (K a0 b0 c0 ...)
// y <- (K a1 b1 c1 ...)
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, tcol: u64) {
  inc_cost(heap, tid);
  let fnum = get_ext(arg0);
  let fari = arity_of(arit, arg0);
  if fari == 0 {
    atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnum, 0));
    atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnum, 0));
    link(heap, host, Ctr(fnum, 0));
    free(heap, tid, get_loc(term, 0), 3);
  } else {
    let ctr0 = get_loc(arg0, 0);
    let ctr1 = alloc(heap, tid, fari);
    for i in 0 .. fari - 1 {
      let leti = alloc(heap, tid, 3);
      link(heap, leti + 2, take_arg(heap, arg0, i));
      link(heap, ctr0 + i, Dp0(get_ext(term), leti));
      link(heap, ctr1 + i, Dp1(get_ext(term), leti));
    }
    let leti = alloc(heap, tid, 3);
    link(heap, leti + 2, take_arg(heap, arg0, fari - 1));
    link(heap, ctr0 + fari - 1, Dp0(get_ext(term), leti));
    link(heap, ctr1 + fari - 1, Dp1(get_ext(term), leti));
    atomic_subst(heap, arit, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnum, ctr0));
    atomic_subst(heap, arit, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnum, ctr1));
    //let done = Ctr(fnum, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
    //link(heap, host, done);
    free(heap, tid, get_loc(term, 0), 3);
  }
}
