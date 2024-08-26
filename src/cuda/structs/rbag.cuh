#ifndef structs_rbag_cuh_INCLUDED
#define structs_rbag_cuh_INCLUDED

#include "../structs.cuh"

__device__ RBag rbag_new() {
  RBag rbag;
  rbag.hi_end = 0;
  rbag.lo_end = 0;
  return rbag;
}

__device__ u32 rbag_len(RBag* rbag) {
  return rbag->hi_end + rbag->lo_end;
}

__device__ u32 rbag_has_highs(RBag* rbag) {
  return rbag->hi_end > 0;
}

__device__ void push_redex(TM* tm, Pair redex) {
  #ifdef DEBUG
  bool free_hi = tm->rbag.hi_end < RLEN;
  bool free_lo = tm->rbag.lo_end < RLEN;
  if (!free_hi || !free_lo) {
    debug("push_redex: limited resources, maybe corrupting memory\n");
  }
  #endif

  Rule rule = get_pair_rule(redex);
  if (is_high_priority(rule)) {
    tm->rbag.hi_buf[tm->rbag.hi_end++] = redex;
  } else {
    tm->rbag.lo_buf[tm->rbag.lo_end++] = redex;
  }
}

__device__ Pair pop_redex(TM* tm) {
  if (tm->rbag.hi_end > 0) {
    return tm->rbag.hi_buf[(--tm->rbag.hi_end) % RLEN];
  } else if (tm->rbag.lo_end > 0) {
    return tm->rbag.lo_buf[(--tm->rbag.lo_end) % RLEN];
  } else {
    return 0;
  }
}

#endif // structs_rbag_cuh_INCLUDED
