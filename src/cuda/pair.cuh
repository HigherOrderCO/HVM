#ifndef pair_cuh_INCLUDED
#define pair_cuh_INCLUDED

#include "common.cuh"
#include "port.cuh"

__host__ __device__ inline Pair new_pair(Port fst, Port snd) {
  return ((u64)snd << 32) | fst;
}

__host__ __device__ inline Port get_fst(Pair pair) {
  return pair & 0xFFFFFFFF;
}

__host__ __device__ inline Port get_snd(Pair pair) {
  return pair >> 32;
}

// Parity Flag
// -----------

__device__ __host__ Pair set_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) | 0x10000000), p2);
  } else {
    return pair;
  }
}

__device__ __host__ Pair clr_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) & 0xFFFFFFF), p2);
  } else {
    return pair;
  }
}

__device__ __host__ bool get_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  if (get_tag(p1) == REF) {
    return (get_val(p1) >> 28) == 1;
  } else {
    return false;
  }
}

#endif // pair_cuh_INCLUDED
