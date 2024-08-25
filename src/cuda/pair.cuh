#ifndef pair_cuh_INCLUDED
#define pair_cuh_INCLUDED

#include "types.cuh"

__host__ __device__ inline Pair new_pair(Port fst, Port snd) {
  return ((u64)snd << 32) | fst;
}

__host__ __device__ inline Port get_fst(Pair pair) {
  return pair & 0xFFFFFFFF;
}

__host__ __device__ inline Port get_snd(Pair pair) {
  return pair >> 32;
}

#endif // pair_cuh_INCLUDED
