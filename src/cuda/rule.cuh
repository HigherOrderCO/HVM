#ifndef rule_cuh_INCLUDED
#define rule_cuh_INCLUDED

#include "common.cuh"
#include "port.cuh"
#include "pair.cuh"

// Given two tags, gets their interaction rule. Uses a u64mask lookup table.
__device__ __host__ inline Rule get_rule(Port A, Port B) {
  const u64 x = 0b0111111010110110110111101110111010110000111100001111000100000010;
  const u64 y = 0b0000110000001100000011100000110011111110111111100010111000000000;
  const u64 z = 0b1111100011111000111100001111000011000000000000000000000000000000;
  const u64 i = ((u64)get_tag(A) << 3) | (u64)get_tag(B);
  return (Rule)((x>>i&1) | (y>>i&1)<<1 | (z>>i&1)<<2);
}

// Same as above, but receiving a pair.
__device__ __host__ inline Rule get_pair_rule(Pair AB) {
  return get_rule(get_fst(AB), get_snd(AB));
}

__device__ __host__ inline bool is_high_priority(Rule rule) {
  return (bool)((0b00011101 >> rule) & 1);
}

#endif // rule_cuh_INCLUDED
