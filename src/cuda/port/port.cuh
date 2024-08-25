#ifndef port_port_cuh_INCLUDED
#define port_port_cuh_INCLUDED

#include "../types.cuh"
#include "../config.cuh"

__host__ __device__ inline Port new_port(Tag tag, Val val) {
  return (val << 3) | tag;
}

__host__ __device__ inline Tag get_tag(Port port) {
  return port & 7;
}

__host__ __device__ inline Val get_val(Port port) {
  return port >> 3;
}

// True if this port has a pointer to a node.
__device__ __host__ inline bool is_nod(Port a) {
  return get_tag(a) >= CON;
}

// True if this port is a variable.
__device__ __host__ inline bool is_var(Port a) {
  return get_tag(a) == VAR;
}

// True if this port is a local node/var (that can leak).
__device__ __host__ inline bool is_local(Port a) {
  return (is_nod(a) || is_var(a)) && get_val(a) < L_NODE_LEN;
}

// True if this port is a global node/var (that can be leaked into).
__device__ __host__ inline bool is_global(Port a) {
  return (is_nod(a) || is_var(a)) && get_val(a) >= L_NODE_LEN;
}

// TODO(enricozb): move swap stuff to some internal file

// Swaps two ports.
__host__ __device__ inline void swap(Port *a, Port *b) {
  Port x = *a; *a = *b; *b = x;
}

// Should we swap ports A and B before reducing this rule?
__device__ __host__ inline bool should_swap(Port A, Port B) {
  return get_tag(B) < get_tag(A);
}

#endif // port_port_cuh_INCLUDED
