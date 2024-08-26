// Synchronization utilities and debugging functions.

#ifndef sync_cuh_INCLUDED
#define sync_cuh_INCLUDED

#include "config.cuh"

// Returns true if all 'x' are true, block-wise
__device__ __noinline__ bool block_all(bool x) {
  __shared__ bool res;
  if (TID() == 0) res = true;
  __syncthreads();
  if (!x) res = false;
  __syncthreads();
  return res;
}

// Returns true if any 'x' is true, block-wise
__device__ __noinline__ bool block_any(bool x) {
  __shared__ bool res;
  if (TID() == 0) res = false;
  __syncthreads();
  if (x) res = true;
  __syncthreads();
  return res;
}

// Returns the sum of a value, block-wise
template <typename A>
__device__ __noinline__ A block_sum(A x) {
  __shared__ A res;
  if (TID() == 0) res = 0;
  __syncthreads();
  atomicAdd(&res, x);
  __syncthreads();
  return res;
}

// Returns the sum of a boolean, block-wise
__device__ __noinline__ u32 block_count(bool x) {
  __shared__ u32 res;
  if (TID() == 0) res = 0;
  __syncthreads();
  atomicAdd(&res, x);
  __syncthreads();
  return res;
}

// Prints a 4-bit value for each thread in a block
__device__ void block_print(u32 x) {
  __shared__ u8 value[TPB];

  value[TID()] = x;
  __syncthreads();

  if (TID() == 0) {
    for (u32 i = 0; i < TPB; ++i) {
      printf("%x", min(value[i],0xF));
    }
  }
  __syncthreads();
}

__device__ u32 NODE_COUNT;
__device__ u32 VARS_COUNT;

__global__ void count_memory(GNet* gnet) {
  u32 node_count = 0;
  u32 vars_count = 0;
  for (u32 i = GID(); i < G_NODE_LEN; i += TPG) {
    if (gnet->node_buf[i] != 0) ++node_count;
    if (gnet->vars_buf[i] != 0) ++vars_count;
  }

  __shared__ u32 block_node_count;
  __shared__ u32 block_vars_count;

  if (TID() == 0) block_node_count = 0;
  if (TID() == 0) block_vars_count = 0;
  __syncthreads();

  atomicAdd(&block_node_count, node_count);
  atomicAdd(&block_vars_count, vars_count);
  __syncthreads();

  if (TID() == 0) atomicAdd(&NODE_COUNT, block_node_count);
  if (TID() == 0) atomicAdd(&VARS_COUNT, block_vars_count);
}

#endif // sync_cuh_INCLUDED
