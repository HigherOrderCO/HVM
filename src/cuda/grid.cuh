#ifndef grid_cuh_INCLUDED
#define grid_cuh_INCLUDED

#include "types.cuh"

__device__ inline u32 TID() {
  return threadIdx.x;
}

__device__ inline u32 BID() {
  return blockIdx.x;
}

__device__ inline u32 GID() {
  return TID() + BID() * blockDim.x;
}

#endif // grid_cuh_INCLUDED
