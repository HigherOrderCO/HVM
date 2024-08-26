#ifndef structs_tm_cuh_INCLUDED
#define structs_tm_cuh_INCLUDED

#include "../structs.cuh"
#include "rbag.cuh"

__device__ TM tmem_new() {
  TM tm;
  tm.rbag = rbag_new();
  tm.nput = 1;
  tm.vput = 1;
  tm.mode = SEED;
  tm.itrs = 0;
  tm.leak = 0;
  return tm;
}

#endif // structs_tm_cuh_INCLUDED
