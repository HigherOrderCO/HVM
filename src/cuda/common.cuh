#ifndef common_cuh_INCLUDED
#define common_cuh_INCLUDED

#include <stdint.h>

typedef  uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;
typedef  int32_t i32;
typedef    float f32;
typedef   double f64;

// Clamps a float between two values.
__host__ __device__ inline f32 clamp(f32 x, f32 min, f32 max) {
  const f32 t = x < min ? min : x;
  return (t > max) ? max : t;
}

// TODO: write a time64() function that returns the time as fast as possible as a u64
__host__ __device__ inline u64 time64() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

#endif // common_cuh_INCLUDED
