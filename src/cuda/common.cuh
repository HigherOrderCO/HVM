// Common type aliases and functions used across the codebase that aren't
// necessarily associated with any specific part.

#ifndef common_cuh_INCLUDED
#define common_cuh_INCLUDED

#include <stdint.h>

// Numeric Type Aliases
// --------------------
typedef  uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;
typedef  int32_t i32;
typedef    float f32;
typedef   double f64;

// Values
// ------
// Val ::= 29-bit (rounded up to u32)
// The 29 least significant bits in a Port
typedef u32 Val;

// Tags
// ----
// Tag ::= 3-bit (rounded up to u8)
// These are the 3 most significant bits in a Port, and
// they identify the type of port.
typedef u8 Tag;

const Tag VAR = 0x0; // variable
const Tag REF = 0x1; // reference
const Tag ERA = 0x2; // eraser
const Tag NUM = 0x3; // number
const Tag CON = 0x4; // constructor
const Tag DUP = 0x5; // duplicator
const Tag OPR = 0x6; // operator
const Tag SWI = 0x7; // switch

// Ports
// -----
// Port ::= Tag + Val (fits a u32)
typedef u32 Port;

/// Pair
/// ----
// Pair ::= Port + Port (fits a u64)
typedef u64 Pair;

// Interaction Rules
// -----------------
// Rule ::= 3-bit (rounded up to 8)
typedef u8 Rule;

const Rule LINK = 0x0;
const Rule CALL = 0x1;
const Rule VOID = 0x2;
const Rule ERAS = 0x3;
const Rule ANNI = 0x4;
const Rule COMM = 0x5;
const Rule OPER = 0x6;
const Rule SWIT = 0x7;

// Grid Functions
// --------------

__device__ inline u32 TID() {
  return threadIdx.x;
}

__device__ inline u32 BID() {
  return blockIdx.x;
}

__device__ inline u32 GID() {
  return TID() + BID() * blockDim.x;
}

#endif // common_cuh_INCLUDED
