#ifndef types_cuh_INCLUDED
#define types_cuh_INCLUDED

#include "common.cuh"
#include "config.cuh"

// Values
// ------
// Val ::= 29-bit (rounded up to u32)
// The 29 least significant bits in a Port
typedef u32 Val;

// Tags
// ----
// Tag ::= 3-bit (rounded up to u8)
// The 3 most significant bits in a Port.
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

// Numbs
// -----
// Numb ::= 29-bit (rounded up to u32)
// One of the 8 possible kinds of values of a port.
typedef u32 Numb;

#endif // types_cuh_INCLUDED
