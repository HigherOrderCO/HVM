#ifndef constants_cuh_INCLUDED
#define constants_cuh_INCLUDED

#include "common.cuh"

// Special Substitution Map Values
const Port FREE = 0x00000000;
const Port ROOT = 0xFFFFFFF8;
const Port NONE = 0xFFFFFFFF;

// Evaluation Modes
const u8 SEED = 0;
const u8 GROW = 1;
const u8 WORK = 2;

#endif // constants_cuh_INCLUDED
