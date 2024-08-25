#ifndef config_cuh_INCLUDED
#define config_cuh_INCLUDED

#include "common.cuh"

// Clocks per Second
const u64 S = 2520000000;

// Threads per Block
const u32 TPB_L2 = 7;
const u32 TPB    = 1 << TPB_L2;

// Blocks per GPU
const u32 BPG_L2 = 7;
const u32 BPG    = 1 << BPG_L2;

// Threads per GPU
const u32 TPG = TPB * BPG;

// Thread Redex Bag Length
const u32 RLEN = 256;

// Local Net
const u32 L_NODE_LEN = 0x2000;
const u32 L_VARS_LEN = 0x2000;

// Global Net
const u32 G_NODE_LEN = 1 << 29; // max 536m nodes
const u32 G_VARS_LEN = 1 << 29; // max 536m vars
const u32 G_RBAG_LEN = TPB * BPG * RLEN * 3; // max 4m redexes

#endif // config_cuh_INCLUDED
