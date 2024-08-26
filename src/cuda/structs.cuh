#ifndef structs_cuh_INCLUDED
#define structs_cuh_INCLUDED

#include "port.cuh"
#include "rule.cuh"
#include "config.cuh"
#include "constants.cuh"

// Structures
// ----------

// Global Net
struct GNet {
  u32  rbag_use_A; // total rbag redex count (buffer A)
  u32  rbag_use_B; // total rbag redex count (buffer B)
  Pair rbag_buf_A[G_RBAG_LEN]; // global redex bag (buffer A)
  Pair rbag_buf_B[G_RBAG_LEN]; // global redex bag (buffer B)
  Pair node_buf[G_NODE_LEN]; // global node buffer
  Port vars_buf[G_VARS_LEN]; // global vars buffer
  u32  node_put[TPB*BPG];
  u32  vars_put[TPB*BPG];
  u32  rbag_pos[TPB*BPG];
  u8   mode; // evaluation mode (curr)
  u64  itrs; // interaction count
  u64  iadd; // interaction count adder
  u64  leak; // leak count
  u32  turn; // turn count
  u8   down; // are we recursing down?
  u8   rdec; // decrease rpos by 1?
};

// Local Net
struct LNet {
  Pair node_buf[L_NODE_LEN];
  Port vars_buf[L_VARS_LEN];
};

// Thread Redex Bag
// It uses the same space to store two stacks:
// - HI: a high-priotity stack, for shrinking reductions
// - LO: a low-priority stack, for growing reductions
struct RBag {
  u32  hi_end;
  Pair hi_buf[RLEN];
  u32  lo_end;
  Pair lo_buf[RLEN];
};

// Thread Memory
struct TM {
  u32  page; // page index
  u32  nput; // node alloc index
  u32  vput; // vars alloc index
  u32  mode; // evaluation mode
  u32  itrs; // interactions
  u32  leak; // leaks
  u32  nloc[L_NODE_LEN/TPB]; // node allocs
  u32  vloc[L_NODE_LEN/TPB]; // vars allocs
  RBag rbag; // tmem redex bag
};

// Top-Level Definition
struct Def {
  char name[256];
  bool safe;
  u32  rbag_len;
  u32  node_len;
  u32  vars_len;
  Port root;
  Pair rbag_buf[L_NODE_LEN/TPB];
  Pair node_buf[L_NODE_LEN/TPB];
};

// A Foreign Function
typedef struct {
  char name[256];
  Port (*func)(GNet*, Port);
} FFn;

// Book of Definitions
struct Book {
  u32 defs_len;
  Def defs_buf[0x4000];
  u32 ffns_len;
  FFn ffns_buf[0x4000];
};

// Static Book
__device__ Book BOOK;

// View Net: includes both GNet and LNet
struct Net {
  i32   l_node_dif; // delta node space
  i32   l_vars_dif; // delta vars space
  Pair *l_node_buf; // local node buffer values
  Port *l_vars_buf; // local vars buffer values
  u32  *g_rbag_use_A; // global rbag count (active buffer)
  u32  *g_rbag_use_B; // global rbag count (inactive buffer)
  Pair *g_rbag_buf_A; // global rbag values (active buffer)
  Pair *g_rbag_buf_B; // global rbag values (inactive buffer)
  Pair *g_node_buf; // global node buffer values
  Port *g_vars_buf; // global vars buffer values
  u32  *g_node_put; // next global node allocation index
  u32  *g_vars_put; // next global vars allocation index
};

#endif // structs_cuh_INCLUDED
