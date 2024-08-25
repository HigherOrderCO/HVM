#ifndef structs_cuh_INCLUDED
#define structs_cuh_INCLUDED

#include "port/port.cuh"
#include "rule.cuh"
#include "config.cuh"
#include "grid.cuh"
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

// RBag
// ----

__device__ RBag rbag_new() {
  RBag rbag;
  rbag.hi_end = 0;
  rbag.lo_end = 0;
  return rbag;
}

__device__ u32 rbag_len(RBag* rbag) {
  return rbag->hi_end + rbag->lo_end;
}

__device__ u32 rbag_has_highs(RBag* rbag) {
  return rbag->hi_end > 0;
}

__device__ void push_redex(TM* tm, Pair redex) {
  #ifdef DEBUG
  bool free_hi = tm->rbag.hi_end < RLEN;
  bool free_lo = tm->rbag.lo_end < RLEN;
  if (!free_hi || !free_lo) {
    debug("push_redex: limited resources, maybe corrupting memory\n");
  }
  #endif

  Rule rule = get_pair_rule(redex);
  if (is_high_priority(rule)) {
    tm->rbag.hi_buf[tm->rbag.hi_end++] = redex;
  } else {
    tm->rbag.lo_buf[tm->rbag.lo_end++] = redex;
  }
}

__device__ Pair pop_redex(TM* tm) {
  if (tm->rbag.hi_end > 0) {
    return tm->rbag.hi_buf[(--tm->rbag.hi_end) % RLEN];
  } else if (tm->rbag.lo_end > 0) {
    return tm->rbag.lo_buf[(--tm->rbag.lo_end) % RLEN];
  } else {
    return 0;
  }
}

// TM
// --

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

// Net
// ----

__device__ Net vnet_new(GNet* gnet, void* smem, u32 turn) {
  Net net;
  net.l_node_dif   = 0;
  net.l_vars_dif   = 0;
  net.l_node_buf   = smem == NULL ? net.l_node_buf : ((LNet*)smem)->node_buf;
  net.l_vars_buf   = smem == NULL ? net.l_vars_buf : ((LNet*)smem)->vars_buf;
  net.g_rbag_use_A = turn % 2 == 0 ? &gnet->rbag_use_A : &gnet->rbag_use_B;
  net.g_rbag_use_B = turn % 2 == 0 ? &gnet->rbag_use_B : &gnet->rbag_use_A;
  net.g_rbag_buf_A = turn % 2 == 0 ? gnet->rbag_buf_A : gnet->rbag_buf_B;
  net.g_rbag_buf_B = turn % 2 == 0 ? gnet->rbag_buf_B : gnet->rbag_buf_A;
  net.g_node_buf   = gnet->node_buf;
  net.g_vars_buf   = gnet->vars_buf;
  net.g_node_put   = &gnet->node_put[GID()];
  net.g_vars_put   = &gnet->vars_put[GID()];
  return net;
}

// Stores a new node on global.
__device__ inline void node_create(Net* net, u32 loc, Pair val) {
  Pair old;
  if (loc < L_NODE_LEN) {
    net->l_node_dif += 1;
    old = atomicExch(&net->l_node_buf[loc], val);
  } else {
    old = atomicExch(&net->g_node_buf[loc], val);
  }
  #ifdef DEBUG
  if (old != 0) printf("[%04x] ERR NODE_CREATE | %04x\n", GID(), loc);
  #endif
}

// Stores a var on global.
__device__ inline void vars_create(Net* net, u32 var, Port val) {
  Port old;
  if (var < L_VARS_LEN) {
    net->l_vars_dif += 1;
    old = atomicExch(&net->l_vars_buf[var], val);
  } else {
    old = atomicExch(&net->g_vars_buf[var], val);
  }
  #ifdef DEBUG
  if (old != 0) printf("[%04x] ERR VARS_CREATE | %04x\n", GID(), var);
  #endif
}

// Reads a node from global.
__device__ __host__ inline Pair node_load(Net* net, u32 loc) {
  Pair got;
  if (loc < L_NODE_LEN) {
    got = net->l_node_buf[loc];
  } else {
    got = net->g_node_buf[loc];
  }
  return got;
}

// Reads a var from global.
__device__ __host__ inline Port vars_load(Net* net, u32 var) {
  Port got;
  if (var < L_VARS_LEN) {
    got = net->l_vars_buf[var];
  } else {
    got = net->g_vars_buf[var];
  }
  return got;
}

// Exchanges a node on global by a value. Returns old.
__device__ inline Pair node_exchange(Net* net, u32 loc, Pair val) {
  Pair got = 0;
  if (loc < L_NODE_LEN) {
    got = atomicExch(&net->l_node_buf[loc], val);
  } else {
    got = atomicExch(&net->g_node_buf[loc], val);
  }
  #ifdef DEBUG
  if (got == 0) printf("[%04x] ERR NODE_EXCHANGE | %04x\n", GID(), loc);
  #endif
  return got;
}

// Exchanges a var on global by a value. Returns old.
__device__ inline Port vars_exchange(Net* net, u32 var, Port val) {
  Port got = 0;
  if (var < L_VARS_LEN) {
    got = atomicExch(&net->l_vars_buf[var], val);
  } else {
    got = atomicExch(&net->g_vars_buf[var], val);
  }
  #ifdef DEBUG
  if (got == 0) printf("[%04x] ERR VARS_EXCHANGE | %04x\n", GID(), var);
  #endif
  return got;
}

// Takes a node.
__device__ inline Pair node_take(Net* net, u32 loc) {
  Pair got = 0;
  if (loc < L_NODE_LEN) {
    net->l_node_dif -= 1;
    got = atomicExch(&net->l_node_buf[loc], 0);
  } else {
    got = atomicExch(&net->g_node_buf[loc], 0);
  }
  #ifdef DEBUG
  if (got == 0) printf("[%04x] ERR NODE_TAKE | %04x\n", GID(), loc);
  #endif
  return got;
}

// Takes a var.
__device__ inline Port vars_take(Net* net, u32 var) {
  Port got = 0;
  if (var < L_VARS_LEN) {
    net->l_vars_dif -= 1;
    got = atomicExch(&net->l_vars_buf[var], 0);
  } else {
    got = atomicExch(&net->g_vars_buf[var], 0);
  }
  #ifdef DEBUG
  if (got == 0) printf("[%04x] ERR VARS_TAKE | %04x\n", GID(), var);
  #endif
  return got;
}

// Finds a variable's value.
__device__ inline Port peek(Net* net, Port var) {
  while (get_tag(var) == VAR) {
    Port val = vars_load(net, get_val(var));
    if (val == NONE) break;
    if (val == 0) break;
    var = val;
  }
  return var;
}

// Finds a variable's value.
__device__ inline Port enter(Net* net, Port var) {
  u32 lps = 0;
  Port init = var;
  // While `B` is VAR: extend it (as an optimization)
  while (get_tag(var) == VAR) {
    // Takes the current `var` substitution as `val`
    Port val = vars_exchange(net, get_val(var), NONE);
    // If there was no `val`, stop, as there is no extension
    if (val == NONE) {
      break;
    }
    // Sanity check: if global A is unfilled, stop
    if (val == 0) {
      break;
    }
    // Otherwise, delete `B` (we own both) and continue
    vars_take(net, get_val(var));
    //if (++lps > 65536) printf("[%04x] BUG A | init=%s var=%s val=%s\n", GID(), show_port(init).x, show_port(var).x, show_port(val).x);
    var = val;
  }
  return var;
}

#endif // structs_cuh_INCLUDED
