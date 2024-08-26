#ifndef structs_vnet_cuh_INCLUDED
#define structs_vnet_cuh_INCLUDED

#include "../structs.cuh"

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

#endif // structs_vnet_cuh_INCLUDED
