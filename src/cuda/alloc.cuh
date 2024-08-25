#ifndef alloc_cuh_INCLUDED
#define alloc_cuh_INCLUDED

#include "types.cuh"
#include "config.cuh"
#include "structs.cuh"
#include <stdio.h>

template <typename A>
__device__ u32 g_alloc_1(Net* net, u32* g_put, A* g_buf) {
  u32 lps = 0;
  while (true) {
    u32 lc = GID()*(G_NODE_LEN/TPG) + (*g_put%(G_NODE_LEN/TPG));
    A elem = g_buf[lc];
    *g_put += 1;
    if (lc >= L_NODE_LEN && elem == 0) {
      return lc;
    }
    if (++lps >= G_NODE_LEN/TPG) printf("OOM\n"); // FIXME: remove
    //assert(++lps < G_NODE_LEN/TPG); // FIXME: enable?
  }
}

template <typename A>
__device__ u32 g_alloc(Net* net, u32* ret, u32* g_put, A* g_buf, u32 num) {
  u32 got = 0;
  u32 lps = 0;
  while (got < num) {
    u32 lc = GID()*(G_NODE_LEN/TPG) + (*g_put%(G_NODE_LEN/TPG));
    A elem = g_buf[lc];
    *g_put += 1;
    if (lc >= L_NODE_LEN && elem == 0) {
      ret[got++] = lc;
    }

    if (++lps >= G_NODE_LEN/TPG) printf("OOM\n"); // FIXME: remove
    //assert(++lps < G_NODE_LEN/TPG); // FIXME: enable?
  }
  return got;

}

template <typename A>
__device__ u32 l_alloc(Net* net, u32* ret, u32* l_put, A* l_buf, u32 num) {
  u32 got = 0;
  u32 lps = 0;
  while (got < num) {
    u32 lc = ((*l_put)++ * TPB) % L_NODE_LEN + TID();
    A elem = l_buf[lc];
    if (++lps >= L_NODE_LEN/TPB) {
      break;
    }
    if (lc > 0 && elem == 0) {
      ret[got++] = lc;
    }
  }
  return got;
}

template <typename A>
__device__ u32 l_alloc_1(Net* net, u32* ret, u32* l_put, A* l_buf, u32* lps) {
  u32 got = 0;
  while (true) {
    u32 lc = ((*l_put)++ * TPB) % L_NODE_LEN + TID();
    A elem = l_buf[lc];
    if (++(*lps) >= L_NODE_LEN/TPB) {
      break;
    }
    if (lc > 0 && elem == 0) {
      return lc;
    }
  }
  return got;
}

__device__ u32 g_node_alloc_1(Net* net) {
  return g_alloc_1(net, net->g_node_put, net->g_node_buf);
}

__device__ u32 g_vars_alloc_1(Net* net) {
  return g_alloc_1(net, net->g_vars_put, net->g_vars_buf);
}

__device__ u32 g_node_alloc(Net* net, TM* tm, u32 num) {
  return g_alloc(net, tm->nloc, net->g_node_put, net->g_node_buf, num);
}

__device__ u32 g_vars_alloc(Net* net, TM* tm, u32 num) {
  return g_alloc(net, tm->vloc, net->g_vars_put, net->g_vars_buf, num);
}

__device__ u32 l_node_alloc(Net* net, TM* tm, u32 num) {
  return l_alloc(net, tm->nloc, &tm->nput, net->l_node_buf, num);
}

__device__ u32 l_vars_alloc(Net* net, TM* tm, u32 num) {
  return l_alloc(net, tm->vloc, &tm->vput, net->l_vars_buf, num);
}

__device__ u32 l_node_alloc_1(Net* net, TM* tm, u32* lps) {
  return l_alloc_1(net, tm->nloc, &tm->nput, net->l_node_buf, lps);
}

__device__ u32 l_vars_alloc_1(Net* net, TM* tm, u32* lps) {
  return l_alloc_1(net, tm->vloc, &tm->vput, net->l_vars_buf, lps);
}

__device__ u32 node_alloc_1(Net* net, TM* tm, u32* lps) {
  if (tm->mode != WORK) {
    return g_node_alloc_1(net);
  } else {
    return l_node_alloc_1(net, tm, lps);
  }
}

__device__ u32 vars_alloc_1(Net* net, TM* tm, u32* lps) {
  if (tm->mode != WORK) {
    return g_vars_alloc_1(net);
  } else {
    return l_vars_alloc_1(net, tm, lps);
  }
}

// Adjusts a newly allocated port.
__device__ inline Port adjust_port(Net* net, TM* tm, Port port) {
  Tag tag = get_tag(port);
  Val val = get_val(port);
  if (is_nod(port)) return new_port(tag, tm->nloc[val]);
  if (is_var(port)) return new_port(tag, tm->vloc[val]);
  return new_port(tag, val);
}

// Adjusts a newly allocated pair.
__device__ inline Pair adjust_pair(Net* net, TM* tm, Pair pair) {
  Port p1 = adjust_port(net, tm, get_fst(pair));
  Port p2 = adjust_port(net, tm, get_snd(pair));
  return new_pair(p1, p2);
}

// Gets the necessary resources for an interaction.
__device__ bool get_resources(Net* net, TM* tm, u32 need_rbag, u32 need_node, u32 need_vars) {
  u32 got_rbag = min(RLEN - tm->rbag.lo_end, RLEN - tm->rbag.hi_end);
  u32 got_node;
  u32 got_vars;
  if (tm->mode != WORK) {
    got_node = g_node_alloc(net, tm, need_node);
    got_vars = g_vars_alloc(net, tm, need_vars);
  } else {
    got_node = l_node_alloc(net, tm, need_node);
    got_vars = l_vars_alloc(net, tm, need_vars);
  }
  return got_rbag >= need_rbag && got_node >= need_node && got_vars >= need_vars;
}

#endif // alloc_cuh_INCLUDED
