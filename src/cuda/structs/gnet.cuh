#ifndef structs_gnet_cuh_INCLUDED
#define structs_gnet_cuh_INCLUDED

#include "../structs.cuh"

// Initializes the GNet
__global__ void initialize(GNet* gnet) {
  gnet->node_put[GID()] = 0;
  gnet->vars_put[GID()] = 0;
  gnet->rbag_pos[GID()] = 0;
  for (u32 i = 0; i < RLEN; ++i) {
    gnet->rbag_buf_A[G_RBAG_LEN / TPG * GID() + i] = 0;
  }
  for (u32 i = 0; i < RLEN; ++i) {
    gnet->rbag_buf_B[G_RBAG_LEN / TPG * GID() + i] = 0;
  }
}

GNet* gnet_create() {
  GNet *gnet;
  cudaMalloc((void**)&gnet, sizeof(GNet));
  initialize<<<BPG, TPB>>>(gnet);
  //cudaMemset(gnet, 0, sizeof(GNet));
  return gnet;
}

__global__ void gnet_inbetween(GNet* gnet) {
  // Clears rbag use counter
  if (gnet->turn % 2 == 0) {
    gnet->rbag_use_A = 0;
  } else {
    gnet->rbag_use_B = 0;
  }

  // Increments gnet turn
  gnet->turn += 1;

  // Increments interaction counter
  gnet->itrs += gnet->iadd;

  // Resets the rdec variable
  gnet->rdec = 0;

  // Moves to next mode
  if (!gnet->down) {
    gnet->mode = min(gnet->mode + 1, WORK);
  }

  // If no work was done...
  if (gnet->iadd == 0) {
    // If on seed mode, go up to GROW mode
    if (gnet->mode == SEED) {
      gnet->mode = GROW;
      gnet->down = 0;
    // Otherwise, go down to SEED mode
    } else {
      gnet->mode = SEED;
      gnet->down = 1;
      gnet->rdec = 1; // peel one rpos
    }
    //printf(">> CHANGE MODE TO %d | %d <<\n", gnet->mode, gnet->down);
  }

  // Reset interaction adder
  gnet->iadd = 0;
}

u32 gnet_get_rlen(GNet* gnet, u32 turn) {
  u32 rbag_use;
  if (turn % 2 == 0) {
    cudaMemcpy(&rbag_use, &gnet->rbag_use_B, sizeof(u32), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(&rbag_use, &gnet->rbag_use_A, sizeof(u32), cudaMemcpyDeviceToHost);
  }
  return rbag_use;
}

u64 gnet_get_itrs(GNet* gnet) {
  u64 itrs;
  cudaMemcpy(&itrs, &gnet->itrs, sizeof(u64), cudaMemcpyDeviceToHost);
  return itrs;
}

u64 gnet_get_leak(GNet* gnet) {
  u64 leak;
  cudaMemcpy(&leak, &gnet->leak, sizeof(u64), cudaMemcpyDeviceToHost);
  return leak;
}

// Reads a device node to host
Pair gnet_node_load(GNet* gnet, u32 loc) {
  Pair pair;
  cudaMemcpy(&pair, &gnet->node_buf[loc], sizeof(Pair), cudaMemcpyDeviceToHost);
  return pair;
}

// Reads a device var to host
Port gnet_vars_load(GNet* gnet, u32 loc) {
  Pair port;
  cudaMemcpy(&port, &gnet->vars_buf[loc], sizeof(Port), cudaMemcpyDeviceToHost);
  return port;
}

// Writes a host var to device
void gnet_vars_create(GNet* gnet, u32 var, Port val) {
  cudaMemcpy(&gnet->vars_buf[var], &val, sizeof(Port), cudaMemcpyHostToDevice);
}

// Like the enter() function, but from host and read-only
Port gnet_peek(GNet* gnet, Port port) {
  while (get_tag(port) == VAR) {
    Port val = gnet_vars_load(gnet, get_val(port));
    if (val == NONE) break;
    port = val;
  }
  return port;
}

// Sets the initial redex.
__global__ void boot_redex(GNet* gnet, Pair redex) {
  // Creates root variable.
  gnet->vars_buf[get_val(ROOT)] = NONE;
  // Creates root redex.
  if (gnet->turn % 2 == 0) {
    gnet->rbag_buf_A[0] = redex;
  } else {
    gnet->rbag_buf_B[0] = redex;
  }
}

void gnet_boot_redex(GNet* gnet, Pair redex) {
  boot_redex<<<BPG, TPB>>>(gnet, redex);
}

#endif // structs_gnet_cuh_INCLUDED
