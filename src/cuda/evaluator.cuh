#ifndef evaluator_cuh_INCLUDED
#define evaluator_cuh_INCLUDED

#include "evaluator/interactions.cuh"
#include "structs/gnet.cuh"
#include "structs/tm.cuh"
#include "sync.cuh"

// Transposes an index over a matrix.
__device__ u32 transpose(u32 idx, u32 width, u32 height) {
  u32 old_row = idx / width;
  u32 old_col = idx % width;
  u32 new_row = old_col % height;
  u32 new_col = old_col / height + old_row * (width / height);
  return new_row * width + new_col;
}


// Moves redexes from shared memory to global bag
__device__ void save_redexes(Net* net, TM *tm, u32 turn) {
  u32 idx = 0;
  u32 bag = tm->mode == SEED ? transpose(GID(), TPB, BPG) : GID();

  // Leaks low-priority redexes
  for (u32 i = 0; i < tm->rbag.lo_end; ++i) {
    Pair R = tm->rbag.lo_buf[i % RLEN];
    Port x = get_fst(R);
    Port y = get_snd(R);
    Port X = new_port(VAR, g_vars_alloc_1(net));
    Port Y = new_port(VAR, g_vars_alloc_1(net));
    vars_create(net, get_val(X), NONE);
    vars_create(net, get_val(Y), NONE);
    link_pair(net, tm, new_pair(X, x));
    link_pair(net, tm, new_pair(Y, y));
    net->g_rbag_buf_B[bag * RLEN + (idx++)] = new_pair(X, Y);
  }
  __syncthreads();
  tm->rbag.lo_end = 0;

  // Executes all high-priority redexes
  while (rbag_has_highs(&tm->rbag)) {
    Pair redex = pop_redex(tm);
    if (!interact(net, tm, redex, turn)) {
      printf("ERROR: failed to clear high-priority redexes");
    }
  }
  __syncthreads();

  #ifdef DEBUG
  if (rbag_len(&tm->rbag) > 0) printf("[%04x] ERR SAVE_REDEXES lo=%d hi=%d tot=%d\n", GID(), tm->rbag.lo_end, tm->rbag.hi_end, rbag_len(&tm->rbag));
  #endif

  // Updates global redex counter
  atomicAdd(net->g_rbag_use_B, idx);
}

// Loads redexes from global bag to shared memory
// FIXME: check if we have enuogh space for all loads
__device__ void load_redexes(Net* net, TM *tm, u32 turn) {
  u32 gid = BID() * TPB + TID();
  u32 bag = tm->mode == SEED ? transpose(GID(), TPB, BPG) : GID();
  for (u32 i = 0; i < RLEN; ++i) {
    Pair redex = atomicExch(&net->g_rbag_buf_A[bag * RLEN + i], 0);
    if (redex != 0) {
      Port a = enter(net, get_fst(redex));
      Port b = enter(net, get_snd(redex));
      #ifdef DEBUG
      if (is_local(a) || is_local(b)) printf("[%04x] ERR LOAD_REDEXES\n", turn);
      #endif
      push_redex(tm, new_pair(a, b));
    } else {
      break;
    }
  }
  __syncthreads();
}

// Kernels
// -------

// EVAL
__global__ void evaluator(GNet* gnet) {
  extern __shared__ char shared_mem[]; // 96 KB
  __shared__ Pair spawn[TPB]; // thread initialized

  // Thread Memory
  TM tm = tmem_new();

  // Net (Local-Global View)
  Net net = vnet_new(gnet, shared_mem, gnet->turn);

  // Clears shared memory
  for (u32 i = 0; i < L_NODE_LEN / TPB; ++i) {
    net.l_node_buf[i * TPB + TID()] = 0;
    net.l_vars_buf[i * TPB + TID()] = 0;
  }
  __syncthreads();

  // Sets mode
  tm.mode = gnet->mode;

  // Loads Redexes
  load_redexes(&net, &tm, gnet->turn);

  // Clears spawn buffer
  spawn[TID()] = rbag_len(&tm.rbag) > 0 ? 0xFFFFFFFFFFFFFFFF : 0;
  __syncthreads();

  // Variables
  u64 INIT = clock64(); // initial time
  u32 HASR = block_count(rbag_len(&tm.rbag) > 0);
  u32 tick = 0;
  u32 bag  = tm.mode == SEED ? transpose(GID(), TPB, BPG) : GID();
  u32 rpos = gnet->rbag_pos[bag] > 0 ? gnet->rbag_pos[bag] - gnet->rdec : gnet->rbag_pos[bag];
  u8  down = gnet->down;

  //if (BID() == 0 && gnet->turn == 0x69) {
    //printf("[%04x] ini rpos is %d | bag=%d\n", GID(), rpos, bag);
  //}

  // Aborts if empty
  if (HASR == 0) {
    return;
  }

  //if (BID() == 0 && rbag_len(&tm.rbag) > 0) {
    //Pair redex = pop_redex(&tm);
    //Pair kn = get_tag(get_snd(redex)) == CON ? node_load(&net, get_val(get_snd(redex))) : 0;
    //printf("[%04x] HAS REDEX %s ~ %s | par? %d | (%s %s)\n",
      //GID(),
      //show_port(get_fst(redex)).x,
      //show_port(get_snd(redex)).x,
      //get_par_flag(redex),
      //show_port(get_fst(kn)).x,
      //show_port(get_snd(kn)).x);
    //push_redex(&tm, redex);
  //}

  //// Display debug rbag
  //if (GID() == 0) {
    //print_rbag(&net, &tm);
    //printf("| rbag_pos = %d | mode = %d | down = %d | turn = %04x\n", gnet->rbag_pos[bag], gnet->mode, down, gnet->turn);
  //}
  //__syncthreads();

  // GROW MODE
  // ---------

  if (tm.mode == SEED || tm.mode == GROW) {
    u32 tlim = tm.mode == SEED ? min(TPB_L2,BPG_L2) : max(TPB_L2,BPG_L2);
    u32 span = 1 << (32 - __clz(TID()));

    Pair redex;

    for (u32 tick = 0; tick < tlim; ++tick) {
      u32 span = 1 << tick;
      u32 targ = TID() ^ span;

      // Attempts to spawn a thread
      if (TID() < span && spawn[targ] == 0) {
        //if (BID() == 0) {
          //if (!TID()) printf("----------------------------------------------------\n");
          //if (!TID()) printf("TIC %04x | span=%d | rlen=%d | ", tick, span, rbag_len(&tm.rbag));
          //block_print(rbag_len(&tm.rbag));
          //if (!TID()) printf("\n");
          //__syncthreads();
        //}

        // Performs some interactions until a parallel redex is found
        for (u32 i = 0; i < 64; ++i) {
          if (tm.rbag.lo_end < rpos) break;
          redex = pop_redex(&tm);
          if (redex == 0) {
            break;
          }
          // If we found a stealable redex, pass it to stealing,
          // and un-mark the redex above it, so we keep it for us.
          if (get_par_flag(redex)) {
            Pair above = pop_redex(&tm);
            if (above != 0) {
              push_redex(&tm, clr_par_flag(above));
            }
            break;
          }
          interact(&net, &tm, redex, gnet->turn);
          redex = 0;
          while (tm.rbag.hi_end > 0) {
            if (!interact(&net, &tm, pop_redex(&tm), gnet->turn)) break;
          }
        }

        // Spawn a thread
        if (redex != 0 && get_par_flag(redex)) {
          //if (BID() == 0) {
            //Pair kn = get_tag(get_snd(redex)) == CON ? node_load(&net, get_val(get_snd(redex))) : 0;
            //printf("[%04x] GIVE %s ~ %s | par? %d | (%s %s) | rbag.lo_end=%d\n", GID(), show_port(get_fst(redex)).x, show_port(get_snd(redex)).x, get_par_flag(redex), show_port(peek(&net, &tm, get_fst(kn))).x, show_port(peek(&net, &tm, get_snd(kn))).x, tm.rbag.lo_end);
          //}

          spawn[targ] = clr_par_flag(redex);
          if (!down) {
            rpos = tm.rbag.lo_end - 1;
          }
        }
      }
      __syncthreads();

      // If we've been spawned, push initial redex
      if (TID() >= span && TID() < span*2 && spawn[TID()] != 0 && spawn[TID()] != 0xFFFFFFFFFFFFFFFF) {
        //if (rbag_len(&tm.rbag) > 0) {
          //printf("[%04x] ERROR: SPAWNED BUT HAVE REDEX\n", GID());
        //}

        push_redex(&tm, atomicExch(&spawn[TID()], 0xFFFFFFFFFFFFFFFF));
        rpos = 0;
        //if (BID() == 0) printf("[%04x] TAKE %016llx\n", GID(), spawn[TID()]);
      }
      __syncthreads();

      //if (BID() == 0) {
        //if (!TID()) printf("TAC %04x | span=%d | rlen=%d | ", tick, span, rbag_len(&tm.rbag));
        //block_print(rbag_len(&tm.rbag));
        //if (!TID()) printf("\n");
        //__syncthreads();
      //}
      //__syncthreads();

      //printf("[%04x] span is %d\n", TID(), span);
      //__syncthreads();
    }

    //if (BID() == 0 && gnet->turn == 0x69) {
      //printf("[%04x] end rpos is %d | bag=%d\n", GID(), rpos, bag);
    //}

    gnet->rbag_pos[bag] = rpos;

  }

  // WORK MODE
  // ---------

  if (tm.mode == WORK) {
    u32 chkt = 0;
    u32 chka = 1;
    u32 bag  = tm.mode == SEED ? transpose(GID(), TPB, BPG) : GID();
    u32 rpos = gnet->rbag_pos[bag];
    for (tick = 0; tick < 1 << 9; ++tick) {
      if (tm.rbag.lo_end > rpos || rbag_has_highs(&tm.rbag)) {
        if (interact(&net, &tm, pop_redex(&tm), gnet->turn)) {
          while (rbag_has_highs(&tm.rbag)) {
            if (!interact(&net, &tm, pop_redex(&tm), gnet->turn)) break;
          }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();

  //u32 ITRS = block_sum(tm.itrs);
  //u32 LOOP = block_sum((u32)tick);
  //u32 RLEN = block_sum(rbag_len(&tm.rbag));
  //u32 FAIL = 0; // block_sum((u32)fail);
  //f64 TIME = (f64)(clock64() - INIT) / (f64)S;
  //f64 MIPS = (f64)ITRS / TIME / (f64)1000000.0;
  ////if (BID() >= 0 && TID() == 0) {
  //if (TID() == 0) {
    //printf("%04x:[%02x]: MODE=%d DOWN=%d ITRS=%d LOOP=%d RLEN=%d FAIL=%d TIME=%f MIPS=%.0f | %d\n",
      //gnet->turn, BID(), tm.mode, down, ITRS, LOOP, RLEN, FAIL, TIME, MIPS, 42);
  //}
  //__syncthreads();

  // Display debug rbag
  //if (BID() == 0) {
    //for (u32 i = 0; i < TPB; ++i) {
      //if (TID() == i && rbag_len(&tm.rbag) > 0) print_rbag(&net, &tm);
      //__syncthreads();
    //}
    //__syncthreads();
  //}

  // Moves rbag to global
  save_redexes(&net, &tm, gnet->turn);

  // Stores rewrites
  atomicAdd(&gnet->iadd, tm.itrs);
  atomicAdd(&gnet->leak, tm.leak);

}

// GNet Host Functions
// -------------------

void gnet_normalize(GNet* gnet) {
  // Invokes the Evaluator Kernel repeatedly
  u32 turn;
  u64 itrs = 0;
  u32 rlen = 0;
  // NORM
  for (turn = 0; turn < 0xFFFFFFFF; ++turn) {
    //printf("\e[1;1H\e[2J");
    //printf("==================================================== ");
    //printf("TURN: %04x | RLEN: %04x | ITRS: %012llu\n", turn, rlen, itrs);
    //cudaDeviceSynchronize();

    evaluator<<<BPG, TPB, sizeof(LNet)>>>(gnet);
    gnet_inbetween<<<1, 1>>>(gnet);
    //cudaDeviceSynchronize();

    //count_memory<<<BPG, TPB>>>(gnet);
    //cudaDeviceSynchronize();

    //print_heatmap<<<1,1>>>(gnet, turn+1);
    //cudaDeviceSynchronize();

    itrs = gnet_get_itrs(gnet);
    rlen = gnet_get_rlen(gnet, turn);
    if (rlen == 0) {
      //printf("Completed after %d kernel launches!\n", turn);
      break;
    }
  }
}

// Expands a REF Port.
Port gnet_expand(GNet* gnet, Port port) {
  Port old = gnet_vars_load(gnet, get_val(ROOT));
  Port got = gnet_peek(gnet, port);
  //printf("expand %s\n", show_port(got).x);
  while (get_tag(got) == REF) {
    gnet_boot_redex(gnet, new_pair(got, ROOT));
    gnet_normalize(gnet);
    got = gnet_peek(gnet, gnet_vars_load(gnet, get_val(ROOT)));
  }
  gnet_vars_create(gnet, get_val(ROOT), old);
  return got;
}

#endif // evaluator_cuh_INCLUDED
