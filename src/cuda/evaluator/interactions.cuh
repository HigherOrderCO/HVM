#ifndef evaluator_interactions_cuh_INCLUDED
#define evaluator_interactions_cuh_INCLUDED

#include "alloc.cuh"
#include "numb.cuh"
#include "structs/rbag.cuh"
#include "structs/gnet.cuh"

// Linking
// -------

// Atomically Links `A ~ B`.
__device__ void link(Net* net, TM* tm, Port A, Port B) {
  #ifdef DEBUG
  Port INI_A = A;
  Port INI_B = B;
  #endif

  u32 lps = 0;

  // Attempts to directionally point `A ~> B`
  while (true) {

    // If `A` is NODE: swap `A` and `B`, and continue
    if (get_tag(A) != VAR && get_tag(B) == VAR) {
      Port X = A; A = B; B = X;
    }

    // If `A` is NODE: create the `A ~ B` redex
    if (get_tag(A) != VAR) {
      //printf("[%04x] new redex A %s ~ %s\n", GID(), show_port(A).x, show_port(B).x);
      push_redex(tm, new_pair(A, B)); // TODO: move global ports to local
      break;
    }

    // While `B` is VAR: extend it (as an optimization)
    B = enter(net, B);

    // Since `A` is VAR: point `A ~> B`.
    if (true) {
      // If B would leak...
      if (is_global(A) && is_local(B)) {
        // If B is a var, just swap it
        if (is_var(B)) {
          Port X = A; A = B; B = X;
          continue;
        }
        // If B is a nod, create a leak interaction
        if (is_nod(B)) {
          //if (!TID()) printf("[%04x] NODE LEAK %s ~ %s\n", GID(), show_port(A).x, show_port(B).x);
          push_redex(tm, new_pair(A, B));
          break;
        }
      }

      // Sanity check: if global A is unfilled, delay this link
      if (is_global(A) && vars_load(net, get_val(A)) == 0) {
        push_redex(tm, new_pair(A, B));
        break;
      }

      // Stores `A -> B`, taking the current `A` subst as `A'`
      Port A_ = vars_exchange(net, get_val(A), B);

      // If there was no `A'`, stop, as we lost B's ownership
      if (A_ == NONE) {
        break;
      }

      #ifdef DEBUG
      if (A_ == 0) printf("[%04x] ERR LINK %s ~ %s | %s ~ %s\n", GID(), show_port(INI_A).x, show_port(INI_B).x, show_port(A).x, show_port(B).x);
      #endif

      // Otherwise, delete `A` (we own both) and link `A' ~ B`
      vars_take(net, get_val(A));
      A = A_;
    }
  }
}

// Links `A ~ B` (as a pair).
__device__ void link_pair(Net* net, TM* tm, Pair AB) {
  link(net, tm, get_fst(AB), get_snd(AB));
}

// Interactions
// ------------

// The Link Interaction.
__device__ bool interact_link(Net* net, TM* tm, Port a, Port b) {
  // If A is a global var and B is a local node, leak it:
  // ^A ~ (b1 b2)
  // ------------- LEAK-NODE
  // ^X ~ b1
  // ^Y ~ b2
  // ^A ~ ^(^X ^Y)
  if (is_global(a) && is_nod(b) && is_local(b)) {
    // Allocates needed nodes and vars.
    if (!get_resources(net, tm, 3, 0, 0)) {
      return false;
    }

    tm->leak += 1;

    // Loads ports.
    Pair l_b  = node_take(net, get_val(b));
    Port l_b1 = enter(net, get_fst(l_b));
    Port l_b2 = enter(net, get_snd(l_b));

    // Leaks port 1.
    Port g_b1;
    if (is_local(l_b1)) {
      g_b1 = new_port(VAR, g_vars_alloc_1(net));
      vars_create(net, get_val(g_b1), NONE);
      link_pair(net, tm, new_pair(g_b1, l_b1));
    } else {
      g_b1 = l_b1;
    }

    // Leaks port 2.
    Port g_b2;
    if (is_local(l_b2)) {
      g_b2 = new_port(VAR, g_vars_alloc_1(net));
      vars_create(net, get_val(g_b2), NONE);
      link_pair(net, tm, new_pair(g_b2, l_b2));
    } else {
      g_b2 = l_b2;
    }

    // Leaks node.
    Port g_b = new_port(get_tag(b), g_node_alloc_1(net));
    node_create(net, get_val(g_b), new_pair(g_b1, g_b2));
    link_pair(net, tm, new_pair(a, g_b));

    return true;

  // Otherwise, just perform a normal link.
  } else {
    // Allocates needed nodes and vars.
    if (!get_resources(net, tm, 1, 0, 0)) {
      return false;
    }

    link_pair(net, tm, new_pair(a, b));
  }

  return true;
}

// The Void Interaction.
__device__ bool interact_void(Net* net, TM* tm, Port a, Port b) {
  return true;
}

// The Eras Interaction.
__device__ bool interact_eras(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return false;
  }

  // Loads ports.
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  // Links.
  link_pair(net, tm, new_pair(a, B1));
  link_pair(net, tm, new_pair(a, B2));

  return true;
}

// The Call Interaction. In COMPILED mode, this interaction is
// provided elsewhere.
#ifdef COMPILED
__device__ bool interact_call(Net* net, TM* tm, Port a, Port b);
#else
__device__ bool interact_call(Net* net, TM* tm, Port a, Port b) {
  // Loads Definition.
  u32 fid  = get_val(a) & 0xFFFFFFF;
  Def* def = &BOOK.defs_buf[fid];

  // Copy Optimization.
  if (def->safe && get_tag(b) == DUP) {
    return interact_eras(net, tm, a, b);
  }

  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, def->rbag_len + 1, def->node_len, def->vars_len)) {
    return false;
  }

  // Stores new vars.
  for (u32 i = 0; i < def->vars_len; ++i) {
    vars_create(net, tm->vloc[i], NONE);
  }

  // Stores new nodes.
  for (u32 i = 0; i < def->node_len; ++i) {
    node_create(net, tm->nloc[i], adjust_pair(net, tm, def->node_buf[i]));
  }

  // Links.
  for (u32 i = 0; i < def->rbag_len; ++i) {
    link_pair(net, tm, adjust_pair(net, tm, def->rbag_buf[i]));
  }
  link_pair(net, tm, new_pair(adjust_port(net, tm, def->root), b));

  return true;
}
#endif

// The Anni Interaction.
__device__ bool interact_anni(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  // Links.
  link_pair(net, tm, new_pair(A1, B1));
  link_pair(net, tm, new_pair(A2, B2));

  return true;
}

// The Comm Interaction.
__device__ bool interact_comm(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 4, 4, 4)) {
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  // Stores new vars.
  vars_create(net, tm->vloc[0], NONE);
  vars_create(net, tm->vloc[1], NONE);
  vars_create(net, tm->vloc[2], NONE);
  vars_create(net, tm->vloc[3], NONE);

  // Stores new nodes.
  node_create(net, tm->nloc[0], new_pair(new_port(VAR, tm->vloc[0]), new_port(VAR, tm->vloc[1])));
  node_create(net, tm->nloc[1], new_pair(new_port(VAR, tm->vloc[2]), new_port(VAR, tm->vloc[3])));
  node_create(net, tm->nloc[2], new_pair(new_port(VAR, tm->vloc[0]), new_port(VAR, tm->vloc[2])));
  node_create(net, tm->nloc[3], new_pair(new_port(VAR, tm->vloc[1]), new_port(VAR, tm->vloc[3])));

  // Links.
  link_pair(net, tm, new_pair(new_port(get_tag(b), tm->nloc[0]), A1));
  link_pair(net, tm, new_pair(new_port(get_tag(b), tm->nloc[1]), A2));
  link_pair(net, tm, new_pair(new_port(get_tag(a), tm->nloc[2]), B1));
  link_pair(net, tm, new_pair(new_port(get_tag(a), tm->nloc[3]), B2));

  return true;
}

// The Oper Interaction.
__device__ bool interact_oper(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 1, 0)) {
    return false;
  }

  // Loads ports.
  Val  av = get_val(a);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = enter(net, get_snd(B));

  // Performs operation.
  if (get_tag(B1) == NUM) {
    Val  bv = get_val(B1);
    Numb cv = operate(av, bv);
    link_pair(net, tm, new_pair(new_port(NUM, cv), B2));
  } else {
    node_create(net, tm->nloc[0], new_pair(a, B2));
    link_pair(net, tm, new_pair(B1, new_port(OPR, tm->nloc[0])));
  }

  return true;
}

// The Swit Interaction.
__device__ bool interact_swit(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 2, 0)) {
    return false;
  }

  // Loads ports.
  u32  av = get_u24(get_val(a));
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  // Stores new nodes.
  if (av == 0) {
    node_create(net, tm->nloc[0], new_pair(B2, new_port(ERA,0)));
    link_pair(net, tm, new_pair(new_port(CON, tm->nloc[0]), B1));
  } else {
    node_create(net, tm->nloc[0], new_pair(new_port(ERA,0), new_port(CON, tm->nloc[1])));
    node_create(net, tm->nloc[1], new_pair(new_port(NUM, new_u24(av-1)), B2));
    link_pair(net, tm, new_pair(new_port(CON, tm->nloc[0]), B1));
  }

  return true;
}

// Pops a local redex and performs a single interaction.
__device__ bool interact(Net* net, TM* tm, Pair redex, u32 turn) {
  // Gets redex ports A and B.
  Port a = get_fst(redex);
  Port b = get_snd(redex);

  // Gets the rule type.
  Rule rule = get_rule(a, b);

  // If there is no redex, stop.
  if (redex != 0) {
    //if (GID() == 0 && turn == 0x201) {
      //Pair kn = get_tag(b) == CON ? node_load(net, get_val(b)) : 0;
      //printf("%04x:[%04x] REDUCE %s ~ %s | par? %d | (%s %s)\n",
        //turn, GID(),
        //show_port(get_fst(redex)).x,
        //show_port(get_snd(redex)).x,
        //get_par_flag(redex),
        //show_port(get_fst(kn)).x,
        //show_port(get_snd(kn)).x);
    //}

    // Used for root redex.
    if (get_tag(a) == REF && b == ROOT) {
      rule = CALL;
    // Swaps ports if necessary.
    } else if (should_swap(a,b)) {
      swap(&a, &b);
    }

    // Dispatches interaction rule.
    bool success;
    switch (rule) {
      case LINK: success = interact_link(net, tm, a, b); break;
      case CALL: success = interact_call(net, tm, a, b); break;
      case VOID: success = interact_void(net, tm, a, b); break;
      case ERAS: success = interact_eras(net, tm, a, b); break;
      case ANNI: success = interact_anni(net, tm, a, b); break;
      case COMM: success = interact_comm(net, tm, a, b); break;
      case OPER: success = interact_oper(net, tm, a, b); break;
      case SWIT: success = interact_swit(net, tm, a, b); break;
    }

    // If error, pushes redex back.
    if (!success) {
      push_redex(tm, redex);
      return false;
    // Else, increments the interaction count.
    } else if (rule != LINK) {
      tm->itrs += 1;
    }
  }

  return true;
}

#endif // evaluator_interactions_cuh_INCLUDED
