#ifndef show_cuh_INCLUDED
#define show_cuh_INCLUDED

#include "port/numb.cuh"
#include "pair.cuh"
#include "structs.cuh"
#include <string.h>
#include <stdio.h>

struct Show {
  char x[13];
};

__device__ __host__ Show show_port(Port port);
__device__ __host__ Show show_rule(Rule rule);

__device__ __host__ void print_net(Net* net, u32 start, u32 end);
__device__ void print_rbag(Net* net, TM* tm);
__device__ void pretty_print_rbag(Net* net, RBag* rbag);
__device__ void pretty_print_port(Net* net, Port port);
__device__ void pretty_print_numb(Numb word);

__global__ void print_heatmap(GNet* gnet, u32 turn);

__device__ __host__ void put_u32(char* B, u32 val) {
  for (int i = 0; i < 8; i++, val >>= 4) {
    B[8-i-1] = "0123456789ABCDEF"[val & 0xF];
  }
}

__device__ __host__ Show show_port(Port port) {
  // NOTE: this is done like that because sprintf seems not to be working
  Show s;
  switch (get_tag(port)) {
    case VAR: memcpy(s.x, "VAR:", 4); put_u32(s.x+4, get_val(port)); break;
    case REF: memcpy(s.x, "REF:", 4); put_u32(s.x+4, get_val(port)); break;
    case ERA: memcpy(s.x, "ERA:________", 12); break;
    case NUM: memcpy(s.x, "NUM:", 4); put_u32(s.x+4, get_val(port)); break;
    case CON: memcpy(s.x, "CON:", 4); put_u32(s.x+4, get_val(port)); break;
    case DUP: memcpy(s.x, "DUP:", 4); put_u32(s.x+4, get_val(port)); break;
    case OPR: memcpy(s.x, "OPR:", 4); put_u32(s.x+4, get_val(port)); break;
    case SWI: memcpy(s.x, "SWI:", 4); put_u32(s.x+4, get_val(port)); break;
  }
  s.x[12] = '\0';
  return s;
}

__device__ Show show_rule(Rule rule) {
  Show s;
  switch (rule) {
    case LINK: memcpy(s.x, "LINK", 4); break;
    case VOID: memcpy(s.x, "VOID", 4); break;
    case ERAS: memcpy(s.x, "ERAS", 4); break;
    case ANNI: memcpy(s.x, "ANNI", 4); break;
    case COMM: memcpy(s.x, "COMM", 4); break;
    case OPER: memcpy(s.x, "OPER", 4); break;
    case SWIT: memcpy(s.x, "SWIT", 4); break;
    case CALL: memcpy(s.x, "CALL", 4); break;
    default  : memcpy(s.x, "????", 4); break;
  }
  s.x[4] = '\0';
  return s;
}

__device__ void print_rbag(Net* net, TM* tm) {
  printf("RBAG | FST-TREE     | SND-TREE    \n");
  printf("---- | ------------ | ------------\n");
  for (u32 i = 0; i < tm->rbag.hi_end; ++i) {
    Pair redex = tm->rbag.hi_buf[i];
    Pair node1 = get_tag(get_snd(redex)) == CON ? node_load(net, get_val(get_fst(redex))) : 0;
    Pair node2 = get_tag(get_snd(redex)) == CON ? node_load(net, get_val(get_snd(redex))) : 0;
    printf("%04X | %s | %s | hi | (%s %s) ~ (%s %s)\n", i,
      show_port(get_fst(redex)).x,
      show_port(get_snd(redex)).x,
      show_port(peek(net, get_fst(node1))).x,
      show_port(peek(net, get_snd(node1))).x,
      show_port(peek(net, get_fst(node2))).x,
      show_port(peek(net, get_snd(node2))).x);
  }
  for (u32 i = 0; i < tm->rbag.lo_end; ++i) {
    Pair redex = tm->rbag.lo_buf[i%RLEN];
    Pair node1 = get_tag(get_snd(redex)) == CON ? node_load(net, get_val(get_fst(redex))) : 0;
    Pair node2 = get_tag(get_snd(redex)) == CON ? node_load(net, get_val(get_snd(redex))) : 0;
    printf("%04X | %s | %s | hi | (%s %s) ~ (%s %s)\n", i,
      show_port(get_fst(redex)).x,
      show_port(get_snd(redex)).x,
      show_port(peek(net, get_fst(node1))).x,
      show_port(peek(net, get_snd(node1))).x,
      show_port(peek(net, get_fst(node2))).x,
      show_port(peek(net, get_snd(node2))).x);
  }
  printf("==== | ============ | ============\n");
}

__device__ __host__ void print_net(Net* net, u32 ini, u32 end) {
  printf("NODE | PORT-1       | PORT-2      \n");
  printf("---- | ------------ | ------------\n");
  for (u32 i = ini; i < end; ++i) {
    Pair node = node_load(net, i);
    if (node != 0) {
      printf("%04X | %s | %s\n", i, show_port(get_fst(node)).x, show_port(get_snd(node)).x);
    }
  }
  printf("==== | ============ |\n");
  printf("VARS | VALUE        |\n");
  printf("---- | ------------ |\n");
  for (u32 i = ini; i < end; ++i) {
    Port var = vars_load(net,i);
    if (var != 0) {
      printf("%04X | %s |\n", i, show_port(vars_load(net,i)).x);
    }
  }
  printf("==== | ============ |\n");
}

__device__ void pretty_print_numb(Numb word) {
  switch (get_typ(word)) {
    case TY_SYM: {
      switch (get_sym(word)) {
        // types
        case TY_U24: printf("[u24]"); break;
        case TY_I24: printf("[i24]"); break;
        case TY_F24: printf("[f24]"); break;
        // operations
        case OP_ADD: printf("[+]"); break;
        case OP_SUB: printf("[-]"); break;
        case FP_SUB: printf("[:-]"); break;
        case OP_MUL: printf("[*]"); break;
        case OP_DIV: printf("[/]"); break;
        case FP_DIV: printf("[:/]"); break;
        case OP_REM: printf("[%%]"); break;
        case FP_REM: printf("[:%%]"); break;
        case OP_EQ:  printf("[=]"); break;
        case OP_NEQ: printf("[!]"); break;
        case OP_LT:  printf("[<]"); break;
        case OP_GT:  printf("[>]"); break;
        case OP_AND: printf("[&]"); break;
        case OP_OR:  printf("[|]"); break;
        case OP_XOR: printf("[^]"); break;
        case OP_SHL: printf("[<<]"); break;
        case FP_SHL: printf("[:<<]"); break;
        case OP_SHR: printf("[>>]"); break;
        case FP_SHR: printf("[:>>]"); break;
        default:     printf("[?]"); break;
      }
      break;
    }
    case TY_U24: {
      printf("%u", get_u24(word));
      break;
    }
    case TY_I24: {
      printf("%+d", get_i24(word));
      break;
    }
    case TY_F24: {
      if (isinf(get_f24(word))) {
        if (signbit(get_f24(word))) {
          printf("-inf");
        } else {
          printf("+inf");
        }
      } else if (isnan(get_f24(word))) {
        printf("+NaN");
      } else {
        printf("%.7e", get_f24(word));
      }
      break;
    }
    default: {
      switch (get_typ(word)) {
        case OP_ADD: printf("[+0x%07X]", get_u24(word)); break;
        case OP_SUB: printf("[-0x%07X]", get_u24(word)); break;
        case FP_SUB: printf("[:-0x%07X]", get_u24(word)); break;
        case OP_MUL: printf("[*0x%07X]", get_u24(word)); break;
        case OP_DIV: printf("[/0x%07X]", get_u24(word)); break;
        case FP_DIV: printf("[:/0x%07X]", get_u24(word)); break;
        case OP_REM: printf("[%%0x%07X]", get_u24(word)); break;
        case FP_REM: printf("[:%%0x%07X]", get_u24(word)); break;
        case OP_EQ:  printf("[=0x%07X]", get_u24(word)); break;
        case OP_NEQ: printf("[!0x%07X]", get_u24(word)); break;
        case OP_LT:  printf("[<0x%07X]", get_u24(word)); break;
        case OP_GT:  printf("[>0x%07X]", get_u24(word)); break;
        case OP_AND: printf("[&0x%07X]", get_u24(word)); break;
        case OP_OR:  printf("[|0x%07X]", get_u24(word)); break;
        case OP_XOR: printf("[^0x%07X]", get_u24(word)); break;
        case OP_SHL: printf("[<<0x%07X]", get_u24(word)); break;
        case FP_SHL: printf("[:<<0x%07X]", get_u24(word)); break;
        case OP_SHR: printf("[>>0x%07X]", get_u24(word)); break;
        case FP_SHR: printf("[:>>0x%07X]", get_u24(word)); break;
        default:     printf("[?0x%07X]", get_u24(word)); break;
      }
      break;
    }
  }
}

__device__ void pretty_print_port(Net* net, Port port) {
  Port stack[4096];
  stack[0] = port;
  u32 len = 1;
  while (len > 0) {
    if (len > 256) {
      printf("ERROR: result too deep to print. This will be fixed soon(TM)");
      --len;
      continue;
    }
    Port cur = stack[--len];
    switch (get_tag(cur)) {
      case CON: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("(");
        stack[len++] = new_port(ERA, (u32)(')'));
        stack[len++] = p2;
        stack[len++] = new_port(ERA, (u32)(' '));
        stack[len++] = p1;
        break;
      }
      case ERA: {
        if (get_val(cur) != 0) {
          printf("%c", (char)get_val(cur));
        } else {
          printf("*");
        }
        break;
      }
      case VAR: {
        Port got = vars_load(net, get_val(cur));
        if (got != NONE) {
          stack[len++] = got;
        } else {
          printf("x%x", get_val(cur));
        }
        break;
      }
      case NUM: {
        pretty_print_numb(get_val(cur));
        break;
      }
      case DUP: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("{");
        stack[len++] = new_port(ERA, (u32)('}'));
        stack[len++] = p2;
        stack[len++] = new_port(ERA, (u32)(' '));
        stack[len++] = p1;
        break;
      }
      case OPR: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("$(");
        stack[len++] = new_port(ERA, (u32)(')'));
        stack[len++] = p2;
        stack[len++] = new_port(ERA, (u32)(' '));
        stack[len++] = p1;
        break;
      }
      case SWI: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("?(");
        stack[len++] = new_port(ERA, (u32)(')'));
        stack[len++] = p2;
        stack[len++] = new_port(ERA, (u32)(' '));
        stack[len++] = p1;
        break;
      }
      case REF: {
        u32  fid = get_val(cur) & 0xFFFFFFF;
        Def* def = &BOOK.defs_buf[fid];
        printf("@%s", def->name);
        break;
      }
    }
  }
}

__device__ void pretty_print_rbag(Net* net, RBag* rbag) {
  for (u32 i = 0; i < rbag->lo_end; ++i) {
    Pair redex = rbag->lo_buf[i%RLEN];
    if (redex != 0) {
      pretty_print_port(net, get_fst(redex));
      printf(" ~ ");
      pretty_print_port(net, get_snd(redex));
      printf("\n");
    }
  }
  for (u32 i = 0; i < rbag->hi_end; ++i) {
    Pair redex = rbag->hi_buf[i];
    if (redex != 0) {
      pretty_print_port(net, get_fst(redex));
      printf(" ~ ");
      pretty_print_port(net, get_snd(redex));
      printf("\n");
    }
  }
}

__global__ void print_heatmap(GNet* gnet, u32 turn) {
  if (GID() > 0) return;

  const char* heatChars[] = {
    //" ", ".", ":", ":",
    //"∴", "⁘", "⁙", "░",
    //"░", "░", "▒", "▒",
    //"▒", "▓", "▓", "▓"
    " ", "1", "2", "3",
    "4", "5", "6", "7",
    "8", "9", "A", "B",
    "C", "D", "E", "F",
  };

  for (u32 bid = 0; bid < BPG; bid++) {
    printf("|");
    for (u32 tid = 0; tid < TPB; tid++) {
      u32 gid = bid * TPB + tid;
      u32 len = 0;
      for (u32 i = 0; i < RLEN; i++) {
        if ( turn % 2 == 0 && gnet->rbag_buf_A[gid * RLEN + i] != 0
          || turn % 2 == 1 && gnet->rbag_buf_B[gid * RLEN + i] != 0) {
          len++;
        }
      }
      u32 pos = gnet->rbag_pos[gid];
      u32 heat = min(len, 0xF);
      printf("%s", heatChars[heat]);
    }
    printf("|\n");
  }
}

__global__ void print_result(GNet* gnet) {
  Net net = vnet_new(gnet, NULL, gnet->turn);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Result: ");
    pretty_print_port(&net, enter(&net, ROOT));
    printf("\n");
  }
}

#endif // show_cuh_INCLUDED
