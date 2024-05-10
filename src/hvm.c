#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INTERPRETED

// Integers
// --------

typedef uint8_t bool;

typedef  uint8_t  u8;
typedef uint16_t u16;
typedef  int32_t i32;
typedef uint32_t u32;
typedef uint64_t u64;

typedef _Atomic(u8) a8;
typedef _Atomic(u16) a16;
typedef _Atomic(u32) a32;
typedef _Atomic(u64) a64;

// Configuration
// -------------

// Threads per CPU
#define TPC_L2  4
#define TPC    (1 << TPC_L2)

// Program
const u32 DEPTH = 10;
const u32 LOOPS = 65536;

// Types
// -----

// Local Types
typedef u8  Tag;  // Tag  ::= 3-bit (rounded up to u8)
typedef u32 Val;  // Val  ::= 29-bit (rounded up to u32)
typedef u32 Port; // Port ::= Tag + Val (fits a u32)
typedef u64 Pair; // Pair ::= Port + Port (fits a u64)

typedef a32 APort; // atomic Port
typedef a64 APair; // atomic Pair

// Rules
typedef u8 Rule; // Rule ::= 3-bit (rounded up to 8)

// Numbs
typedef u32 Numb; // Numb ::= 29-bit (rounded up to u32)

// Tags
const Tag VAR = 0x0; // variable
const Tag REF = 0x1; // reference
const Tag ERA = 0x2; // eraser
const Tag NUM = 0x3; // number
const Tag CON = 0x4; // constructor
const Tag DUP = 0x5; // duplicator
const Tag OPR = 0x6; // operator
const Tag SWI = 0x7; // switch

// Interaction Rule Values
const Rule LINK = 0x0;
const Rule CALL = 0x1;
const Rule VOID = 0x2;
const Rule ERAS = 0x3;
const Rule ANNI = 0x4;
const Rule COMM = 0x5;
const Rule OPER = 0x6;
const Rule SWIT = 0x7;

// Numbers
const Tag SYM = 0x0;
const Tag U24 = 0x1;
const Tag I24 = 0x2;
const Tag F24 = 0x3;
const Tag ADD = 0x4;
const Tag SUB = 0x5;
const Tag MUL = 0x6;
const Tag DIV = 0x7;
const Tag REM = 0x8;
const Tag EQ  = 0x9;
const Tag NEQ = 0xA;
const Tag LT  = 0xB;
const Tag GT  = 0xC;
const Tag AND = 0xD;
const Tag OR  = 0xE;
const Tag XOR = 0xF;

// Constants
const Port FREE = 0x00000000;
const Port ROOT = 0xFFFFFFF8;
const Port NONE = 0xFFFFFFFF;

// Cache Padding
const u32 CACHE_PAD = 64;

// Global Net
#define RLEN (1 << 24) // max 16m redexes
#define G_NODE_LEN (1 << 29) // max 536m nodes
#define G_VARS_LEN (1 << 29) // max 536m vars
#define G_RBAG_LEN (TPC * RLEN)

typedef struct Net {
  APair node_buf[G_NODE_LEN]; // global node buffer
  APort vars_buf[G_VARS_LEN]; // global vars buffer
  APair rbag_buf[G_RBAG_LEN]; // global rbag buffer
  a64 itrs; // interaction count
  a32 idle; // idle thread counter
} Net;

// Top-Level Definition
typedef struct Def {
  char name[32];
  bool safe;
  u32  rbag_len;
  u32  node_len;
  u32  vars_len;
  Port root;
  Pair rbag_buf[32];
  Pair node_buf[32];
} Def;

// Book of Definitions
typedef struct Book {
  u32 defs_len;
  Def defs_buf[256];
} Book;

// Local Thread Memory
typedef struct TM {
  u32  tid; // thread id
  u32  itrs; // interaction count
  u32  nput; // next node allocation attempt index
  u32  vput; // next vars allocation attempt index
  u32  rput; // next rbag push index
  u32  sidx; // steal index
  u32  nloc[32]; // node allocation indices
  u32  vloc[32]; // vars allocation indices
} TM;

// Booleans
const bool true  = 1;
const bool false = 0;

// Debugger
// --------

typedef struct {
  char x[13];
} Show;

void put_u16(char* B, u16 val);
Show show_port(Port port);
Show show_rule(Rule rule);
//void print_rbag(RBag* rbag);
void print_net(Net* net);
void pretty_print_port(Net* net, Port port);
//void pretty_print_rbag(Net* net, RBag* rbag);

// Port: Constructor and Getters
// -----------------------------

static inline Port new_port(Tag tag, Val val) {
  return (val << 3) | tag;
}

static inline Tag get_tag(Port port) {
  return port & 7;
}

static inline Val get_val(Port port) {
  return port >> 3;
}

// Pair: Constructor and Getters
// -----------------------------

static inline const Pair new_pair(Port fst, Port snd) {
  return ((u64)snd << 32) | fst;
}

static inline Port get_fst(Pair pair) {
  return pair & 0xFFFFFFFF;
}

static inline Port get_snd(Pair pair) {
  return pair >> 32;
}

// Utils
// -----

// Swaps two ports.
static inline void swap(Port *a, Port *b) {
  Port x = *a; *a = *b; *b = x;
}

u32 min(u32 a, u32 b) {
  return (a < b) ? a : b;
}

// A simple spin-wait barrier using atomic operations
a64 a_reached = 0; // number of threads that reached the current barrier
a64 a_barrier = 0; // number of barriers passed during this program

void sync_threads() {
  u64 barrier_old = atomic_load_explicit(&a_barrier, memory_order_relaxed);
  if (atomic_fetch_add_explicit(&a_reached, 1, memory_order_relaxed) == (TPC - 1)) {
    // Last thread to reach the barrier resets the counter and advances the barrier
    atomic_store_explicit(&a_reached, 0, memory_order_relaxed);
    atomic_store_explicit(&a_barrier, barrier_old + 1, memory_order_release);
  } else {
    u32 tries = 0;
    while (atomic_load_explicit(&a_barrier, memory_order_acquire) == barrier_old) {
      sched_yield();
    }
  }
}

// Global sum function
static a32 GLOBAL_SUM = 0;
u32 global_sum(u32 x) {
  atomic_fetch_add_explicit(&GLOBAL_SUM, x, memory_order_relaxed);
  sync_threads();
  u32 sum = atomic_load_explicit(&GLOBAL_SUM, memory_order_relaxed);
  sync_threads();
  atomic_store_explicit(&GLOBAL_SUM, 0, memory_order_relaxed);
  return sum;
}

// TODO: write a time64() function that returns the time as fast as possible as a u64
static inline u64 time64() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

// Ports / Pairs / Rules
// ---------------------

// True if this port has a pointer to a node.
static inline bool is_nod(Port a) {
  return get_tag(a) >= CON;
}

// True if this port is a variable.
static inline bool is_var(Port a) {
  return get_tag(a) == VAR;
}

// Given two tags, gets their interaction rule.
static inline Rule get_rule(Port a, Port b) {
  const u8 table[8][8] = {
    //VAR  REF  ERA  NUM  CON  DUP  OPR  SWI
    {LINK,LINK,LINK,LINK,LINK,LINK,LINK,LINK}, // VAR
    {LINK,VOID,VOID,VOID,CALL,CALL,CALL,CALL}, // REF
    {LINK,VOID,VOID,VOID,ERAS,ERAS,ERAS,ERAS}, // ERA
    {LINK,VOID,VOID,VOID,ERAS,ERAS,OPER,SWIT}, // NUM
    {LINK,CALL,ERAS,ERAS,ANNI,COMM,COMM,COMM}, // CON
    {LINK,CALL,ERAS,ERAS,COMM,ANNI,COMM,COMM}, // DUP
    {LINK,CALL,ERAS,OPER,COMM,COMM,ANNI,COMM}, // OPR
    {LINK,CALL,ERAS,SWIT,COMM,COMM,COMM,ANNI}, // SWI
  };
  return table[get_tag(a)][get_tag(b)];
}

// Same as above, but receiving a pair.
static inline Rule get_pair_rule(Pair AB) {
  return get_rule(get_fst(AB), get_snd(AB));
}

// Should we swap ports A and B before reducing this rule?
static inline bool should_swap(Port A, Port B) {
  return get_tag(B) < get_tag(A);
}

// Gets a rule's priority
static inline bool is_high_priority(Rule rule) {
  return (bool)((0b00011101 >> rule) & 1);
}

// Adjusts a newly allocated port.
static inline Port adjust_port(Net* net, TM* tm, Port port) {
  Tag tag = get_tag(port);
  Val val = get_val(port);
  if (is_nod(port)) return new_port(tag, tm->nloc[val]);
  if (is_var(port)) return new_port(tag, tm->vloc[val]);
  return new_port(tag, val);
}

// Adjusts a newly allocated pair.
static inline Pair adjust_pair(Net* net, TM* tm, Pair pair) {
  Port p1 = adjust_port(net, tm, get_fst(pair));
  Port p2 = adjust_port(net, tm, get_snd(pair));
  return new_pair(p1, p2);
}

// Numbs
// -----

// Constructor and getters for SYM (operation selector)
static inline Numb new_sym(u32 val) {
  return ((val & 0xF) << 4) | SYM;
}

static inline u32 get_sym(Numb word) {
  return (word >> 4) & 0xF;
}

// Constructor and getters for U24 (unsigned 24-bit integer)
static inline Numb new_u24(u32 val) {
  return ((val & 0xFFFFFF) << 4) | U24;
}

static inline u32 get_u24(Numb word) {
  return (word >> 4) & 0xFFFFFF;
}

// Constructor and getters for I24 (signed 24-bit integer)
static inline Numb new_i24(i32 val) {
  return (((u32)val << 4) & 0xFFFFFF) | I24;
}

static inline i32 get_i24(Numb word) {
  return (((word >> 4) & 0xFFFFFF) << 8) >> 8;
}

// Constructor and getters for F24 (24-bit float)
static inline Numb new_f24(float val) {
  u32 bits = *(u32*)&val;
  u32 sign = (bits >> 31) & 0x1;
  i32 expo = ((bits >> 23) & 0xFF) - 127;
  u32 mant = bits & 0x7FFFFF;
  u32 uexp = expo + 63;
  u32 bts1 = (sign << 23) | (uexp << 16) | (mant >> 7);
  return (bts1 << 4) | F24;
}

static inline float get_f24(Numb word) {
  u32 bits = (word >> 4) & 0xFFFFFF;
  u32 sign = (bits >> 23) & 0x1;
  u32 expo = (bits >> 16) & 0x7F;
  u32 mant = bits & 0xFFFF;
  i32 iexp = expo - 63;
  u32 bts0 = (sign << 31) | ((iexp + 127) << 23) | (mant << 7);
  u32 bts1 = (mant == 0 && iexp == -63) ? (sign << 31) : bts0;
  return *(float*)&bts1;
}

// Flip flag
static inline Tag get_typ(Numb word) {
  return word & 0xF;
}

static inline bool get_flp(Numb word) {
  return ((word >> 28) & 1) == 1;
}

static inline Numb set_flp(Numb word) {
  return word | 0x10000000;
}

static inline Numb flp_flp(Numb word) {
  return word ^ 0x10000000;
}

// Partial application
static inline Numb partial(Numb a, Numb b) {
  return b & 0xFFFFFFF0 | get_sym(a);
}

// Operate function
static inline Numb operate(Numb a, Numb b) {
  if (get_flp(a) ^ get_flp(b)) {
    Numb t = a; a = b; b = t;
  }
  Tag at = get_typ(a);
  Tag bt = get_typ(b);
  if (at == SYM && bt == SYM) {
    return new_u24(0);
  }
  if (at == SYM && bt != SYM) {
    return partial(a, b);
  }
  if (at != SYM && bt == SYM) {
    return partial(b, a);
  }
  if (at >= ADD && bt >= ADD) {
    return new_u24(0);
  }
  if (at < ADD && bt < ADD) {
    return new_u24(0);
  }
  Tag op = (at >= ADD) ? at : bt;
  Tag ty = (at >= ADD) ? bt : at;
  switch (ty) {
    case U24: {
      u32 av = get_u24(a);
      u32 bv = get_u24(b);
      switch (op) {
        case ADD: return new_u24(av + bv);
        case SUB: return new_u24(av - bv);
        case MUL: return new_u24(av * bv);
        case DIV: return new_u24(av / bv);
        case REM: return new_u24(av % bv);
        case EQ:  return new_u24(av == bv);
        case NEQ: return new_u24(av != bv);
        case LT:  return new_u24(av < bv);
        case GT:  return new_u24(av > bv);
        case AND: return new_u24(av & bv);
        case OR:  return new_u24(av | bv);
        case XOR: return new_u24(av ^ bv);
        default:  return new_u24(0);
      }
    }
    case I24: {
      i32 av = get_i24(a);
      i32 bv = get_i24(b);
      switch (op) {
        case ADD: return new_i24(av + bv);
        case SUB: return new_i24(av - bv);
        case MUL: return new_i24(av * bv);
        case DIV: return new_i24(av / bv);
        case REM: return new_i24(av % bv);
        case EQ:  return new_i24(av == bv);
        case NEQ: return new_i24(av != bv);
        case LT:  return new_i24(av < bv);
        case GT:  return new_i24(av > bv);
        case AND: return new_i24(av & bv);
        case OR:  return new_i24(av | bv);
        case XOR: return new_i24(av ^ bv);
        default:  return new_i24(0);
      }
    }
    case F24: {
      float av = get_f24(a);
      float bv = get_f24(b);
      switch (op) {
        case ADD: return new_f24(av + bv);
        case SUB: return new_f24(av - bv);
        case MUL: return new_f24(av * bv);
        case DIV: return new_f24(av / bv);
        case REM: return new_f24(fmodf(av, bv));
        case EQ:  return new_u24(av == bv);
        case NEQ: return new_u24(av != bv);
        case LT:  return new_u24(av < bv);
        case GT:  return new_u24(av > bv);
        case AND: return new_f24(atan2f(av, bv));
        case OR:  return new_f24(logf(bv) / logf(av));
        case XOR: return new_f24(powf(av, bv));
        default:  return new_f24(0);
      }
    }
    default: return new_u24(0);
  }
}

// RBag
// ----

static inline void push_redex(Net* net, TM* tm, Pair redex) {
  atomic_store_explicit(&net->rbag_buf[tm->tid*(G_RBAG_LEN/TPC) + (tm->rput++)], redex, memory_order_relaxed);
}

static inline Pair pop_redex(Net* net, TM* tm) {
  if (tm->rput > 0) {
    return atomic_exchange_explicit(&net->rbag_buf[tm->tid*(G_RBAG_LEN/TPC) + (--tm->rput)], 0, memory_order_relaxed);
  } else {
    return 0;
  }
}

static inline u32 rbag_len(Net* net, TM* tm) {
  return tm->rput;
}

// TM
// --

void tmem_init(TM* tm, u32 tid) {
  tm->tid  = tid;
  tm->itrs = 0;
  tm->nput = 1;
  tm->vput = 1;
  tm->rput = 0;
  tm->sidx = 0;
}

// Net
// ----

// Stores a new node on global.
static inline void node_create(Net* net, u32 loc, Pair val) {
  atomic_store_explicit(&net->node_buf[loc], val, memory_order_relaxed);
}

// Stores a var on global. Returns old.
static inline void vars_create(Net* net, u32 var, Port val) {
  atomic_store_explicit(&net->vars_buf[var], val, memory_order_relaxed);
}

// Reads a node from global.
static inline Pair node_load(Net* net, u32 loc) {
  return atomic_load_explicit(&net->node_buf[loc], memory_order_relaxed);
}

// Reads a var from global.
static inline Port vars_load(Net* net, u32 var) {
  return atomic_load_explicit(&net->vars_buf[var], memory_order_relaxed);
}

// Stores a node on global.
static inline void node_store(Net* net, u32 loc, Pair val) {
  atomic_store_explicit(&net->node_buf[loc], val, memory_order_relaxed);
}

// Stores a var on global. Returns old.
static inline void vars_store(Net* net, u32 var, Port val) {
  atomic_store_explicit(&net->vars_buf[var], val, memory_order_relaxed);
}

// Exchanges a node on global by a value. Returns old.
static inline Pair node_exchange(Net* net, u32 loc, Pair val) {
  return atomic_exchange_explicit(&net->node_buf[loc], val, memory_order_relaxed);
}

// Exchanges a var on global by a value. Returns old.
static inline Port vars_exchange(Net* net, u32 var, Port val) {
  return atomic_exchange_explicit(&net->vars_buf[var], val, memory_order_relaxed);
}

// Takes a node.
static inline Pair node_take(Net* net, u32 loc) {
  return node_exchange(net, loc, 0);
}

// Takes a var.
static inline Port vars_take(Net* net, u32 var) {
  return vars_exchange(net, var, 0);
}


// Net
// ---

// Initializes a net.
static inline void net_init(Net* net) {
  // Initializes the root var.
  vars_create(net, get_val(ROOT), NONE);
  // Initializes variables
  atomic_store(&net->itrs, 0);
  atomic_store(&net->idle, 0);
}

// Allocator
// ---------

u32 node_alloc_1(Net* net, TM* tm, u32* lps) {
  while (true) {
    u32 lc = tm->tid*(G_NODE_LEN/TPC) + (tm->nput%(G_NODE_LEN/TPC));
    Pair elem = net->node_buf[lc];
    tm->nput += 1;
    if (lc > 0 && elem == 0) {
      return lc;
    }
    // FIXME: check this decently
    if (++(*lps) >= G_NODE_LEN/TPC) printf("OOM\n");
  }
}

u32 vars_alloc_1(Net* net, TM* tm, u32* lps) {
  while (true) {
    u32 lc = tm->tid*(G_NODE_LEN/TPC) + (tm->vput%(G_NODE_LEN/TPC));
    Port elem = net->vars_buf[lc];
    tm->vput += 1;
    if (lc > 0 && elem == 0) {
      return lc;
    }
    // FIXME: check this decently
    if (++(*lps) >= G_NODE_LEN/TPC) printf("OOM\n");
  }
}

u32 node_alloc(Net* net, TM* tm, u32 num) {
  u32 got = 0;
  u32 lps = 0;
  while (got < num) {
    u32 lc = tm->tid*(G_NODE_LEN/TPC) + (tm->nput%(G_NODE_LEN/TPC));
    Pair elem = net->node_buf[lc];
    tm->nput += 1;
    if (lc > 0 && elem == 0) {
      tm->nloc[got++] = lc;
    }
    // FIXME: check this decently
    if (++lps >= G_NODE_LEN/TPC) printf("OOM\n");
  }
  return got;
}

u32 vars_alloc(Net* net, TM* tm, u32 num) {
  u32 got = 0;
  u32 lps = 0;
  while (got < num) {
    u32 lc = tm->tid*(G_NODE_LEN/TPC) + (tm->vput%(G_NODE_LEN/TPC));
    Port elem = net->vars_buf[lc];
    tm->vput += 1;
    if (lc > 0 && elem == 0) {
      tm->vloc[got++] = lc;
    }
    // FIXME: check this decently
    if (++lps >= G_NODE_LEN/TPC) printf("OOM\n");
  }
  return got;
}

// Gets the necessary resources for an interaction. Returns success.
static inline bool get_resources(Net* net, TM* tm, u8 need_rbag, u8 need_node, u8 need_vars) {
  u32 got_rbag = 0xFF; // FIXME: implement
  u32 got_node = node_alloc(net, tm, need_node);
  u32 got_vars = vars_alloc(net, tm, need_vars);
  return got_rbag >= need_rbag && got_node >= need_node && got_vars >= need_vars;
}

// Linking
// -------

// Finds a variable's value.
static inline Port enter(Net* net, TM* tm, Port var) {
  // While `B` is VAR: extend it (as an optimization)
  while (get_tag(var) == VAR) {
    // Takes the current `var` substitution as `val`
    Port val = vars_exchange(net, get_val(var), NONE);
    // If there was no `val`, stop, as there is no extension
    if (val == NONE || val == 0) {
      break;
    }
    // Otherwise, delete `B` (we own both) and continue
    vars_take(net, get_val(var));
    var = val;
  }
  return var;
}

// Atomically Links `A ~ B`.
static inline void link(Net* net, TM* tm, Port A, Port B) {
  //printf("LINK %s ~> %s\n", show_port(A).x, show_port(B).x);

  // Attempts to directionally point `A ~> B`
  while (true) {
    // If `A` is NODE: swap `A` and `B`, and continue
    if (get_tag(A) != VAR && get_tag(B) == VAR) {
      Port X = A; A = B; B = X;
    }
    
    // If `A` is NODE: create the `A ~ B` redex
    if (get_tag(A) != VAR) {
      push_redex(net, tm, new_pair(A, B)); // TODO: move global ports to local
      break;
    }

    // Extends B (as an optimization)
    B = enter(net, tm, B);

    // Since `A` is VAR: point `A ~> B`.
    if (true) {
      // Stores `A -> B`, taking the current `A` subst as `A'`
      Port A_ = vars_exchange(net, get_val(A), B);
      // If there was no `A'`, stop, as we lost B's ownership
      if (A_ == NONE) {
        break;
      }
      //if (A_ == 0) { ? } // FIXME: must handle on the move-to-global algo
      // Otherwise, delete `A` (we own both) and link `A' ~ B`
      vars_take(net, get_val(A));
      A = A_;
    }
  }
}

// Links `A ~ B` (as a pair).
static inline void link_pair(Net* net, TM* tm, Pair AB) {
  //printf("link_pair %016llx\n", AB);
  link(net, tm, get_fst(AB), get_snd(AB));
}

// Sharing
// -------

// Sends redex to a friend local thread, when it is starving.
//void share_redexes(TM* tm, APair* share, u32 tid) {
  //Pair send = new_pair(NONE, NONE);
  //Pair recv = new_pair(NONE, NONE);
  //u32*  ini = &tm->rbag.lo_ini;
  //u32*  end = &tm->rbag.lo_end;
  //Pair* bag = tm->rbag.lo_buf;
  //for (u32 i = 0; i < TPC_L2; ++i) {
    //u32 a = tm->tid;
    //u32 b = a ^ (1 << i);
    //recv = new_pair(NONE, NONE);
    //send = (*end - *ini) > 1 ? bag[*ini%RLEN] : 0;
    //atomic_exchange_explicit(&share[a*CACHE_PAD], send, memory_order_relaxed);
    //while (recv == new_pair(NONE, NONE)) {
      //recv = atomic_exchange_explicit(&share[b*CACHE_PAD], new_pair(NONE, NONE), memory_order_relaxed);
    //}
    //if (!send &&  recv) bag[((*end)++)%RLEN] = recv;
    //if ( send && !recv) ++(*ini);
    //sync_threads();
  //}
//}

// Interactions
// ------------

// The Link Interaction.
static inline bool interact_link(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 0, 0)) {
    return false;
  }
  
  // Links.
  link_pair(net, tm, new_pair(a, b));

  return true;
}

// The Call Interaction.
#ifdef COMPILED
///COMPILED_INTERACT_CALL///
#else
static inline bool interact_eras(Net* net, TM* tm, Port a, Port b);
static inline bool interact_call(Net* net, TM* tm, Port a, Port b, Book* book) {
  // Loads Definition.
  u32  fid = get_val(a) & 0xFFFFFFF;
  Def* def = &book->defs_buf[fid];

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
    //printf("vars_create vloc[%04x] %04x\n", i, tm->vloc[i]);
  }

  // Stores new nodes.
  for (u32 i = 0; i < def->node_len; ++i) {
    node_create(net, tm->nloc[i], adjust_pair(net, tm, def->node_buf[i]));
    //printf("node_create nloc[%04x] %08llx\n", i-1, def->node_buf[i]);
  }

  // Links.
  for (u32 i = 0; i < def->rbag_len; ++i) {
    link_pair(net, tm, adjust_pair(net, tm, def->rbag_buf[i]));
  }
  link_pair(net, tm, new_pair(adjust_port(net, tm, def->root), b));

  return true;
}
#endif

// The Void Interaction.
static inline bool interact_void(Net* net, TM* tm, Port a, Port b) {
  return true;
}

// The Eras Interaction.
static inline bool interact_eras(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return false;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable0: %s\n", tid, show_port(b).x);
    return false;
  }

  // Loads ports.
  Pair B  = node_exchange(net, get_val(b), 0);
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);
  
  //if (B == 0) printf("[%04x] ERROR2: %s\n", tid, show_port(b).x);

  // Links.
  link_pair(net, tm, new_pair(a, B1));
  link_pair(net, tm, new_pair(a, B2));

  return true;
}

// The Anni Interaction.
static inline bool interact_anni(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    //printf("AAA\n");
    return false;
  }

  // Checks availability
  if (node_load(net, get_val(a)) == 0 || node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable1: %s | %s\n", tid, show_port(a).x, show_port(b).x);
    //printf("BBB\n");
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);
      
  //if (A == 0) printf("[%04x] ERROR3: %s\n", tid, show_port(a).x);
  //if (B == 0) printf("[%04x] ERROR4: %s\n", tid, show_port(b).x);

  // Links.
  link_pair(net, tm, new_pair(A1, B1));
  link_pair(net, tm, new_pair(A2, B2));

  return true;
}

// The Comm Interaction.
static inline bool interact_comm(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 4, 4, 4)) {
    return false;
  }

  // Checks availability
  if (node_load(net, get_val(a)) == 0 || node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable2: %s | %s\n", tid, show_port(a).x, show_port(b).x);
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  //if (A == 0) printf("[%04x] ERROR5: %s\n", tid, show_port(a).x);
  //if (B == 0) printf("[%04x] ERROR6: %s\n", tid, show_port(b).x);

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
static inline bool interact_oper(Net* net, TM* tm, Port a, Port b) {
  //printf("OPER %08x %08x\n", a, b);

  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 1, 0)) {
    return false;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
    return false;
  }

  // Loads ports.
  Val  av = get_val(a);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = enter(net, tm, get_snd(B));
     
  // Performs operation.
  if (get_tag(B1) == NUM) {
    Val  bv = get_val(B1);
    Numb cv = operate(av, bv);
    link_pair(net, tm, new_pair(new_port(NUM, cv), B2));
  } else {
    node_create(net, tm->nloc[0], new_pair(new_port(get_tag(a), flp_flp(av)), B2));
    link_pair(net, tm, new_pair(B1, new_port(OPR, tm->nloc[0])));
  }

  return true;
}

// The Swit Interaction.
static inline bool interact_swit(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 2, 0)) {
    return false;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
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
static inline bool interact(Net* net, TM* tm, Book* book) {
  // Pops a redex.
  Pair redex = pop_redex(net, tm);

  // If there is no redex, stop.
  if (redex != 0) {
    // Gets redex ports A and B.
    Port a = get_fst(redex);
    Port b = get_snd(redex);

    // Gets the rule type.
    Rule rule = get_rule(a, b);

    // Used for root redex.
    if (get_tag(a) == REF && b == ROOT) {
      rule = CALL;
    // Swaps ports if necessary.
    } else if (should_swap(a,b)) {
      swap(&a, &b);
    }

    //printf("[%04x] REDUCE %s ~ %s | %s\n", tm->tid, show_port(a).x, show_port(b).x, show_rule(rule).x);

    // Dispatches interaction rule.
    bool success;
    switch (rule) {
      case LINK: success = interact_link(net, tm, a, b); break;
      #ifdef COMPILED
      case CALL: success = interact_call(net, tm, a, b); break;
      #else
      case CALL: success = interact_call(net, tm, a, b, book); break;
      #endif
      case VOID: success = interact_void(net, tm, a, b); break;
      case ERAS: success = interact_eras(net, tm, a, b); break;
      case ANNI: success = interact_anni(net, tm, a, b); break;
      case COMM: success = interact_comm(net, tm, a, b); break;
      case OPER: success = interact_oper(net, tm, a, b); break;
      case SWIT: success = interact_swit(net, tm, a, b); break;
    }

    // If error, pushes redex back.
    if (!success) {
      push_redex(net, tm, redex);
      return false;
    // Else, increments the interaction count.
    } else if (rule != LINK) {
      tm->itrs += 1;
    }
  }

  return true;
}

// Evaluator
// ---------

//void evaluator(Net* net, TM* tm, Book* book) {
  //u32 turn = 0;
  //while (rbag_len(&tm->rbag) > 0 && ++turn < 0x10000000) {
    //interact(net, tm, book);
    //while (rbag_has_highs(&tm->rbag)) {
      //if (!interact(net, tm, book)) break;
    //}
  //}
  //atomic_fetch_add(&net->itrs, tm->itrs);
  //tm->itrs = 0;
//}

void evaluator(Net* net, TM* tm, Book* book) {
  // Initializes the global idle counter
  atomic_store_explicit(&net->idle, TPC - 1, memory_order_relaxed);
  sync_threads();

  // Performs some interactions
  u32  tick = 0;
  bool busy = tm->tid == 0;
  while (true) {
    tick += 1;
    // If we have redexes...
    if (rbag_len(net, tm) > 0) {
      // Update global idle counter
      if (!busy) atomic_fetch_sub_explicit(&net->idle, 1, memory_order_relaxed);
      busy = true;
      // Perform an interaction
      interact(net, tm, book);
    // If we have no redexes...
    } else {
      // Update global idle counter
      if (busy) atomic_fetch_add_explicit(&net->idle, 1, memory_order_relaxed);
      busy = false;
      // Attempt to steal a redex
      u32  sid = (tm->tid - 1) % TPC;
      u32  idx = sid*(G_RBAG_LEN/TPC) + (tm->sidx++);
      Pair got = atomic_exchange_explicit(&net->rbag_buf[idx], 0, memory_order_relaxed);
      if (got != 0) {
        push_redex(net, tm, got);
        //printf("[%04x] stolen one task from %04x | itrs=%d idle=%d | %s ~ %s\n", tm->tid, sid, tm->itrs, atomic_load_explicit(&net->idle, memory_order_relaxed),show_port(get_fst(got)).x, show_port(get_snd(got)).x);
      } else {
        //printf("[%04x] failed to steal from %04x | itrs=%d idle=%d |\n", tm->tid, sid, tm->itrs, atomic_load_explicit(&net->idle, memory_order_relaxed));
        tm->sidx = 0;
      }
      // Chill...
      sched_yield();
      // Halt if all threads are idle
      if (tick % 256 == 0) {
        if (atomic_load_explicit(&net->idle, memory_order_relaxed) == TPC) {
          break;
        }
      }
    }
  }

  sync_threads();

  atomic_fetch_add(&net->itrs, tm->itrs);
  tm->itrs = 0;
}

void normalize(Net* net, TM* tm, Book* book) {
}

// Book Loader
// -----------

void book_load(u32* buf, Book* book) {
  // Reads defs_len
  book->defs_len = *buf++;

  // Parses each def
  for (u32 i = 0; i < book->defs_len; ++i) {
    // Reads fid
    u32 fid = *buf++;

    // Gets def
    Def* def = &book->defs_buf[fid];
    
    // Reads name
    memcpy(def->name, buf, 32);
    buf += 8;

    // Reads safe flag
    def->safe = *buf++;

    // Reads lengths
    def->rbag_len = *buf++;
    def->node_len = *buf++;
    def->vars_len = *buf++;

    // Reads root
    def->root = *buf++;

    // Reads rbag_buf
    memcpy(def->rbag_buf, buf, 8*def->rbag_len);
    buf += def->rbag_len * 2;
    
    // Reads node_buf
    memcpy(def->node_buf, buf, 8*def->node_len);
    buf += def->node_len * 2;
  }
}

// Debug Printing
// --------------

void put_u32(char* B, u32 val) {
  for (int i = 0; i < 8; i++, val >>= 4) {
    B[8-i-1] = "0123456789ABCDEF"[val & 0xF];
  }  
}

Show show_port(Port port) {
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

Show show_rule(Rule rule) {
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

//void print_rbag(RBag* rbag) {
  //printf("RBAG | FST-TREE     | SND-TREE    \n");
  //printf("---- | ------------ | ------------\n");
  //for (u32 i = rbag->lo_ini; i < rbag->lo_end; ++i) {
    //Pair redex = rbag->lo_buf[i%RLEN];
    //printf("%04X | %s | %s\n", i, show_port(get_fst(redex)).x, show_port(get_snd(redex)).x);
  //}
  //for (u32 i = 0; i > rbag->hi_end; ++i) {
    //Pair redex = rbag->hi_buf[i];
    //printf("%04X | %s | %s\n", i, show_port(get_fst(redex)).x, show_port(get_snd(redex)).x);
  //}
  //printf("==== | ============ | ============\n");
//}

void print_net(Net* net) {
  printf("NODE | PORT-1       | PORT-2      \n");
  printf("---- | ------------ | ------------\n");
  for (u32 i = 0; i < G_NODE_LEN; ++i) {
    Pair node = node_load(net, i);
    if (node != 0) {
      printf("%04X | %s | %s\n", i, show_port(get_fst(node)).x, show_port(get_snd(node)).x);
    }
  }
  printf("==== | ============ |\n");
  printf("VARS | VALUE        |\n");
  printf("---- | ------------ |\n");
  for (u32 i = 0; i < G_VARS_LEN; ++i) {
    Port var = vars_load(net,i);
    if (var != 0) {
      printf("%04X | %s |\n", i, show_port(vars_load(net,i)).x);
    }
  }
  printf("==== | ============ |\n");
}

void pretty_print_port(Net* net, Port port) {
  Port stack[32];
  stack[0] = port;
  u32 len = 1;
  u32 num = 0;
  while (len > 0) {
    if (++num > 256) {
      printf("(...)\n");
      return;
    }
    if (len > 32) {
      printf("...");
      --len;
      continue;
    }
    Port cur = stack[--len];
    if (cur > 0xFFFFFF00) {
      printf("%c", (char)(cur&0xFF));
      continue;
    }
    switch (get_tag(cur)) {
      case CON: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("(");
        stack[len++] = (0xFFFFFF00) | (u32)(')');
        stack[len++] = p2;
        stack[len++] = (0xFFFFFF00) | (u32)(' ');
        stack[len++] = p1;
        break;
      }
      case ERA: {
        printf("*");
        break;
      }
      case VAR: {
        printf("x%x", get_val(cur));
        Port got = vars_load(net, get_val(cur));
        if (got != NONE) {
          printf("=");
          stack[len++] = got;
        }
        break;
      }
      case NUM: {
        Numb word = get_val(cur);
        switch (get_typ(word)) {
          case SYM: printf("[%x]", get_sym(word)); break;
          case U24: printf("%u", get_u24(word)); break;
          case I24: printf("%d", get_i24(word)); break;
          case F24: printf("%f", get_f24(word)); break;
        }
        break;
      }
      case DUP: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("{");
        stack[len++] = (0xFFFFFF00) | (u32)('}');
        stack[len++] = p2;
        stack[len++] = (0xFFFFFF00) | (u32)(' ');
        stack[len++] = p1;
        break;
      }
      case OPR: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("<+ ");
        stack[len++] = (0xFFFFFF00) | (u32)('>');
        stack[len++] = p2;
        stack[len++] = (0xFFFFFF00) | (u32)(' ');
        stack[len++] = p1;
        break;
      }
      case SWI: {
        Pair node = node_load(net,get_val(cur));
        Port p2   = get_snd(node);
        Port p1   = get_fst(node);
        printf("?<");
        stack[len++] = (0xFFFFFF00) | (u32)('>');
        stack[len++] = p2;
        stack[len++] = (0xFFFFFF00) | (u32)(' ');
        stack[len++] = p1;
        break;
      }
      case REF: {
        printf("@%d", get_val(cur));
        break;
      }
    }
  }
}

//void pretty_print_rbag(Net* net, RBag* rbag) {
  //for (u32 i = rbag->lo_ini; i < rbag->lo_end; ++i) {
    //Pair redex = rbag->lo_buf[i];
    //if (redex != 0) {
      //pretty_print_port(net, get_fst(redex));
      //printf(" ~ ");
      //pretty_print_port(net, get_snd(redex));
      //printf("\n");
    //}
  //}
  //for (u32 i = 0; i > rbag->hi_end; ++i) {
    //Pair redex = rbag->hi_buf[i];
    //if (redex != 0) {
      //pretty_print_port(net, get_fst(redex));
      //printf(" ~ ");
      //pretty_print_port(net, get_snd(redex));
      //printf("\n");
    //}
  //}
//}

// Main
// ----

// stress 2^10 x 131072
static const u8 DEMO_BOOK[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 128, 0, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 36, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 112, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 33, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0};

// loop 10m
//static const u8 DEMO_BOOK[] = {3, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 64, 75, 76, 0, 0, 0, 0, 1, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 36, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 22, 0, 0, 0, 16, 0, 0, 0, 3, 2, 0, 128, 30, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0};

// radix 16
//static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 9, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 0, 68, 0, 0, 0, 57, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 0, 52, 0, 0, 0, 81, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 0, 20, 0, 0, 0, 161, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 0, 84, 0, 0, 0, 201, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 0, 36, 0, 0, 0, 225, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 15, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};

// Thread data
typedef struct {
  Net*  net;
  TM*   tm;
  Book* book;
} ThreadArgs;

void* thread_func(void* arg) {
  ThreadArgs* data = (ThreadArgs*)arg;
  evaluator(data->net, data->tm, data->book);
  return NULL;
}

void hvm_c(u32* book_buffer) {
  // Loads the Book
  Book* book = NULL;
  if (book_buffer) {
    book = (Book*)malloc(sizeof(Book));
    book_load(book_buffer, book);
  }

  // GMem
  Net *net = malloc(sizeof(Net));
  net_init(net);

  // Alloc and init TPC TM's
  TM* tm[TPC];
  ThreadArgs thread_args[TPC];
  pthread_t threads[TPC];
  for (u32 t = 0; t < TPC; ++t) {
    tm[t] = malloc(sizeof(TM));
    tmem_init(tm[t], t);
    thread_args[t].net  = net;
    thread_args[t].tm   = tm[t];
    thread_args[t].book = book;
  }

  // Creates an initial redex that calls main
  push_redex(net, tm[0], new_pair(new_port(REF, 0), ROOT));

  // Replace the clock() based timing with time64() based timing
  u64 start = time64();

  // Start the threads
  for (u32 t = 0; t < TPC; ++t) {
    pthread_create(&threads[t], NULL, thread_func, &thread_args[t]);
  }

  // Wait for the threads to finish 
  for (u32 t = 0; t < TPC; ++t) {
    pthread_join(threads[t], NULL);
  }

  u64 end = time64();
  double duration = (end - start) / 1000000000.0; // convert to seconds

  // Prints the result
  printf("Result: ");
  pretty_print_port(net, enter(net, tm[0], ROOT));
  printf("\n");

  // Prints interactions and time
  u64 itrs = atomic_load(&net->itrs);
  printf("- ITRS: %llu\n", itrs);
  printf("- TIME: %.2fs\n", duration);
  printf("- MIPS: %.2f\n", (double)itrs / duration / 1000000.0);

  // Frees values
  for (u32 t = 0; t < TPC; ++t) {
    free(tm[t]);
  }
  free(net);
  free(book);
}

int main() {
  hvm_c((u32*)DEMO_BOOK);
  //hvm_c(NULL);
  return 0;
}
