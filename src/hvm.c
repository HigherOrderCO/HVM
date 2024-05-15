#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INTERPRETED
#define RUN_IO
#define WITHOUT_MAIN
//#define IO_DRAWIMAGE

#ifdef IO_DRAWIMAGE
#include <SDL2/SDL.h>
#endif

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
#define TPC_L2 0
#define TPC    (1 << TPC_L2)

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
#define VAR 0x0 // variable
#define REF 0x1 // reference
#define ERA 0x2 // eraser
#define NUM 0x3 // number
#define CON 0x4 // constructor
#define DUP 0x5 // duplicator
#define OPR 0x6 // operator
#define SWI 0x7 // switch

// Interaction Rule Values
#define LINK 0x0
#define CALL 0x1
#define VOID 0x2
#define ERAS 0x3
#define ANNI 0x4
#define COMM 0x5
#define OPER 0x6
#define SWIT 0x7

// Numbers
#define SYM 0x0
#define U24 0x1
#define I24 0x2
#define F24 0x3
#define ADD 0x4
#define SUB 0x5
#define MUL 0x6
#define DIV 0x7
#define REM 0x8
#define EQ  0x9
#define NEQ 0xA
#define LT  0xB
#define GT  0xC
#define AND 0xD
#define OR  0xE
#define XOR 0xF

// Constants
#define FREE 0x00000000
#define ROOT 0xFFFFFFF8
#define NONE 0xFFFFFFFF

// Cache Padding
#define CACHE_PAD 64

// Global Net
#define HLEN (1 << 16) // max 16k high-priority redexes
#define RLEN (1 << 24) // max 16m low-priority redexes
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
  u32  hput; // next hbag push index
  u32  rput; // next rbag push index
  u32  sidx; // steal index
  u32  nloc[32]; // node allocation indices
  u32  vloc[32]; // vars allocation indices
  Pair hbag_buf[HLEN]; // high-priority redexes
} TM;

// Readback: λ-Encoded Ctr
typedef struct Ctr {
  u32  tag;
  u32  args_len;
  Port args_buf[16];
} Ctr;

// Readback: λ-Encoded Str (UTF-16)
// FIXME: this is actually ASCII :|
// FIXME: remove len limit
typedef struct Str {
  u32  text_len;
  char text_buf[256];
} Str;

// IO Tags
#define DONE      0
#define PUTTEXT   1
#define GETTEXT   2
#define WRITEFILE 3
#define READFILE  4
#define GETTIME   5
#define SLEEP     6
#define DRAWIMAGE 7

// List Type
#define NIL  0
#define CONS 1

// Booleans
#define TRUE  1
#define FALSE 0

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

Pair set_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) | 0x10000000), p2);
  } else {
    return pair;
  }
}

Pair clr_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) & 0xFFFFFFF), p2);
  } else {
    return pair;
  }
}

bool get_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  if (get_tag(p1) == REF) {
    return (get_val(p1) >> 28) == 1;
  } else {
    return FALSE;
  }
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

// FIXME: what about some bound checks?

static inline void push_redex(Net* net, TM* tm, Pair redex) {
  if (is_high_priority(get_pair_rule(redex))) {
    tm->hbag_buf[tm->hput++] = redex;
  } else {
    atomic_store_explicit(&net->rbag_buf[tm->tid*(G_RBAG_LEN/TPC) + (tm->rput++)], redex, memory_order_relaxed);
  }
}

static inline Pair pop_redex(Net* net, TM* tm) {
  if (tm->hput > 0) {
    return tm->hbag_buf[--tm->hput];
  } else if (tm->rput > 0) {
    return atomic_exchange_explicit(&net->rbag_buf[tm->tid*(G_RBAG_LEN/TPC) + (--tm->rput)], 0, memory_order_relaxed);
  } else {
    return 0;
  }
}

static inline u32 rbag_len(Net* net, TM* tm) {
  return tm->rput + tm->hput;
}

// TM
// --

static TM* tm[TPC];

TM* tm_new(u32 tid) {
  TM* tm   = malloc(sizeof(TM));
  tm->tid  = tid;
  tm->itrs = 0;
  tm->nput = 1;
  tm->vput = 1;
  tm->rput = 0;
  tm->hput = 0;
  tm->sidx = 0;
  return tm;
}

void alloc_static_tms() {
  for (u32 t = 0; t < TPC; ++t) {
    tm[t] = tm_new(t);
  }
}

void free_static_tms() {
  for (u32 t = 0; t < TPC; ++t) {
    free(tm[t]);
  }
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
  // is that needed?
  atomic_store(&net->itrs, 0);
  atomic_store(&net->idle, 0);
}

// Allocator
// ---------

u32 node_alloc_1(Net* net, TM* tm, u32* lps) {
  while (TRUE) {
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
  while (TRUE) {
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

// Peeks a variable's final target without modifying it.
static inline Port peek(Net* net, Port var) {
  while (get_tag(var) == VAR) {
    Port val = vars_load(net, get_val(var));
    if (val == NONE) break;
    if (val == 0) break;
    var = val;
  }
  return var;
}

// Finds a variable's value.
static inline Port enter(Net* net, Port var) {
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
  while (TRUE) {
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
    B = enter(net, B);

    // Since `A` is VAR: point `A ~> B`.
    if (TRUE) {
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

// Interactions
// ------------

// The Link Interaction.
static inline bool interact_link(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 0, 0)) {
    return FALSE;
  }
  
  // Links.
  link_pair(net, tm, new_pair(a, b));

  return TRUE;
}

// Declared here for use in call interactions.
static inline bool interact_eras(Net* net, TM* tm, Port a, Port b);

// The Call Interaction.
#ifdef COMPILED
///COMPILED_INTERACT_CALL///
#else
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
    return FALSE;
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

  return TRUE;
}
#endif

// The Void Interaction.
static inline bool interact_void(Net* net, TM* tm, Port a, Port b) {
  return TRUE;
}

// The Eras Interaction.
static inline bool interact_eras(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return FALSE;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable0: %s\n", tid, show_port(b).x);
    return FALSE;
  }

  // Loads ports.
  Pair B  = node_exchange(net, get_val(b), 0);
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);
  
  //if (B == 0) printf("[%04x] ERROR2: %s\n", tid, show_port(b).x);

  // Links.
  link_pair(net, tm, new_pair(a, B1));
  link_pair(net, tm, new_pair(a, B2));

  return TRUE;
}

// The Anni Interaction.
static inline bool interact_anni(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return FALSE;
  }

  // Checks availability
  if (node_load(net, get_val(a)) == 0 || node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable1: %s | %s\n", tid, show_port(a).x, show_port(b).x);
    //printf("BBB\n");
    return FALSE;
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

  return TRUE;
}

// The Comm Interaction.
static inline bool interact_comm(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 4, 4, 4)) {
    return FALSE;
  }

  // Checks availability
  if (node_load(net, get_val(a)) == 0 || node_load(net, get_val(b)) == 0) {
    //printf("[%04x] unavailable2: %s | %s\n", tid, show_port(a).x, show_port(b).x);
    return FALSE;
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

  return TRUE;
}

// The Oper Interaction.
static inline bool interact_oper(Net* net, TM* tm, Port a, Port b) {
  //printf("OPER %08x %08x\n", a, b);

  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 1, 0)) {
    return FALSE;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
    return FALSE;
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
    node_create(net, tm->nloc[0], new_pair(new_port(get_tag(a), flp_flp(av)), B2));
    link_pair(net, tm, new_pair(B1, new_port(OPR, tm->nloc[0])));
  }

  return TRUE;
}

// The Swit Interaction.
static inline bool interact_swit(Net* net, TM* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 2, 0)) {
    return FALSE;
  }

  // Checks availability
  if (node_load(net, get_val(b)) == 0) {
    return FALSE;
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

  return TRUE;
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
      return FALSE;
    // Else, increments the interaction count.
    } else if (rule != LINK) {
      tm->itrs += 1;
    }
  }

  return TRUE;
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
  while (TRUE) {
    tick += 1;

    //if (tm->tid == 1) printf("think %d\n", rbag_len(net, tm));

    // If we have redexes...
    if (rbag_len(net, tm) > 0) {
      // Update global idle counter
      if (!busy) atomic_fetch_sub_explicit(&net->idle, 1, memory_order_relaxed);
      busy = TRUE;
      // Perform an interaction
      interact(net, tm, book);
    // If we have no redexes...
    } else {
      // Update global idle counter
      if (busy) atomic_fetch_add_explicit(&net->idle, 1, memory_order_relaxed);
      busy = FALSE;
      
      //// Peeks a redex from target
      u32  sid = (tm->tid - 1) % TPC;
      u32  idx = sid*(G_RBAG_LEN/TPC) + (tm->sidx++);

      // Steal Parallel: this will only steal parallel redexes

      //Pair trg = atomic_load_explicit(&net->rbag_buf[idx], memory_order_relaxed);
      //// If we're ahead of target, reset
      //if (trg == 0) {
        //tm->sidx = 0;
      //// If the redex is parallel, attempt to steal it
      //} else if (get_par_flag(trg)) {
        //bool stolen = atomic_compare_exchange_weak_explicit(&net->rbag_buf[idx], &trg, 0, memory_order_relaxed, memory_order_relaxed);
        //if (stolen) {
          //push_redex(net, tm, trg);
        //} else {
          //// do nothing: will sched_yield
        //}
      //// If we see a non-stealable redex, try the next one
      //} else {
        //continue;
      //}

      // Stealing Everything: this will steal all redexes
      
      Pair got = atomic_exchange_explicit(&net->rbag_buf[idx], 0, memory_order_relaxed);
      if (got != 0) {
        //printf("[%04x] stolen one task from %04x | itrs=%d idle=%d | %s ~ %s\n", tm->tid, sid, tm->itrs, atomic_load_explicit(&net->idle, memory_order_relaxed),show_port(get_fst(got)).x, show_port(get_snd(got)).x);
        push_redex(net, tm, got);
        continue;
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

// Normalizer
// ----------

// Thread data
typedef struct {
  Net*  net;
  TM*   tm;
  Book* book;
} ThreadArg;

void* thread_func(void* arg) {
  ThreadArg* data = (ThreadArg*)arg;
  evaluator(data->net, data->tm, data->book);
  return NULL;
}

// Sets the initial redex.
void boot_redex(Net* net, Pair redex) {
  net->vars_buf[get_val(ROOT)] = NONE;
  net->rbag_buf[0] = redex;
}

// Evaluates all redexes.
// TODO: cache threads to avoid spawning overhead
void normalize(Net* net, Book* book) {
  // Inits thread_arg objects
  ThreadArg thread_arg[TPC];
  for (u32 t = 0; t < TPC; ++t) {
    thread_arg[t].net  = net;
    thread_arg[t].tm   = tm[t];
    thread_arg[t].book = book;
  }

  // Spawns the evaluation threads
  pthread_t threads[TPC];
  for (u32 t = 0; t < TPC; ++t) {
    pthread_create(&threads[t], NULL, thread_func, &thread_arg[t]);
  }

  // Wait for the threads to finish
  for (u32 t = 0; t < TPC; ++t) {
    pthread_join(threads[t], NULL);
  }
}

// Monadic IO
// ----------

// Util: expands a REF Port.
Port expand(Net* net, Book* book, Port port) {
  Port got = peek(net, port);
  while (get_tag(got) == REF) {
    boot_redex(net, new_pair(new_port(REF,get_val(got)), ROOT));
    normalize(net, book);
    got = peek(net, vars_load(net, get_val(ROOT)));
  }
  return got;
}

// Util: creates a node.
Port make_node(Net* net, Tag tag, Port fst, Port snd) {
  u32 lps = 0;
  u32 loc = node_alloc_1(net, tm[0], &lps);
  node_create(net, loc, new_pair(fst, snd));
  return new_port(tag, loc);
}

// Util: applies an IO cont to the intermediate result.
Port run_io_next(Net* net, Port cont, Port arg) {
  Port app = make_node(net, CON, arg, ROOT);
  boot_redex(net, new_pair(app, cont));
  return ROOT;
}

// Reads back a λ-Encoded constructor from device to host.
// Encoding: λt ((((t TAG) arg0) arg1) ...)
Ctr read_ctr(Net* net, Book* book, Port port) {
  Ctr ctr;
  ctr.tag = -1;
  ctr.args_len = 0;

  // Loads root lambda
  Port lam_port = expand(net, book, port);
  if (get_tag(lam_port) != CON) return ctr;
  Pair lam_node = node_load(net, get_val(lam_port));

  // Loads first application
  Port app_port = expand(net, book, get_fst(lam_node));
  if (get_tag(app_port) != CON) return ctr;
  Pair app_node = node_load(net, get_val(app_port));

  // Loads first argument (as the tag)
  Port arg_port = expand(net, book, get_fst(app_node));
  if (get_tag(arg_port) != NUM) return ctr;
  ctr.tag = get_u24(get_val(arg_port));

  // Loads remaining arguments
  while (TRUE) {
    app_port = expand(net, book, get_snd(app_node));
    if (get_tag(app_port) != CON) break;
    app_node = node_load(net, get_val(app_port));
    arg_port = expand(net, book, get_fst(app_node));
    ctr.args_buf[ctr.args_len++] = arg_port;
  }

  return ctr;
}

// Reads back a UTF-16 string.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
Str read_str(Net* net, Book* book, Port port) {
  // Result
  Str str;
  str.text_len = 0;
  
  // Readback loop
  while (TRUE) {
    // Normalizes the net
    normalize(net, book);

    //printf("reading str %s\n", show_port(peek(net, port)).x);

    // Reads the λ-Encoded Ctr
    Ctr ctr = read_ctr(net, book, peek(net, port));

    //printf("reading tag %d | len %d\n", ctr.tag, ctr.args_len);

    // Reads string layer
    switch (ctr.tag) {
      case NIL: {
        break;
      }
      case CONS: {
        if (ctr.args_len != 2) break;
        if (get_tag(ctr.args_buf[0]) != NUM) break;
        //printf("reading chr %d\n", get_u24(get_val(ctr.args_buf[0])));
        str.text_buf[str.text_len++] = get_u24(get_val(ctr.args_buf[0]));
        boot_redex(net, new_pair(ctr.args_buf[1], ROOT));
        port = ROOT;
        continue;
      }
    }
    break;
  }

  str.text_buf[str.text_len] = '\0';

  return str;
}

// Reads back an image.
// Encoding: (<tree>,<tree>) | #RRGGBB
void read_img(Net* net, Port port, u32 width, u32 height, u32* buffer) {
  pretty_print_port(net, port);
  printf("\n");
  typedef struct {
    Port port; u32 lv;
    u32 x0; u32 x1;
    u32 y0; u32 y1;
  } Rect;
  Rect stk[24];
  u32 pos = 0;
  stk[pos++] = (Rect){port, 0, 0, width, 0, height};
  while (pos > 0) {
    Rect rect = stk[--pos];
    Port port = enter(net, rect.port);
    u32  lv   = rect.lv;
    u32  x0   = rect.x0;
    u32  x1   = rect.x1; 
    u32  y0   = rect.y0;
    u32  y1   = rect.y1;
    if (get_tag(port) == CON) {
      Pair nd = node_load(net, get_val(port));
      Port p1 = get_fst(nd);
      Port p2 = get_snd(nd);
      u32  xm = (x0 + x1) / 2;
      u32  ym = (y0 + y1) / 2;
      if (lv % 2 == 0) {
        stk[pos++] = (Rect){p2, lv+1, xm, x1, y0, y1};
        stk[pos++] = (Rect){p1, lv+1, x0, xm, y0, y1};
      } else {
        stk[pos++] = (Rect){p2, lv+1, x0, x1, ym, y1};
        stk[pos++] = (Rect){p1, lv+1, x0, x1, y0, ym};
      }
      continue;
    }
    if (get_tag(port) == NUM) {
      u32 color = get_u24(get_val(port));
      printf("COL=%08x x0=%04u x1=%04u y0=%04u y1=%04u | %s\n", color, x0, x1, y0, y1, show_port(port).x);
      for (u32 y = y0; y < y1; y++) {
        for (u32 x = x0; x < x1; x++) {
          buffer[y*width + x] = 0xFF000000 | color;
        }
      }
      continue;
    }
    break;
  }
}

// IO: PutText
Port io_puttext(Net* net, Book* book, u32 argc, Port* argv) {
  // Checks argument count
  if (argc != 1) return NONE;

  // Converts first argument to C string
  Str str = read_str(net, book, argv[0]);

  // Prints it
  printf("%s", str.text_buf);

  // Returns result (in this case, just an eraser)
  return new_port(ERA, 0);
}

#ifdef IO_DRAWIMAGE
// Global variables for the window and renderer
static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;
static SDL_Texture *texture = NULL;
// Function to close the SDL window and clean up resources
void close_sdl(void) {
  if (texture != NULL) {
    SDL_DestroyTexture(texture);
    texture = NULL;
  }
  if (renderer != NULL) {
    SDL_DestroyRenderer(renderer);
    renderer = NULL;
  }
  if (window != NULL) {
    SDL_DestroyWindow(window);
    window = NULL;
  }
  SDL_Quit();
}
// Function to render an image to the SDL window
void render(uint32_t width, uint32_t height, uint32_t *buffer) {
  // Initialize SDL if it hasn't been initialized
  if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
      fprintf(stderr, "SDL could not initialize! SDL Error: %s\n", SDL_GetError());
      return;
    }
  }
  // Create window and renderer if they don't exist
  if (window == NULL) {
    window = SDL_CreateWindow("SDL Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    if (window == NULL) {
      fprintf(stderr, "Window could not be created! SDL Error: %s\n", SDL_GetError());
      return;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
      SDL_DestroyWindow(window);
      window = NULL;
      fprintf(stderr, "Renderer could not be created! SDL Error: %s\n", SDL_GetError());
      return;
    }
  }
  // Create or recreate the texture if necessary
  if (texture == NULL) {
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (texture == NULL) {
      fprintf(stderr, "Texture could not be created! SDL Error: %s\n", SDL_GetError());
      return;
    }
  }
  // Update the texture with the new buffer
  if (SDL_UpdateTexture(texture, NULL, buffer, width * sizeof(uint32_t)) < 0) {
    fprintf(stderr, "Texture could not be updated! SDL Error: %s\n", SDL_GetError());
    return;
  }
  // Clear the renderer
  SDL_RenderClear(renderer);
  // Copy the texture to the renderer
  SDL_RenderCopy(renderer, texture, NULL, NULL);
  // Update the screen
  SDL_RenderPresent(renderer);
  // Process events to prevent the OS from thinking the application is unresponsive
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_QUIT) {
      close_sdl();
      exit(0);
    }
  }
}
// IO: DrawImage
Port io_drawimage(Net* net, Book* book, u32 argc, Port* argv) {
  u32 width = 256;
  u32 height = 256;
  // Create a buffer
  uint32_t *buffer = (uint32_t *)malloc(width * height * sizeof(uint32_t));
  if (buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory for buffer\n");
    return 1;
  }
  // Initialize buffer to a dark blue background
  for (int i = 0; i < width * height; ++i) {
    buffer[i] = 0xFF000030; // Dark blue background
  }
  // Converts a HVM2 tuple-encoded quadtree to a color buffer
  read_img(net, argv[0], width, height, buffer);
  // Render the buffer to the screen
  render(width, height, buffer);
  // Wait some time
  SDL_Delay(2000);
  // Free the buffer
  free(buffer);
  return new_port(ERA, 0);
}
#else
// IO: DrawImage
Port io_drawimage(Net* net, Book* book, u32 argc, Port* argv) {
  printf("DRAWIMAGE: disabled.\n");
  printf("Image rendering is a WIP. For now, to enable it, you must:\n");
  printf("1. Generate a C file, with `hvm gen-c your_file.hvm`.\n");
  printf("2. Manually un-comment the '#define IO_DRAWIMAGE' line on it.\n");
  printf("3. Have SDL installed and compile it with '-lSDL2'.\n");
  return new_port(ERA, 0);
}
#endif

// IO: GetTime
Port io_gettime(Net* net, Book* book, u32 argc, Port* argv) {
  // Get the current time in nanoseconds 
  u64 time_ns = time64();

  // Encode the time as a 64-bit unsigned integer
  u32 time_hi = (u32)(time_ns >> 24) & 0xFFFFFFF;
  u32 time_lo = (u32)(time_ns & 0xFFFFFFF);
  
  // Allocate a node to store the time
  u32 loc = node_alloc_1(net, tm[0], &argc);
  node_create(net, loc, new_pair(new_port(NUM, new_u24(time_hi)), new_port(NUM, new_u24(time_lo))));

  // Return the encoded time
  return new_port(CON, loc);
}

// IO: Sleep 
// NOTE: receives a CON node. it decodes it into a u48 similarly to above.
Port io_sleep(Net* net, Book* book, u32 argc, Port* argv) {
  // Check argument count
  if (argc != 1) return NONE;

    // Get the sleep duration node
  Pair dur_node = node_load(net, get_val(argv[0]));
  
  // Get the high and low 24-bit parts of the duration 
  u32 dur_hi = get_u24(get_val(get_fst(dur_node)));
  u32 dur_lo = get_u24(get_val(get_snd(dur_node)));

  // Combine into a 48-bit duration in nanoseconds
  u64 dur_ns = (((u64)dur_hi) << 24) | dur_lo;
  
  // Sleep for the specified duration
  struct timespec ts;
  ts.tv_sec = dur_ns / 1000000000;
  ts.tv_nsec = dur_ns % 1000000000;
  nanosleep(&ts, NULL);

  // Return an eraser
  return new_port(ERA, 0);
}

// Runs an IO computation.
bool do_run_io(Net* net, Book* book, Port port) {
  // IO loop
  while (TRUE) {
    // Normalizes the net
    normalize(net, book);

    // Reads the λ-Encoded Ctr
    Ctr ctr = read_ctr(net, book, peek(net, port));

    // Dispatches IO function
    switch (ctr.tag) {
      case PUTTEXT: {
        Port res = io_puttext(net, book, ctr.args_len-1, ctr.args_buf);
        if (res == NONE) break;
        port = run_io_next(net, ctr.args_buf[1], res);
        continue;
      }
      case GETTEXT: {
        printf("GETTEXT: not implemented yet :(\n");
        continue;
      }
      case WRITEFILE: {
        printf("WRITEFILE: not implemented yet :(\n");
        continue;
      }
      case READFILE: {
        printf("READFILE: not implemented yet :(\n");
        continue;
      }
      case GETTIME: {
        Port res = io_gettime(net, book, ctr.args_len-1, ctr.args_buf);
        if (res == NONE) break;
        port = run_io_next(net, ctr.args_buf[0], res);
        continue;
      }
      case SLEEP: {
        Port res = io_sleep(net, book, ctr.args_len-1, ctr.args_buf);
        if (res == NONE) break;
        port = run_io_next(net, ctr.args_buf[1], res);
        continue;
      }
      case DRAWIMAGE: {
        Port res = io_drawimage(net, book, ctr.args_len-1, ctr.args_buf);
        if (res == NONE) break;
        port = run_io_next(net, ctr.args_buf[1], res);
        continue;
      }

      default: {
        break;
      }
    }

    break;
  }

  return TRUE;
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
  Port stack[256];
  stack[0] = port;
  u32 len = 1;
  u32 num = 0;
  while (len > 0) {
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
        Port got = vars_load(net, get_val(cur));
        if (got != NONE) {
          stack[len++] = got;
        } else {
          printf("x%x", get_val(cur));
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

// Demos
// -----

  // stress_test 2^18 x 131072
  //static const u8 DEMO_BOOK[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 20, 0, 0, 0, 9, 0, 0, 128, 44, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 111, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0};

  // stress_test 2^14 x 131072
  //static const u8 DEMO_BOOK[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 20, 0, 0, 0, 9, 0, 0, 128, 44, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 111, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0};

  // stress_test 2^10 x 131072
  //static const u8 DEMO_BOOK[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 20, 0, 0, 0, 9, 0, 0, 128, 44, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 111, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0};

  // bitonic_sort 16
  //static const u8 DEMO_BOOK[] = {17, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 41, 0, 0, 0, 44, 0, 0, 0, 11, 8, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 8, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 8, 0, 0, 52, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 113, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 32, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 65, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 115, 111, 114, 116, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 57, 0, 0, 128, 92, 0, 0, 0, 57, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 139, 0, 0, 0, 132, 0, 0, 0, 40, 0, 0, 0, 64, 0, 0, 0, 9, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 10, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 73, 0, 0, 128, 36, 0, 0, 0, 73, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 11, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 15, 0, 0, 0, 119, 97, 114, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 0, 76, 0, 0, 0, 14, 0, 0, 0, 28, 0, 0, 0, 131, 7, 0, 128, 22, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 37, 0, 0, 0, 60, 0, 0, 0, 46, 0, 0, 0, 24, 0, 0, 0, 3, 6, 0, 128, 54, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 113, 0, 0, 128, 92, 0, 0, 0, 113, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 116, 0, 0, 0, 48, 0, 0, 0, 124, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 140, 0, 0, 0, 24, 0, 0, 0, 148, 0, 0, 0, 40, 0, 0, 0, 156, 0, 0, 0, 56, 0, 0, 0, 164, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0};

  // bitonic_sort 18
  //static const u8 DEMO_BOOK[] = {17, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 41, 0, 0, 0, 44, 0, 0, 0, 11, 9, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 9, 0, 0, 52, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 113, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 32, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 65, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 115, 111, 114, 116, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 57, 0, 0, 128, 92, 0, 0, 0, 57, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 139, 0, 0, 0, 132, 0, 0, 0, 40, 0, 0, 0, 64, 0, 0, 0, 9, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 10, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 73, 0, 0, 128, 36, 0, 0, 0, 73, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 11, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 15, 0, 0, 0, 119, 97, 114, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 0, 76, 0, 0, 0, 14, 0, 0, 0, 28, 0, 0, 0, 131, 7, 0, 128, 22, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 37, 0, 0, 0, 60, 0, 0, 0, 46, 0, 0, 0, 24, 0, 0, 0, 3, 6, 0, 128, 54, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 113, 0, 0, 128, 92, 0, 0, 0, 113, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 116, 0, 0, 0, 48, 0, 0, 0, 124, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 140, 0, 0, 0, 24, 0, 0, 0, 148, 0, 0, 0, 40, 0, 0, 0, 156, 0, 0, 0, 56, 0, 0, 0, 164, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0};

  // bitonic_sort 20
  //static const u8 DEMO_BOOK[] = {17, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 41, 0, 0, 0, 44, 0, 0, 0, 11, 10, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 10, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 52, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 113, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 32, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 65, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 115, 111, 114, 116, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 57, 0, 0, 128, 92, 0, 0, 0, 57, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 139, 0, 0, 0, 132, 0, 0, 0, 40, 0, 0, 0, 64, 0, 0, 0, 9, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 10, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 73, 0, 0, 128, 36, 0, 0, 0, 73, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 11, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 15, 0, 0, 0, 119, 97, 114, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 0, 76, 0, 0, 0, 14, 0, 0, 0, 28, 0, 0, 0, 131, 7, 0, 128, 22, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 37, 0, 0, 0, 60, 0, 0, 0, 46, 0, 0, 0, 24, 0, 0, 0, 3, 6, 0, 128, 54, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 113, 0, 0, 128, 92, 0, 0, 0, 113, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 116, 0, 0, 0, 48, 0, 0, 0, 124, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 140, 0, 0, 0, 24, 0, 0, 0, 148, 0, 0, 0, 40, 0, 0, 0, 156, 0, 0, 0, 56, 0, 0, 0, 164, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0};

  // bitonic_sort 22
  //static const u8 DEMO_BOOK[] = {17, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 41, 0, 0, 0, 44, 0, 0, 0, 11, 11, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 11, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 11, 0, 0, 52, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 113, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 32, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 65, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 115, 111, 114, 116, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 57, 0, 0, 128, 92, 0, 0, 0, 57, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 139, 0, 0, 0, 132, 0, 0, 0, 40, 0, 0, 0, 64, 0, 0, 0, 9, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 10, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 73, 0, 0, 128, 36, 0, 0, 0, 73, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 11, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 15, 0, 0, 0, 119, 97, 114, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 0, 76, 0, 0, 0, 14, 0, 0, 0, 28, 0, 0, 0, 131, 7, 0, 128, 22, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 37, 0, 0, 0, 60, 0, 0, 0, 46, 0, 0, 0, 24, 0, 0, 0, 3, 6, 0, 128, 54, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 113, 0, 0, 128, 92, 0, 0, 0, 113, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 116, 0, 0, 0, 48, 0, 0, 0, 124, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 140, 0, 0, 0, 24, 0, 0, 0, 148, 0, 0, 0, 40, 0, 0, 0, 156, 0, 0, 0, 56, 0, 0, 0, 164, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0};

  // bitonic_sort 24
  //static const u8 DEMO_BOOK[] = {17, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 41, 0, 0, 0, 44, 0, 0, 0, 11, 12, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 12, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 12, 0, 0, 52, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 32, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 113, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 100, 0, 0, 0, 32, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 65, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 115, 111, 114, 116, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 57, 0, 0, 128, 92, 0, 0, 0, 57, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 139, 0, 0, 0, 132, 0, 0, 0, 40, 0, 0, 0, 64, 0, 0, 0, 9, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 10, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 73, 0, 0, 128, 36, 0, 0, 0, 73, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 11, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 129, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 15, 0, 0, 0, 119, 97, 114, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 0, 76, 0, 0, 0, 14, 0, 0, 0, 28, 0, 0, 0, 131, 7, 0, 128, 22, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 37, 0, 0, 0, 60, 0, 0, 0, 46, 0, 0, 0, 24, 0, 0, 0, 3, 6, 0, 128, 54, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 113, 0, 0, 128, 92, 0, 0, 0, 113, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 32, 0, 0, 0, 116, 0, 0, 0, 48, 0, 0, 0, 124, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0, 8, 0, 0, 0, 140, 0, 0, 0, 24, 0, 0, 0, 148, 0, 0, 0, 40, 0, 0, 0, 156, 0, 0, 0, 56, 0, 0, 0, 164, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0};

  // radix_sort 14
  //static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 7, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 128, 68, 0, 0, 0, 57, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 128, 52, 0, 0, 0, 81, 0, 0, 128, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 128, 20, 0, 0, 0, 161, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 128, 84, 0, 0, 0, 201, 0, 0, 128, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 128, 36, 0, 0, 0, 225, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};

  // radix_sort 16
  //static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 8, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 128, 68, 0, 0, 0, 57, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 128, 52, 0, 0, 0, 81, 0, 0, 128, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 128, 20, 0, 0, 0, 161, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 128, 84, 0, 0, 0, 201, 0, 0, 128, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 128, 36, 0, 0, 0, 225, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};

  // radix_sort 17
  //static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 139, 8, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 128, 68, 0, 0, 0, 57, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 128, 52, 0, 0, 0, 81, 0, 0, 128, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 128, 20, 0, 0, 0, 161, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 128, 84, 0, 0, 0, 201, 0, 0, 128, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 128, 36, 0, 0, 0, 225, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};

  // radix_sort 18
  //static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 9, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 128, 68, 0, 0, 0, 57, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 128, 52, 0, 0, 0, 81, 0, 0, 128, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 128, 20, 0, 0, 0, 161, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 128, 84, 0, 0, 0, 201, 0, 0, 128, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 128, 36, 0, 0, 0, 225, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};
  
  // radix_sort 20
  //static const u8 DEMO_BOOK[] = {31, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 161, 0, 0, 0, 4, 0, 0, 0, 153, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 66, 117, 115, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 67, 111, 110, 99, 97, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 3, 0, 0, 0, 69, 109, 112, 116, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 114, 101, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 78, 111, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 6, 0, 0, 0, 83, 105, 110, 103, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 65, 0, 0, 0, 73, 0, 0, 0, 8, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 103, 101, 110, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 52, 0, 0, 0, 57, 0, 0, 128, 68, 0, 0, 0, 57, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 32, 0, 0, 0, 51, 1, 0, 0, 37, 0, 0, 0, 16, 0, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 60, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 10, 0, 0, 0, 109, 101, 114, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 20, 0, 0, 0, 113, 0, 0, 0, 28, 0, 0, 0, 105, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 81, 0, 0, 128, 52, 0, 0, 0, 81, 0, 0, 128, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 24, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 48, 0, 0, 0, 12, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 97, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 89, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 109, 101, 114, 103, 101, 36, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 9, 0, 0, 0, 28, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 114, 97, 100, 105, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 76, 0, 0, 0, 20, 0, 0, 0, 52, 0, 0, 0, 28, 0, 0, 0, 145, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 18, 0, 0, 0, 114, 97, 100, 105, 120, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 76, 0, 0, 0, 177, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 21, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 30, 0, 0, 0, 131, 6, 0, 128, 38, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 68, 0, 0, 0, 62, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 32, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 24, 0, 0, 0, 116, 0, 0, 0, 40, 0, 0, 0, 124, 0, 0, 0, 33, 0, 0, 0, 56, 0, 0, 0, 19, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 201, 0, 0, 0, 12, 0, 0, 0, 225, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 20, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 161, 0, 0, 128, 20, 0, 0, 0, 161, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 2, 0, 128, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 185, 0, 0, 0, 193, 0, 0, 0, 23, 0, 0, 0, 115, 119, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 115, 119, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 116, 111, 95, 97, 114, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 25, 0, 0, 0, 217, 0, 0, 0, 36, 0, 0, 0, 209, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 17, 0, 0, 0, 68, 0, 0, 0, 201, 0, 0, 128, 84, 0, 0, 0, 201, 0, 0, 128, 100, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 29, 0, 0, 0, 32, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 62, 0, 0, 0, 35, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 8, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 27, 0, 0, 0, 116, 111, 95, 97, 114, 114, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 49, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 116, 111, 95, 109, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 20, 0, 0, 0, 241, 0, 0, 0, 28, 0, 0, 0, 233, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 128, 36, 0, 0, 0, 225, 0, 0, 128, 44, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 32, 0, 0, 0, 30, 0, 0, 0, 116, 111, 95, 109, 97, 112, 36, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 10, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 139, 0, 0, 0, 36, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0};

  // tree_sum 24
  //static const u8 DEMO_BOOK[] = {5, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 11, 12, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 12, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 68, 0, 0, 0, 9, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 4, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 36, 0, 0, 0, 25, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0};

  // tree_sum 26
  //static const u8 DEMO_BOOK[] = {5, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 11, 13, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 13, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 68, 0, 0, 0, 9, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 4, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 36, 0, 0, 0, 25, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0};

  // tree_sum 28
  //static const u8 DEMO_BOOK[] = {5, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 20, 0, 0, 0, 11, 14, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 14, 0, 0, 28, 0, 0, 0, 11, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 103, 101, 110, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 68, 0, 0, 0, 9, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 51, 1, 0, 0, 46, 0, 0, 0, 163, 0, 0, 0, 16, 0, 0, 0, 51, 1, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 3, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 36, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 4, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 36, 0, 0, 0, 25, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 2, 0, 128, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0};

  // count 16k
  //static const u8 DEMO_BOOK[] = {3, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 0, 32, 0, 0, 0, 0, 0, 1, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 36, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 22, 0, 0, 0, 16, 0, 0, 0, 3, 2, 0, 128, 30, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0};

  // count 1m
  //static const u8 DEMO_BOOK[] = {3, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 8, 0, 0, 0, 0, 1, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 17, 0, 0, 0, 2, 0, 0, 0, 115, 117, 109, 36, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 36, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 22, 0, 0, 0, 16, 0, 0, 0, 3, 2, 0, 128, 30, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0};

  // DEMO IO
  static const u8 DEMO_BOOK[] = {37, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 145, 0, 0, 0, 4, 0, 0, 0, 137, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 67, 79, 78, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 0, 0, 0, 2, 0, 0, 0, 68, 79, 78, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 3, 0, 0, 0, 68, 82, 65, 87, 73, 77, 65, 71, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 3, 0, 0, 4, 0, 0, 0, 71, 69, 84, 84, 69, 88, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0, 5, 0, 0, 0, 71, 69, 84, 84, 73, 77, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 2, 0, 0, 6, 0, 0, 0, 73, 79, 47, 68, 111, 110, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0, 2, 0, 0, 0, 76, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 73, 79, 47, 68, 114, 97, 119, 73, 109, 97, 103, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0, 2, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 73, 79, 47, 71, 101, 116, 84, 101, 120, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0, 2, 0, 0, 0, 76, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 73, 79, 47, 71, 101, 116, 84, 105, 109, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 76, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 10, 0, 0, 0, 73, 79, 47, 80, 117, 116, 84, 101, 120, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 2, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0, 2, 0, 0, 0, 76, 0, 0, 0, 2, 0, 0, 0, 84, 0, 0, 0, 2, 0, 0, 0, 92, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 11, 0, 0, 0, 73, 79, 47, 82, 101, 97, 100, 70, 105, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 2, 0, 0, 0, 84, 0, 0, 0, 2, 0, 0, 0, 92, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 12, 0, 0, 0, 73, 79, 47, 83, 108, 101, 101, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 2, 0, 0, 0, 60, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0, 76, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0, 73, 79, 47, 87, 114, 105, 116, 101, 70, 105, 108, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 2, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 92, 0, 0, 0, 2, 0, 0, 0, 100, 0, 0, 0, 2, 0, 0, 0, 108, 0, 0, 0, 2, 0, 0, 0, 24, 0, 0, 0, 14, 0, 0, 0, 77, 97, 105, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 13, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 217, 0, 0, 0, 4, 0, 0, 0, 17, 1, 0, 0, 12, 0, 0, 0, 17, 1, 0, 0, 28, 0, 0, 0, 17, 1, 0, 0, 44, 0, 0, 0, 17, 1, 0, 0, 60, 0, 0, 0, 17, 1, 0, 0, 76, 0, 0, 0, 17, 1, 0, 0, 92, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 139, 59, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 139, 55, 0, 0, 36, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 11, 57, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 11, 54, 0, 0, 68, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 11, 50, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 40, 0, 0, 0, 11, 5, 0, 0, 100, 0, 0, 0, 25, 1, 0, 0, 48, 0, 0, 0, 15, 0, 0, 0, 77, 97, 105, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 81, 0, 0, 0, 12, 0, 0, 0, 57, 0, 0, 0, 36, 0, 0, 0, 97, 0, 0, 0, 84, 0, 0, 0, 49, 0, 0, 0, 116, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 113, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 68, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 11, 0, 128, 127, 11, 128, 127, 0, 139, 127, 0, 0, 139, 127, 128, 127, 76, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 92, 0, 0, 0, 100, 0, 0, 0, 139, 59, 0, 0, 11, 0, 202, 26, 108, 0, 0, 0, 16, 0, 0, 0, 2, 0, 0, 0, 24, 0, 0, 0, 11, 21, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 77, 97, 105, 110, 95, 95, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 13, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 217, 0, 0, 0, 4, 0, 0, 0, 17, 1, 0, 0, 12, 0, 0, 0, 17, 1, 0, 0, 28, 0, 0, 0, 17, 1, 0, 0, 44, 0, 0, 0, 17, 1, 0, 0, 60, 0, 0, 0, 17, 1, 0, 0, 76, 0, 0, 0, 17, 1, 0, 0, 92, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 11, 52, 0, 0, 20, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 139, 50, 0, 0, 36, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 11, 54, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 11, 54, 0, 0, 68, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 139, 55, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 40, 0, 0, 0, 11, 5, 0, 0, 100, 0, 0, 0, 25, 1, 0, 0, 48, 0, 0, 0, 17, 0, 0, 0, 77, 97, 105, 110, 95, 95, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 4, 0, 0, 0, 129, 0, 0, 0, 12, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 77, 107, 73, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 209, 0, 0, 0, 20, 0, 0, 0, 201, 0, 0, 0, 28, 0, 0, 0, 193, 0, 0, 0, 36, 0, 0, 0, 185, 0, 0, 0, 44, 0, 0, 0, 177, 0, 0, 0, 52, 0, 0, 0, 169, 0, 0, 0, 60, 0, 0, 0, 161, 0, 0, 0, 68, 0, 0, 0, 153, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 36, 0, 0, 0, 32, 0, 0, 0, 25, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 20, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 36, 0, 0, 0, 32, 0, 0, 0, 9, 1, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 21, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 52, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 41, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 22, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 36, 0, 0, 0, 32, 0, 0, 0, 1, 1, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 23, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 40, 0, 0, 0, 33, 1, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 76, 0, 0, 0, 40, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 52, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 33, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 25, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 145, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 36, 0, 0, 0, 32, 0, 0, 0, 249, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 52, 0, 0, 0, 60, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 26, 0, 0, 0, 77, 107, 73, 79, 95, 95, 67, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 17, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 27, 0, 0, 0, 77, 107, 83, 116, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 233, 0, 0, 0, 20, 0, 0, 0, 225, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 77, 107, 83, 116, 114, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 241, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 77, 107, 83, 116, 114, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 217, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 24, 0, 0, 0, 9, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 30, 0, 0, 0, 78, 73, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 31, 0, 0, 0, 80, 85, 84, 84, 69, 88, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 0, 0, 0, 32, 0, 0, 0, 82, 69, 65, 68, 70, 73, 76, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 2, 0, 0, 33, 0, 0, 0, 83, 76, 69, 69, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 3, 0, 0, 34, 0, 0, 0, 83, 116, 114, 105, 110, 103, 47, 99, 111, 110, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 35, 0, 0, 0, 83, 116, 114, 105, 110, 103, 47, 110, 105, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 87, 82, 73, 84, 69, 70, 73, 76, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 1, 0, 0};

// Main
// ----

void hvm_c(u32* book_buffer, bool run_io) {
  // Creates static TMs
  alloc_static_tms();

  // Loads the Book
  Book* book = NULL;
  if (book_buffer) {
    book = (Book*)malloc(sizeof(Book));
    book_load(book_buffer, book);
  }

  // Starts the timer
  u64 start = time64();

  // GMem
  Net *net = malloc(sizeof(Net));
  net_init(net);

  // Creates an initial redex that calls main
  boot_redex(net, new_pair(new_port(REF, 0), ROOT));

  // Normalizes and runs IO
  if (run_io) {
    do_run_io(net, book, ROOT);
  } else {
    normalize(net, book);
  }

  // Prints the result
  printf("Result: ");
  pretty_print_port(net, enter(net, ROOT));
  printf("\n");

  // Stops the timer
  double duration = (time64() - start) / 1000000000.0; // seconds

  // Prints interactions and time
  u64 itrs = atomic_load(&net->itrs);
  printf("- ITRS: %lu\n", itrs);
  printf("- TIME: %.2fs\n", duration);
  printf("- MIPS: %.2f\n", (double)itrs / duration / 1000000.0);

  // Frees everything
  free_static_tms();
  free(net);
  free(book);
}

#ifdef WITH_MAIN
int main() {
  //hvm_c((u32*)DEMO_BOOK, false);
  
#ifdef RUN_IO
  hvm_c(NULL, TRUE);
#else
  hvm_c(NULL, FALSE);
#endif

  return 0;
}
#endif
