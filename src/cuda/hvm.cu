#define INTERPRETED
#define WITHOUT_MAIN

#ifdef DEBUG
  #define debug(...) printf(__VA_ARGS__)
#else
  #define debug(...)
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Integers
// --------

typedef  uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef  int32_t i32;
typedef    float f32;
typedef   double f64;
typedef unsigned long long int u64;

// Configuration
// -------------

// Clocks per Second
const u64 S = 2520000000;

// Threads per Block
const u32 TPB_L2 = 7;
const u32 TPB    = 1 << TPB_L2;

// Blocks per GPU
const u32 BPG_L2 = 7;
const u32 BPG    = 1 << BPG_L2;

// Threads per GPU
const u32 TPG = TPB * BPG;

//#define ALLOC_MODE SHARED
//#define ALLOC_MODE GLOBAL

// Types
// -----

// Local Types
typedef u8  Tag;  // Tag  ::= 3-bit (rounded up to u8)
typedef u32 Val;  // Val  ::= 29-bit (rounded up to u32)
typedef u32 Port; // Port ::= Tag + Val (fits a u32)
typedef u64 Pair; // Pair ::= Port + Port (fits a u64)

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

// Constants
const Port FREE = 0x00000000;
const Port ROOT = 0xFFFFFFF8;
const Port NONE = 0xFFFFFFFF;

// Numbers
const Tag TY_SYM = 0x00;
const Tag TY_U24 = 0x01;
const Tag TY_I24 = 0x02;
const Tag TY_F24 = 0x03;
const Tag OP_ADD = 0x04;
const Tag OP_SUB = 0x05;
const Tag FP_SUB = 0x06;
const Tag OP_MUL = 0x07;
const Tag OP_DIV = 0x08;
const Tag FP_DIV = 0x09;
const Tag OP_REM = 0x0A;
const Tag FP_REM = 0x0B;
const Tag OP_EQ  = 0x0C;
const Tag OP_NEQ = 0x0D;
const Tag OP_LT  = 0x0E;
const Tag OP_GT  = 0x0F;
const Tag OP_AND = 0x10;
const Tag OP_OR  = 0x11;
const Tag OP_XOR = 0x12;
const Tag OP_SHL = 0x13;
const Tag FP_SHL = 0x14;
const Tag OP_SHR = 0x15;
const Tag FP_SHR = 0x16;

// Evaluation Modes
const u8 SEED = 0;
const u8 GROW = 1;
const u8 WORK = 2;

// Thread Redex Bag Length
const u32 RLEN = 256;

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

// Local Net
const u32 L_NODE_LEN = 0x2000;
const u32 L_VARS_LEN = 0x2000;
struct LNet {
  Pair node_buf[L_NODE_LEN];
  Port vars_buf[L_VARS_LEN];
};

// Global Net
const u32 G_NODE_LEN = 1 << 29; // max 536m nodes
const u32 G_VARS_LEN = 1 << 29; // max 536m vars
const u32 G_RBAG_LEN = TPB * BPG * RLEN * 3; // max 4m redexes
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

typedef struct Book Book;

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

// Debugger
// --------

struct Show {
  char x[13];
};

__device__ __host__ void put_u16(char* B, u16 val);
__device__ __host__ Show show_port(Port port);
__device__ Show show_rule(Rule rule);
__device__ void print_rbag(Net* net, TM* tm);
__device__ __host__ void print_net(Net* net, u32, u32);
__device__ void pretty_print_numb(Numb word);
__device__ void pretty_print_port(Net* net, Port port);
__device__ void pretty_print_rbag(Net* net, RBag* rbag);
__global__ void print_heatmap(GNet* gnet, u32 turn);

// Utils
// -----

__device__ __host__ f32 clamp(f32 x, f32 min, f32 max) {
  const f32 t = x < min ? min : x;
  return (t > max) ? max : t;
}

// TODO: write a time64() function that returns the time as fast as possible as a u64
static inline u64 time64() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

__device__ inline u32 TID() {
  return threadIdx.x;
}

__device__ inline u32 BID() {
  return blockIdx.x;
}

__device__ inline u32 GID() {
  return TID() + BID() * blockDim.x;
}

__device__ __host__ inline u32 div(u32 a, u32 b) {
  return (a + b - 1) / b;
}

__device__ u32 push_index(u32 msk, u32 idx) {
  return msk | (1U << (31 - idx));
}

__device__ u32 pop_index(u32* msk) {
  u32 idx = __clz(*msk);
  *msk &= ~(1U << (31 - idx));
  return idx;
}

// Port: Constructor and Getters
// -----------------------------

__device__ __host__ inline Port new_port(Tag tag, Val val) {
  return (val << 3) | tag;
}

__device__ __host__ inline Tag get_tag(Port port) {
  return port & 7;
}

__device__ __host__ inline Val get_val(Port port) {
  return port >> 3;
}

// Pair: Constructor and Getters
// -----------------------------

__device__ __host__ inline Pair new_pair(Port fst, Port snd) {
  return ((u64)snd << 32) | fst;
}

__device__ __host__ inline Port get_fst(Pair pair) {
  return pair & 0xFFFFFFFF;
}

__device__ __host__ inline Port get_snd(Pair pair) {
  return pair >> 32;
}

__device__ __host__ Pair set_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) | 0x10000000), p2);
  } else {
    return pair;
  }
}

__device__ __host__ Pair clr_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  Port p2 = get_snd(pair);
  if (get_tag(p1) == REF) {
    return new_pair(new_port(get_tag(p1), get_val(p1) & 0xFFFFFFF), p2);
  } else {
    return pair;
  }
}

__device__ __host__ bool get_par_flag(Pair pair) {
  Port p1 = get_fst(pair);
  if (get_tag(p1) == REF) {
    return (get_val(p1) >> 28) == 1;
  } else {
    return false;
  }
}

// Utils
// -----

// Swaps two ports.
__device__ __host__ inline void swap(Port *a, Port *b) {
  Port x = *a; *a = *b; *b = x;
}

// Transposes an index over a matrix.
__device__ u32 transpose(u32 idx, u32 width, u32 height) {
  u32 old_row = idx / width;
  u32 old_col = idx % width;
  u32 new_row = old_col % height;
  u32 new_col = old_col / height + old_row * (width / height);
  return new_row * width + new_col;
}

// Returns true if all 'x' are true, block-wise
__device__ __noinline__ bool block_all(bool x) {
  __shared__ bool res;
  if (TID() == 0) res = true;
  __syncthreads();
  if (!x) res = false;
  __syncthreads();
  return res;
}

// Returns true if any 'x' is true, block-wise
__device__ __noinline__ bool block_any(bool x) {
  __shared__ bool res;
  if (TID() == 0) res = false;
  __syncthreads();
  if (x) res = true;
  __syncthreads();
  return res;
}

// Returns the sum of a value, block-wise
template <typename A>
__device__ __noinline__ A block_sum(A x) {
  __shared__ A res;
  if (TID() == 0) res = 0;
  __syncthreads();
  atomicAdd(&res, x);
  __syncthreads();
  return res;
}

// Returns the sum of a boolean, block-wise
__device__ __noinline__ u32 block_count(bool x) {
  __shared__ u32 res;
  if (TID() == 0) res = 0;
  __syncthreads();
  atomicAdd(&res, x);
  __syncthreads();
  return res;
}

// Prints a 4-bit value for each thread in a block
__device__ void block_print(u32 x) {
  __shared__ u8 value[TPB];

  value[TID()] = x;
  __syncthreads();

  if (TID() == 0) {
    for (u32 i = 0; i < TPB; ++i) {
      printf("%x", min(value[i],0xF));
    }
  }
  __syncthreads();
}

// Ports / Pairs / Rules
// ---------------------

// True if this port has a pointer to a node.
__device__ __host__ inline bool is_nod(Port a) {
  return get_tag(a) >= CON;
}

// True if this port is a variable.
__device__ __host__ inline bool is_var(Port a) {
  return get_tag(a) == VAR;
}

// True if this port is a local node/var (that can leak).
__device__ __host__ inline bool is_local(Port a) {
  return (is_nod(a) || is_var(a)) && get_val(a) < L_NODE_LEN;
}

// True if this port is a global node/var (that can be leaked into).
__device__ __host__ inline bool is_global(Port a) {
  return (is_nod(a) || is_var(a)) && get_val(a) >= L_NODE_LEN;
}

// Given two tags, gets their interaction rule. Uses a u64mask lookup table.
__device__ __host__ inline Rule get_rule(Port A, Port B) {
  const u64 x = 0b0111111010110110110111101110111010110000111100001111000100000010;
  const u64 y = 0b0000110000001100000011100000110011111110111111100010111000000000;
  const u64 z = 0b1111100011111000111100001111000011000000000000000000000000000000;
  const u64 i = ((u64)get_tag(A) << 3) | (u64)get_tag(B);
  return (Rule)((x>>i&1) | (y>>i&1)<<1 | (z>>i&1)<<2);
}

// Same as above, but receiving a pair.
__device__ __host__ inline Rule get_pair_rule(Pair AB) {
  return get_rule(get_fst(AB), get_snd(AB));
}

// Should we swap ports A and B before reducing this rule?
__device__ __host__ inline bool should_swap(Port A, Port B) {
  return get_tag(B) < get_tag(A);
}
// Gets a rule's priority
__device__ __host__ inline bool is_high_priority(Rule rule) {
  return (bool)((0b00011101 >> rule) & 1);
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

// Words
// -----

// Constructor and getters for SYM (operation selector)
__device__ __host__ inline Numb new_sym(u32 val) {
  return (val << 5) | TY_SYM;
}

__device__ __host__ inline u32 get_sym(Numb word) {
  return (word >> 5);
}

// Constructor and getters for U24 (unsigned 24-bit integer)
__device__ __host__ inline Numb new_u24(u32 val) {
  return (val << 5) | TY_U24;
}

__device__ __host__ inline u32 get_u24(Numb word) {
  return word >> 5;
}

// Constructor and getters for I24 (signed 24-bit integer)
__device__ __host__ inline Numb new_i24(i32 val) {
  return ((u32)val << 5) | TY_I24;
}

__device__ __host__ inline i32 get_i24(Numb word) {
  return ((i32)word) << 3 >> 8;
}

// Constructor and getters for F24 (24-bit float)
__device__ __host__ inline Numb new_f24(f32 val) {
  u32 bits = *(u32*)&val;
  u32 shifted_bits = bits >> 8;
  u32 lost_bits = bits & 0xFF;
  // round ties to even
  shifted_bits += (!isnan(val)) & ((lost_bits - ((lost_bits >> 7) & !shifted_bits)) >> 7);
  // ensure NaNs don't become infinities
  shifted_bits |= isnan(val);
  return (shifted_bits << 5) | TY_F24;
}

__device__ __host__ inline f32 get_f24(Numb word) {
  u32 bits = (word << 3) & 0xFFFFFF00;
  return *(f32*)&bits;
}

// Flip flag
__device__ __host__ inline Tag get_typ(Numb word) {
  return word & 0x1F;
}

__device__ __host__ inline bool is_num(Numb word) {
  return get_typ(word) >= TY_U24 && get_typ(word) <= TY_F24;
}

__device__ __host__ inline bool is_cast(Numb word) {
  return get_typ(word) == TY_SYM && get_sym(word) >= TY_U24 && get_sym(word) <= TY_F24;
}

// Cast a number to another type.
// The semantics are meant to spiritually resemble rust's numeric casts:
// - i24 <-> u24: is just reinterpretation of bits
// - f24  -> i24,
//   f24  -> u24: casts to the "closest" integer representing this float,
//                saturating if out of range and 0 if NaN
// - i24  -> f24,
//   u24  -> f24: casts to the "closest" float representing this integer.
__device__ __host__ inline Numb cast(Numb a, Numb b) {
  if (get_sym(a) == TY_U24 && get_typ(b) == TY_U24) return b;
  if (get_sym(a) == TY_U24 && get_typ(b) == TY_I24) {
    // reinterpret bits
    i32 val = get_i24(b);
    return new_u24(*(u32*) &val);
  }
  if (get_sym(a) == TY_U24 && get_typ(b) == TY_F24) {
    f32 val = get_f24(b);
    if (isnan(val)) {
      return new_u24(0);
    }
    return new_u24((u32) clamp(val, 0.0, 16777215));
  }

  if (get_sym(a) == TY_I24 && get_typ(b) == TY_U24) {
    // reinterpret bits
    u32 val = get_u24(b);
    return new_i24(*(i32*) &val);
  }
  if (get_sym(a) == TY_I24 && get_typ(b) == TY_I24) return b;
  if (get_sym(a) == TY_I24 && get_typ(b) == TY_F24) {
    f32 val = get_f24(b);
    if (isnan(val)) {
      return new_i24(0);
    }
    return new_i24((i32) clamp(val, -8388608.0, 8388607.0));
  }

  if (get_sym(a) == TY_F24 && get_typ(b) == TY_U24) return new_f24((f32) get_u24(b));
  if (get_sym(a) == TY_F24 && get_typ(b) == TY_I24) return new_f24((f32) get_i24(b));
  if (get_sym(a) == TY_F24 && get_typ(b) == TY_F24) return b;

  return new_u24(0);
}

// Partial application
__device__ __host__ inline Numb partial(Numb a, Numb b) {
  return (b & ~0x1F) | get_sym(a);
}

// Operate function
__device__ __host__ inline Numb operate(Numb a, Numb b) {
  Tag at = get_typ(a);
  Tag bt = get_typ(b);
  if (at == TY_SYM && bt == TY_SYM) {
    return new_u24(0);
  }
  if (is_cast(a) && is_num(b)) {
    return cast(a, b);
  }
  if (is_cast(b) && is_num(a)) {
    return cast(b, a);
  }
  if (at == TY_SYM && bt != TY_SYM) {
    return partial(a, b);
  }
  if (at != TY_SYM && bt == TY_SYM) {
    return partial(b, a);
  }
  if (at >= OP_ADD && bt >= OP_ADD) {
    return new_u24(0);
  }
  if (at < OP_ADD && bt < OP_ADD) {
    return new_u24(0);
  }
  Tag op, ty;
  Numb swp;
  if (at >= OP_ADD) {
    op = at; ty = bt;
  } else {
    op = bt; ty = at; swp = a; a = b; b = swp;
  }
  switch (ty) {
    case TY_U24: {
      u32 av = get_u24(a);
      u32 bv = get_u24(b);
      switch (op) {
        case OP_ADD: return new_u24(av + bv);
        case OP_SUB: return new_u24(av - bv);
        case FP_SUB: return new_u24(bv - av);
        case OP_MUL: return new_u24(av * bv);
        case OP_DIV: return new_u24(av / bv);
        case FP_DIV: return new_u24(bv / av);
        case OP_REM: return new_u24(av % bv);
        case FP_REM: return new_u24(bv % av);
        case OP_EQ:  return new_u24(av == bv);
        case OP_NEQ: return new_u24(av != bv);
        case OP_LT:  return new_u24(av < bv);
        case OP_GT:  return new_u24(av > bv);
        case OP_AND: return new_u24(av & bv);
        case OP_OR:  return new_u24(av | bv);
        case OP_XOR: return new_u24(av ^ bv);
        case OP_SHL: return new_u24(av << (bv & 31));
        case FP_SHL: return new_u24(bv << (av & 31));
        case OP_SHR: return new_u24(av >> (bv & 31));
        case FP_SHR: return new_u24(bv >> (av & 31));
        default:     return new_u24(0);
      }
    }
    case TY_I24: {
      i32 av = get_i24(a);
      i32 bv = get_i24(b);
      switch (op) {
        case OP_ADD: return new_i24(av + bv);
        case OP_SUB: return new_i24(av - bv);
        case FP_SUB: return new_i24(bv - av);
        case OP_MUL: return new_i24(av * bv);
        case OP_DIV: return new_i24(av / bv);
        case FP_DIV: return new_i24(bv / av);
        case OP_REM: return new_i24(av % bv);
        case FP_REM: return new_i24(bv % av);
        case OP_EQ:  return new_u24(av == bv);
        case OP_NEQ: return new_u24(av != bv);
        case OP_LT:  return new_u24(av < bv);
        case OP_GT:  return new_u24(av > bv);
        case OP_AND: return new_i24(av & bv);
        case OP_OR:  return new_i24(av | bv);
        case OP_XOR: return new_i24(av ^ bv);
        default:     return new_i24(0);
      }
    }
    case TY_F24: {
      float av = get_f24(a);
      float bv = get_f24(b);
      switch (op) {
        case OP_ADD: return new_f24(av + bv);
        case OP_SUB: return new_f24(av - bv);
        case FP_SUB: return new_f24(bv - av);
        case OP_MUL: return new_f24(av * bv);
        case OP_DIV: return new_f24(av / bv);
        case FP_DIV: return new_f24(bv / av);
        case OP_REM: return new_f24(fmodf(av, bv));
        case FP_REM: return new_f24(fmodf(bv, av));
        case OP_EQ:  return new_u24(av == bv);
        case OP_NEQ: return new_u24(av != bv);
        case OP_LT:  return new_u24(av < bv);
        case OP_GT:  return new_u24(av > bv);
        case OP_AND: return new_f24(atan2f(av, bv));
        case OP_OR:  return new_f24(logf(bv) / logf(av));
        case OP_XOR: return new_f24(powf(av, bv));
        case OP_SHL: return new_f24(sin(av + bv));
        case OP_SHR: return new_f24(tan(av + bv));
        default:     return new_f24(0);
      }
    }
    default: return new_u24(0);
  }
}

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

// Allocator
// ---------

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

// Linking
// -------

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

// Atomically Links `A ~ B`.
__device__ void link(Net* net, TM* tm, Port A, Port B) {
  Port INI_A = A;
  Port INI_B = B;

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

// Resources
// ---------

// Gets the necessary resources for an interaction.
__device__ bool get_resources(Net* net, TM* tm, u32 need_rbag, u32 need_node, u32 need_vars) {
  u32 got_rbag = min(RLEN - tm->rbag.lo_end, RLEN - tm->rbag.hi_end);
  u32 got_node;
  u32 got_vars;
  if (tm->mode != WORK) {
    debug("allocating need_rbag=%u need_node=%u need_vars=%u\n", need_rbag, need_node, need_vars);

    got_node = g_node_alloc(net, tm, need_node);
    got_vars = g_vars_alloc(net, tm, need_vars);
  } else {
    got_node = l_node_alloc(net, tm, need_node);
    got_vars = l_vars_alloc(net, tm, need_vars);
  }
  return got_rbag >= need_rbag && got_node >= need_node && got_vars >= need_vars;
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

// Declared here for use in call interactions.
static inline bool interact_eras(Net* net, TM* tm, Port a, Port b);

// The Call Interaction.
#ifdef COMPILED
///COMPILED_INTERACT_CALL///
#else
__device__ bool interact_eras(Net* net, TM* tm, Port a, Port b);
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

// RBag Save/Load
// --------------

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

// Creates a node.
__global__ void make_node(GNet* gnet, Tag tag, Port fst, Port snd, Port* ret) {
  if (GID() == 0) {
    Net net = vnet_new(gnet, NULL, gnet->turn);
    u32 loc = g_node_alloc_1(&net);
    node_create(&net, loc, new_pair(fst, snd));
    *ret = new_port(tag, loc);
  }
}

__global__ void inbetween(GNet* gnet) {
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

void gnet_boot_redex(GNet* gnet, Pair redex) {
  boot_redex<<<BPG, TPB>>>(gnet, redex);
}

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
    inbetween<<<1, 1>>>(gnet);
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

// Allocs and creates a node, returning its port.
Port gnet_make_node(GNet* gnet, Tag tag, Port fst, Port snd) {
  Port ret;
  Port* d_ret;
  cudaMalloc(&d_ret, sizeof(Port));
  make_node<<<1,1>>>(gnet, tag, fst, snd, d_ret);
  cudaMemcpy(&ret, d_ret, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);
  return ret;
}

// Book Loader
// -----------

bool book_load(Book* book, u32* buf) {
  // Reads defs_len
  book->defs_len = *buf++;

  // Parses each def
  for (u32 i = 0; i < book->defs_len; ++i) {
    // Reads fid
    u32 fid = *buf++;

    // Gets def
    Def* def = &book->defs_buf[fid];

    // Reads name
    memcpy(def->name, buf, 256);
    buf += 64;

    // Reads safe flag
    def->safe = *buf++;

    // Reads lengths
    def->rbag_len = *buf++;
    def->node_len = *buf++;
    def->vars_len = *buf++;

    if (def->rbag_len > L_NODE_LEN/TPB) {
      fprintf(stderr, "def '%s' has too many redexes: %u\n", def->name, def->rbag_len);
      return false;
    }

    if (def->node_len > L_NODE_LEN/TPB) {
      fprintf(stderr, "def '%s' has too many nodes: %u\n", def->name, def->node_len);
      return false;
    }

    // Reads root
    def->root = *buf++;

    // Reads rbag_buf
    memcpy(def->rbag_buf, buf, 8*def->rbag_len);
    buf += def->rbag_len * 2;

    // Reads node_buf
    memcpy(def->node_buf, buf, 8*def->node_len);
    buf += def->node_len * 2;
  }

  return true;
}

// Debug Printing
// --------------

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

__device__ u32 NODE_COUNT;
__device__ u32 VARS_COUNT;

__global__ void count_memory(GNet* gnet) {
  u32 node_count = 0;
  u32 vars_count = 0;
  for (u32 i = GID(); i < G_NODE_LEN; i += TPG) {
    if (gnet->node_buf[i] != 0) ++node_count;
    if (gnet->vars_buf[i] != 0) ++vars_count;
  }

  __shared__ u32 block_node_count;
  __shared__ u32 block_vars_count;

  if (TID() == 0) block_node_count = 0;
  if (TID() == 0) block_vars_count = 0;
  __syncthreads();

  atomicAdd(&block_node_count, node_count);
  atomicAdd(&block_vars_count, vars_count);
  __syncthreads();

  if (TID() == 0) atomicAdd(&NODE_COUNT, block_node_count);
  if (TID() == 0) atomicAdd(&VARS_COUNT, block_vars_count);
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

// Demos
// -----

  // stress_test 2^10 x 65536
  //static const u8 BOOK_BUF[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 20, 0, 0, 0, 9, 0, 0, 128, 44, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 4, 0, 0, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 111, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0};

  // stress_test 2^18 x 65536
  //static const u8 BOOK_BUF[] = {6, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 11, 18, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 102, 117, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 25, 0, 0, 0, 2, 0, 0, 0, 102, 117, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 4, 0, 0, 0, 11, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 102, 117, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 128, 20, 0, 0, 0, 9, 0, 0, 128, 44, 0, 0, 0, 13, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 3, 4, 0, 0, 38, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 108, 111, 111, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 41, 0, 0, 0, 5, 0, 0, 0, 108, 111, 111, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0};

  // bitonic_sort 2^20
  //static const u8 BOOK_BUF[] = {19, 0, 0, 0, 0, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 89, 0, 0, 0, 4, 0, 0, 0, 11, 18, 0, 0, 12, 0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 100, 111, 119, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 100, 111, 119, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 128, 60, 0, 0, 0, 25, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 45, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 32, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 40, 0, 0, 0, 100, 0, 0, 0, 24, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 102, 108, 111, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 4, 0, 0, 0, 102, 108, 111, 119, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 60, 0, 0, 0, 129, 0, 0, 0, 84, 0, 0, 0, 13, 0, 0, 0, 28, 0, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 35, 1, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 53, 0, 0, 0, 48, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 32, 0, 0, 0, 76, 0, 0, 0, 56, 0, 0, 0, 48, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 40, 0, 0, 0, 100, 0, 0, 0, 16, 0, 0, 0, 108, 0, 0, 0, 24, 0, 0, 0, 56, 0, 0, 0, 5, 0, 0, 0, 103, 101, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 103, 101, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 41, 0, 0, 128, 68, 0, 0, 0, 41, 0, 0, 128, 84, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 60, 0, 0, 0, 38, 0, 0, 0, 54, 0, 0, 0, 59, 2, 0, 0, 46, 0, 0, 0, 35, 1, 0, 0, 16, 0, 0, 0, 59, 2, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 92, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 7, 0, 0, 0, 109, 97, 105, 110, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 4, 0, 0, 0, 11, 18, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 109, 97, 105, 110, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 4, 0, 0, 0, 11, 18, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 20, 0, 0, 0, 57, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 115, 111, 114, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 60, 0, 0, 0, 20, 0, 0, 0, 44, 0, 0, 0, 28, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 68, 0, 0, 0, 8, 0, 0, 0, 24, 0, 0, 0, 10, 0, 0, 0, 115, 111, 114, 116, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 25, 0, 0, 0, 60, 0, 0, 0, 73, 0, 0, 128, 92, 0, 0, 0, 73, 0, 0, 128, 116, 0, 0, 0, 13, 0, 0, 0, 36, 0, 0, 0, 22, 0, 0, 0, 29, 0, 0, 0, 35, 1, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 0, 40, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 11, 0, 0, 0, 108, 0, 0, 0, 24, 0, 0, 0, 56, 0, 0, 0, 16, 0, 0, 0, 124, 0, 0, 0, 11, 1, 0, 0, 132, 0, 0, 0, 32, 0, 0, 0, 64, 0, 0, 0, 11, 0, 0, 0, 115, 117, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 115, 117, 109, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 89, 0, 0, 128, 36, 0, 0, 0, 89, 0, 0, 128, 68, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 32, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 54, 0, 0, 0, 3, 4, 0, 0, 62, 0, 0, 0, 40, 0, 0, 0, 32, 0, 0, 0, 8, 0, 0, 0, 76, 0, 0, 0, 24, 0, 0, 0, 40, 0, 0, 0, 13, 0, 0, 0, 115, 119, 97, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 44, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 113, 0, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 52, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 14, 0, 0, 0, 115, 119, 97, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 115, 119, 97, 112, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 119, 97, 114, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 52, 0, 0, 0, 20, 0, 0, 0, 28, 0, 0, 0, 137, 0, 0, 0, 145, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 8, 0, 0, 0, 44, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 16, 0, 0, 0, 60, 0, 0, 0, 8, 0, 0, 0, 68, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 17, 0, 0, 0, 119, 97, 114, 112, 95, 95, 67, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 105, 0, 0, 0, 76, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0, 0, 0, 52, 0, 0, 0, 38, 0, 0, 0, 24, 0, 0, 0, 3, 15, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 62, 0, 0, 0, 40, 0, 0, 0, 3, 18, 0, 0, 70, 0, 0, 0, 16, 0, 0, 0, 32, 0, 0, 0, 32, 0, 0, 0, 84, 0, 0, 0, 24, 0, 0, 0, 92, 0, 0, 0, 8, 0, 0, 0, 40, 0, 0, 0, 18, 0, 0, 0, 119, 97, 114, 112, 95, 95, 67, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 21, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 129, 0, 0, 128, 92, 0, 0, 0, 129, 0, 0, 128, 132, 0, 0, 0, 13, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 36, 0, 0, 0, 16, 0, 0, 0, 24, 0, 0, 0, 44, 0, 0, 0, 52, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 61, 0, 0, 0, 68, 0, 0, 0, 48, 0, 0, 0, 56, 0, 0, 0, 76, 0, 0, 0, 84, 0, 0, 0, 64, 0, 0, 0, 72, 0, 0, 0, 80, 0, 0, 0, 88, 0, 0, 0, 8, 0, 0, 0, 100, 0, 0, 0, 56, 0, 0, 0, 108, 0, 0, 0, 40, 0, 0, 0, 116, 0, 0, 0, 24, 0, 0, 0, 124, 0, 0, 0, 72, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 140, 0, 0, 0, 48, 0, 0, 0, 148, 0, 0, 0, 32, 0, 0, 0, 156, 0, 0, 0, 16, 0, 0, 0, 164, 0, 0, 0, 64, 0, 0, 0, 80, 0, 0, 0};

//COMPILED_BOOK_BUF//

// Main
// ----

#ifdef IO
void do_run_io(GNet* gnet, Book* book, Port port);
#endif

extern "C" void hvm_cu(u32* book_buffer) {
  // Loads the Book
  Book* book = (Book*)malloc(sizeof(Book));
  if (book_buffer) {
    if (!book_load(book, (u32*)book_buffer)) {
      fprintf(stderr, "failed to load book\n");

      return;
    }
    cudaMemcpyToSymbol(BOOK, book, sizeof(Book));
  }

  // Configures Shared Memory Size
  cudaFuncSetAttribute(evaluator, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(LNet));

  // Creates a new GNet
  GNet* gnet = gnet_create();

  // Start the timer
  clock_t start = clock();

  // Boots root redex, to expand @main
  gnet_boot_redex(gnet, new_pair(new_port(REF, 0), ROOT));

  #ifdef IO
  do_run_io(gnet, book, ROOT);
  #else
  gnet_normalize(gnet);
  #endif

  cudaDeviceSynchronize();

  // Stops the timer
  clock_t end = clock();
  double duration = ((double)(end - start)) / CLOCKS_PER_SEC;

  // Prints the result
  print_result<<<1,1>>>(gnet);

  // Reports errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernels. Error code: %s.\n", cudaGetErrorString(err));
    if (err == cudaErrorInvalidConfiguration) {
      fprintf(stderr, "Note: for now, HVM-CUDA requires a GPU with at least 128 KB of L1 cache per SM.\n");
    }
    exit(EXIT_FAILURE);
  }

  // Prints entire memdump
  //{
    //// Allocate host memory for the net
    //GNet *h_gnet = (GNet*)malloc(sizeof(GNet));

    //// Copy the net from device to host
    //cudaMemcpy(h_gnet, gnet, sizeof(GNet), cudaMemcpyDeviceToHost);

    //// Create a Net view of the host GNet
    //Net net;
    //net.g_node_buf = h_gnet->node_buf;
    //net.g_vars_buf = h_gnet->vars_buf;

    //// Print the net
    //print_net(&net, L_NODE_LEN, G_NODE_LEN);

    //// Free host memory
    //free(h_gnet);
  //}

  // Gets interaction count
  //cudaMemcpy(&itrs, &gnet->itrs, sizeof(u64), cudaMemcpyDeviceToHost);

  // Prints interactions, time and MIPS
  printf("- ITRS: %llu\n", gnet_get_itrs(gnet));
  printf("- LEAK: %llu\n", gnet_get_leak(gnet));
  printf("- TIME: %.2fs\n", duration);
  printf("- MIPS: %.2f\n", (double)gnet_get_itrs(gnet) / duration / 1000000.0);
}

#ifdef WITH_MAIN
int main() {
  hvm_cu((u32*)BOOK_BUF);
  return 0;
}
#endif
