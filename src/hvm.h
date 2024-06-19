#ifndef hvm_h_INCLUDED
#define hvm_h_INCLUDED

#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

// Booleans
#define TRUE  1
#define FALSE 0

// Integers
// --------

typedef uint8_t bool;

typedef  uint8_t  u8;
typedef uint16_t u16;
typedef  int32_t i32;
typedef uint32_t u32;
typedef uint64_t u64;
typedef    float f32;
typedef   double f64;

typedef _Atomic(u8) a8;
typedef _Atomic(u16) a16;
typedef _Atomic(u32) a32;
typedef _Atomic(u64) a64;

// Configuration
// -------------

// Threads per CPU
#ifndef TPC_L2
#define TPC_L2 0
#endif
#define TPC (1ul << TPC_L2)

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
static const f32 U24_MAX = (f32) (1 << 24) - 1;
static const f32 U24_MIN = 0.0;
static const f32 I24_MAX = (f32) (1 << 23) - 1;
static const f32 I24_MIN = (f32) (i32) ((-1u) << 23);
#define TY_SYM 0x00
#define TY_U24 0x01
#define TY_I24 0x02
#define TY_F24 0x03
#define OP_ADD 0x04
#define OP_SUB 0x05
#define FP_SUB 0x06
#define OP_MUL 0x07
#define OP_DIV 0x08
#define FP_DIV 0x09
#define OP_REM 0x0A
#define FP_REM 0x0B
#define OP_EQ  0x0C
#define OP_NEQ 0x0D
#define OP_LT  0x0E
#define OP_GT  0x0F
#define OP_AND 0x10
#define OP_OR  0x11
#define OP_XOR 0x12
#define OP_SHL 0x13
#define FP_SHL 0x14
#define OP_SHR 0x15
#define FP_SHR 0x16

// Constants
#define FREE 0x00000000
#define ROOT 0xFFFFFFF8
#define NONE 0xFFFFFFFF

// Cache Padding
#define CACHE_PAD 64

// Global Net
#define HLEN (1ul << 16) // max 16k high-priority redexes
#define RLEN (1ul << 24) // max 16m low-priority redexes
#define G_NODE_LEN (1ul << 29) // max 536m nodes
#define G_VARS_LEN (1ul << 29) // max 536m vars
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
  char name[256];
  bool safe;
  u32  rbag_len;
  u32  node_len;
  u32  vars_len;
  Port root;
  Pair rbag_buf[0xFFF];
  Pair node_buf[0xFFF];
} Def;

typedef struct Book Book;

// A Foreign Function
typedef struct {
  char name[256];
  Port (*func)(Net*, Book*, Port);
} FFn;

// Book of Definitions
typedef struct Book {
  u32 defs_len;
  Def defs_buf[0x4000];
  u32 ffns_len;
  FFn ffns_buf[0x4000];
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
  u32  nloc[0xFFF]; // node allocation indices
  u32  vloc[0xFFF]; // vars allocation indices
  Pair hbag_buf[HLEN]; // high-priority redexes
} TM;

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
void pretty_print_numb(Numb word);
void pretty_print_port(Net* net, Book* book, Port port);
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

static inline u32 min(u32 a, u32 b) {
  return (a < b) ? a : b;
}

static inline f32 clamp(f32 x, f32 min, f32 max) {
  const f32 t = x < min ? min : x;
  return (t > max) ? max : t;
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
  return (val << 5) | TY_SYM;
}

static inline u32 get_sym(Numb word) {
  return (word >> 5);
}

// Constructor and getters for U24 (unsigned 24-bit integer)
static inline Numb new_u24(u32 val) {
  return (val << 5) | TY_U24;
}

static inline u32 get_u24(Numb word) {
  return word >> 5;
}

// Constructor and getters for I24 (signed 24-bit integer)
static inline Numb new_i24(i32 val) {
  return ((u32)val << 5) | TY_I24;
}

static inline i32 get_i24(Numb word) {
  return ((i32)word) << 3 >> 8;
}

// Constructor and getters for F24 (24-bit float)
static inline Numb new_f24(float val) {
  u32 bits = *(u32*)&val;
  u32 shifted_bits = bits >> 8;
  u32 lost_bits = bits & 0xFF;
  // round ties to even
  shifted_bits += (!isnan(val)) & ((lost_bits - ((lost_bits >> 7) & !shifted_bits)) >> 7);
  // ensure NaNs don't become infinities
  shifted_bits |= isnan(val);
  return (shifted_bits << 5) | TY_F24;
}

static inline float get_f24(Numb word) {
  u32 bits = (word << 3) & 0xFFFFFF00;
  return *(float*)&bits;
}

// Flip flag
static inline Tag get_typ(Numb word) {
  return word & 0x1F;
}

static inline bool is_num(Numb word) {
  return get_typ(word) >= TY_U24 && get_typ(word) <= TY_F24;
}

static inline bool is_cast(Numb word) {
  return get_typ(word) == TY_SYM && get_sym(word) >= TY_U24 && get_sym(word) <= TY_F24;
}

// Partial application
static inline Numb partial(Numb a, Numb b) {
  return (b & ~0x1F) | get_sym(a);
}

// Readback
// ---------

// Readback: Tuples
typedef struct Tup {
  u32  elem_len;
  Port elem_buf[8];
} Tup;

extern Tup readback_tup(Net* net, Book* book, Port port, u32 size);

typedef struct Str {
  u32  text_len;
  char text_buf[256];
} Str;

extern Str readback_str(Net* net, Book* book, Port port);

typedef struct Bytes {
  u32  len;
  char *buf;
} Bytes;

extern Bytes readback_bytes(Net* net, Book* book, Port port);

extern Port inject_bytes(Net* net, Bytes *bytes);

#endif // hvm_h_INCLUDED
