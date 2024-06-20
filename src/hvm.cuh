#ifndef hvm_cuh_INCLUDED
#define hvm_cuh_INCLUDED

#include <math.h>
#include <stdint.h>

// Types
// -----

typedef uint8_t bool;
typedef  uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;
typedef  int32_t i32;
typedef    float f32;
typedef   double f64;

// Local Types
typedef u8  Tag;  // Tag  ::= 3-bit (rounded up to u8)
typedef u32 Val;  // Val  ::= 29-bit (rounded up to u32)
typedef u32 Port; // Port ::= Tag + Val (fits a u32)
typedef u64 Pair; // Pair ::= Port + Port (fits a u64)

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

// Numbers
static const f32 U24_MAX = (f32) (1 << 24) - 1;
static const f32 U24_MIN = 0.0;
static const f32 I24_MAX = (f32) (1 << 23) - 1;
static const f32 I24_MIN = (f32) (i32) ((-1u) << 23);
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

typedef struct GNet GNet;

// Debugger
// --------

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

// Reads a tuple of `size` elements from `port`.
// Tuples are con nodes nested to the right auxilliary port,
// For example, `(CON a (CON b (CON c)))` is a 3-tuple (a, b, c).
extern Tup gnet_readback_tup(GNet* gnet, Port port, u32 size);

typedef struct Str {
  u32  text_len;
  char text_buf[256];
} Str;

// Reads a constructor-encoded string (of length at most 255 characters),
// into a null-terminated `Str`.
extern Str gnet_readback_str(GNet* gnet, Port port);

typedef struct Bytes {
  u32  len;
  char *buf;
} Bytes;

// Reads a constructor-encoded string (of length at most 256 characters),
// into a `Bytes`. The returned `Bytes` is not null terminated.
extern Bytes gnet_readback_bytes(GNet* net, Port port);

// Creates a construtor-encoded string of arbitrary length from the
// provided `bytes`. This string can be consumed on the HVM-side. This
// will return an `ERA` if nodes cannot be allocated.
extern Port gnet_inject_bytes(GNet* net, Bytes *bytes);

#endif // hvm_cuh_INCLUDED
