// This header file defines the interface between
// IO platforms and (generated) program runtimes

// ///////////////////
// Common Dependencies
// ///////////////////

#include <stdint.h>
#include <stdbool.h>

// /////////////////////////////
// Common constants & data types
// /////////////////////////////

// Types
// -----

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

// Consts
// ------

#define U64_PER_KB (0x80)
#define U64_PER_MB (0x20000)
#define U64_PER_GB (0x8000000)


// Terms
// -----
// HVM's runtime stores terms in a 64-bit memory. Each element is a Link, which
// usually points to a constructor. It stores a Tag representing the ctor's
// variant, and possibly a position on the memory. So, for example, `Ptr ptr =
// APP * TAG | 137` creates a pointer to an app node stored on position 137.
// Some links deal with variables: DP0, DP1, VAR, ARG and ERA.  The OP2 link
// represents a numeric operation, and NUM and FLO links represent unboxed nums.

typedef u64 Ptr;

#define VAL ((u64) 1)
#define EXT ((u64) 0x100000000)
#define ARI ((u64) 0x100000000000000)
#define TAG ((u64) 0x1000000000000000)

#define NUM_MASK ((u64) 0xFFFFFFFFFFFFFFF)

#define DP0 (0x0) // points to the dup node that binds this variable (left side)
#define DP1 (0x1) // points to the dup node that binds this variable (right side)
#define VAR (0x2) // points to the λ that binds this variable
#define ARG (0x3) // points to the occurrence of a bound variable a linear argument
#define ERA (0x4) // signals that a binder doesn't use its bound variable
#define LAM (0x5) // arity = 2
#define APP (0x6) // arity = 2
#define SUP (0x7) // arity = 2 // TODO: rename to SUP
#define CTR (0x8) // arity = user defined
#define FUN (0x9) // arity = user defined
#define OP2 (0xA) // arity = 2
#define NUM (0xB) // arity = 0 (unboxed)
#define FLO (0xC) // arity = 0 (unboxed)
#define NIL (0xF) // not used

#define ADD (0x0)
#define SUB (0x1)
#define MUL (0x2)
#define DIV (0x3)
#define MOD (0x4)
#define AND (0x5)
#define OR  (0x6)
#define XOR (0x7)
#define SHL (0x8)
#define SHR (0x9)
#define LTN (0xA)
#define LTE (0xB)
#define EQL (0xC)
#define GTE (0xD)
#define GTN (0xE)
#define NEQ (0xF)

// ///////////////////////////
// Provided by program runtime
// ///////////////////////////

extern u64 ffi_cost;
extern u64 ffi_size;

// ///////////////////////
// Provided by IO platform
// ///////////////////////

// Called once on program start
void* io_setup();

// Called when the program wants to do IO
// Return true  => "please run again"
// Return false => "please halt"
bool io_step (void* state, Ptr* node);
