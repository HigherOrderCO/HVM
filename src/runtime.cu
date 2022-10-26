// Compile with: `nvcc -arch=sm_70 -O3 runtime.cu -o runtime`

// Adjust these params to test and benchmark:
#define W 32   // threads per block
#define H 128  // blocks per grid
#define D 22   // depth of parallel recursion
#define L 1024 // length of sequential recursion

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned long long Ptr;
typedef Ptr* Heap;

// gets current time in seconds
double now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

__device__ u64 atomicLoad(u64* addr) {
  const volatile u64 *vaddr = addr; // volatile to bypass cache
  //__threadfence();
  const u64 value = *vaddr;
  //__threadfence();
  return value; 
}

__device__ void atomicStore(u64 *addr, u64 value) {
  volatile u64 *vaddr = addr; // volatile to bypass cache
  //__threadfence();
  *vaddr = value;
}

__device__ void atomicStore16(u16 *addr, u16 value) {
  volatile u16 *vaddr = addr; // volatile to bypass cache
  //__threadfence(); // fence to ensure that previous non-atomic stores are visible to other threads
  *vaddr = value;
}

// Global Memory Layout
// --------------------

// - LV |  2^18 words | 0x0000_0000 .. 0x0030_0000 | Local Vars
// - RB |  2^24 words | 0x0030_0000 .. 0x0130_0000 | Redex Bag
// - VQ |  2^24 words | 0x0130_0000 .. 0x0230_0000 | Visit Queue
// - CS |  2^24 words | 0x0230_0000 .. 0x0330_0000 | Collect Stack
// - LK |  2^28 bytes | 0x0330_0000 .. ?           | Lock Buffer | TODO: fix, needs 16-bit atomicCAS
// - NB | ~2^28 words |           ? .. ?           | Node Buffer | TODO: fix, needs 16-bit atomicCAS
// INDEX(tid.used) = 16 * (tid + 0x0000) + 0; // total used memory
// INDEX(tid.aloc) = 16 * (tid + 0x0000) + 1; // cur allocation index
// INDEX(tid.coll) = 16 * (tid + 0x0000) + 2; // collection stack index
// INDEX(tid.dups) = 16 * (tid + 0x0000) + 3; // total generated dups
// INDEX(tid.cost) = 16 * (tid + 0x0000) + 4; // rewrite rule count
// INDEX(tid.next) = 16 * (tid + 0x0000) + 5; // redex bag next index
// INDEX(tid.endq) = 16 * (tid + 0x0000) + 6; // visit queue end index
// INDEX(tid.begq) = 16 * (tid + 0x1000); // visit queue beg index
// INDEX(glo.stop) = 0x0020_0000
// Total: ~4.2 GB

#define TIDS     (W * H)
#define MAX_TIDS (65536)
#define RB_EXIT  (0xFFFFFFull)
#define RB_SIZE  (0x1000000ull)
#define VQ_SIZE  (0x1000000ull)
#define HD_SIZE  (0x03300000ull)
#define NB_SIZE  (0x20000000ull) // 0x04000000ull or 0x20000000ull
#define LK_SIZE  (NB_SIZE >> 2)

#define MINI_SIZE (HD_SIZE + LK_SIZE + NB_SIZE / 16)
#define HEAP_SIZE (HD_SIZE + LK_SIZE + NB_SIZE)

// Returns &tid.used
__device__ __host__ u64* ref_used(Heap heap, u64 tid) {
  return &heap[16ull * tid + 0ull];
}

// Returns &tid.aloc
__device__ __host__ u64* ref_aloc(Heap heap, u64 tid) {
  return &heap[16ull * tid + 1ull];
}

// Returns &tid.coll
__device__ __host__ u64* ref_coll(Heap heap, u64 tid) {
  return &heap[16ull * tid + 2ull];
}

// Returns &tid.dups
__device__ __host__ u64* ref_dups(Heap heap, u64 tid) {
  return &heap[16ull * tid + 3ull];
}

// Returns &tid.cost
__device__ __host__ u64* ref_cost(Heap heap, u64 tid) {
  return &heap[16ull * tid + 4ull];
}

// Returns &tid.next
__device__ __host__ u64* ref_next(Heap heap, u64 tid) {
  return &heap[16ull * tid + 5ull];
}

// Returns &tid.begq
__device__ __host__ u64* ref_endq(Heap heap, u64 tid) {
  return &heap[16ull * tid + 6ull];
}

// Returns &tid.endq
__device__ __host__ u64* ref_begq(Heap heap, u64 tid) {
  return &heap[0x00100000ull + 16ull * tid];
}

// Returns &glo.stop
__device__ __host__ u64* ref_stop(Heap heap) {
  return &heap[0x00200000ull];
}

// Returns &glo.maxv
__device__ __host__ u64* ref_maxv(Heap heap) {
  return &heap[0x00200100ull];
}

// Returns &glo.maxr
__device__ __host__ u64* ref_maxr(Heap heap) {
  return &heap[0x00200200ull];
}

// Returns &glo.maxa
__device__ __host__ u64* ref_maxa(Heap heap) {
  return &heap[0x00200300ull];
}

// Returns &rb.data[index]
__device__ __host__ u64* ref_rb_data(Heap heap, u64 index) {
  return &heap[0x00300000ull + index];
}

// Returns &vq.data[index]
__device__ __host__ u64* ref_vq_data(Heap heap, u64 index) {
  return &heap[0x01300000ull + index];
}

// Returns &cs.data[index]
__device__ __host__ u64* ref_cs_data(Heap heap, u64 index) {
  return &heap[0x02300000ull + index];
}

// Returns &lk.data[index]
__device__ __host__ u16* ref_lk_data(Heap heap, u64 index) {
  return ((u16*)(heap + HD_SIZE)) + index;
}

// Returns &nb.data[index]
__device__ __host__ u64* ref_nb_data(Heap heap, u64 index) {
  return &heap[HD_SIZE + LK_SIZE + index];
}

// Visit Queue
// -----------

// Returns the index of a tid's area on the VisitQueue
__device__ __host__ u64 vq_area_index(u64 tid) {
  return VQ_SIZE / MAX_TIDS * tid;
}

// TODO: must use this to check bounds
// Returns the length of a tid's area on the VisitQueue
__device__ __host__ u64 vq_area_length(u64 tid) {
  return VQ_SIZE / MAX_TIDS;
}

__device__ __host__ u64 new_visit(u64 host, u64 cont) {
  return (host << 32) | cont;
}

__device__ __host__ u64 get_visit_host(u64 visit) {
  return visit >> 32;
}

__device__ __host__ u64 get_visit_cont(u64 visit) {
  return visit & 0xFFFFFFFFull;
}

__device__ void vq_push(Heap heap, u64 tid, u64 value) {
  u64* endq_ref = ref_endq(heap, tid);
  u64 index = atomicAdd(endq_ref, 1);
  atomicStore(ref_vq_data(heap, vq_area_index(tid) + index), value);
}

__device__ u64 vq_pop(Heap heap, u64 tid) {
  while (1) {
    u64* endq_ref = ref_endq(heap, tid);
    u64 endq = *endq_ref;
    if (endq > 0ull) {
      *endq_ref -= 1ull;
      u64 visit = atomicExch(ref_vq_data(heap, vq_area_index(tid) + endq - 1ull), 0ull);
      if (visit == 0ull) {
        continue;
      } else {
        atomicMin(ref_begq(heap, tid), endq - 1ull);
        return visit;
      }
    } else {
      atomicStore(ref_begq(heap, tid), 0ull);
      return -1ull;
    }
  }
}

__device__ u64 vq_steal(Heap heap, u64 tid) {
  u64* begq_ref = ref_begq(heap, tid);
  u64 index = atomicLoad(begq_ref);
  u64* visit_ref = ref_vq_data(heap, vq_area_index(tid) + index);
  u64 visit = atomicLoad(visit_ref);
  if (visit != 0ull) {
    u64 old = atomicCAS(visit_ref, visit, 0ull);
    if (old == visit) { 
      atomicAdd(begq_ref, 1ull);
      return visit;
    }
  }
  return -1ull;
}

// Redex Bag
// ---------

__device__ __host__ u64 new_redex(u64 host, u64 cont, u64 left) {
  return (host << 32) | (cont << 8) | left;
}

__device__ __host__ u64 get_redex_host(u64 redex) {
  return redex >> 32;
}

__device__ __host__ u64 get_redex_cont(u64 redex) {
  return (redex >> 8) & 0xFFFFFF;
}

__device__ __host__ u64 get_redex_left(u64 redex) {
  return redex & 0xFF;
}

// Returns the index of a tid's area on the RedexBag
__device__ __host__ u64 rb_area_index(u64 tid) {
  return RB_SIZE / MAX_TIDS * tid;
}

// Returns the length of a tid's area on the RedexBag
__device__ __host__ u64 rb_area_length(u64 tid) {
  return RB_SIZE / MAX_TIDS - 2; // FIXME: 2 or 1?
}

__device__ u64 rb_insert(Heap heap, u64 tid, u64 redex) {
  u64* next_ref = ref_next(heap, tid);
  while (1) {
    u64 index = (*next_ref + rb_area_index(tid)) % (RB_SIZE - 2);
    *next_ref += 1;
    u64* data_ref = ref_rb_data(heap, index);
    u64 old = atomicCAS(data_ref, 0ull, redex);
    if (old == 0ull) {
      return index;
    }
  }
}

__device__ u64 rb_complete(Heap heap, u64 tid, u64 index) {
  u64* data_ref = ref_rb_data(heap, index);
  u64 redex = atomicAdd(data_ref, -1ull);
  if (get_redex_left(redex) == 1ull) {
    atomicStore(data_ref, 0ull);
    return redex;
  } else {
    return -1ull;
  }
}

// Pointers
// --------

#define VAL (1ull)
#define EXT (0x100000000ull)
#define TAG (0x1000000000000000ull)

#define NUM_MASK (0xFFFFFFFFFFFFFFFull)

#define DP0 (0x0ull) // points to the dup node that binds this variable (left side)
#define DP1 (0x1ull) // points to the dup node that binds this variable (right side)
#define VAR (0x2ull) // points to the λ that binds this variable
#define ARG (0x3ull) // points to the occurrence of a bound variable a linear argument
#define ERA (0x4ull) // signals that a binder doesn't use its bound variable
#define LAM (0x5ull) // arity = 2
#define APP (0x6ull) // arity = 2
#define SUP (0x7ull) // arity = 2 // TODO: rename to SUP
#define CTR (0x8ull) // arity = user defined
#define FUN (0x9ull) // arity = user defined
#define OP2 (0xAull) // arity = 2
#define NUM (0xBull) // arity = 0 (unboxed)
#define FLO (0xCull) // arity = 0 (unboxed)
#define NIL (0xFull) // not used

#define ADD (0x0ull)
#define SUB (0x1ull)
#define MUL (0x2ull)
#define DIV (0x3ull)
#define MOD (0x4ull)
#define AND (0x5ull)
#define OR  (0x6ull)
#define XOR (0x7ull)
#define SHL (0x8ull)
#define SHR (0x9ull)
#define LTN (0xAull)
#define LTE (0xBull)
#define EQL (0xCull)
#define GTE (0xDull)
#define GTN (0xEull)
#define NEQ (0xFull)

// Test HVM function used on benchmarks:
// (Foo 0 0) = 1
// (Foo 0 m) = (Foo 0 (- m 1))
// (Foo n m) = (+ (Foo (- n 1) m) (Foo (- n 1) m))
#define FOO (0x100ull)

__device__ __host__ Ptr Var(u64 pos) {
  return (VAR * TAG) | pos;
}

__device__ __host__ Ptr Dp0(u64 col, u64 pos) {
  return (DP0 * TAG) | (col * EXT) | pos;
}

__device__ __host__ Ptr Dp1(u64 col, u64 pos) {
  return (DP1 * TAG) | (col * EXT) | pos;
}

__device__ __host__ Ptr Arg(u64 pos) {
  return (ARG * TAG) | pos;
}

__device__ __host__ Ptr Era(void) {
  return (ERA * TAG);
}

__device__ __host__ Ptr Lam(u64 pos) {
  return (LAM * TAG) | pos;
}

__device__ __host__ Ptr App(u64 pos) {
  return (APP * TAG) | pos;
}

__device__ __host__ Ptr Sup(u64 col, u64 pos) {
  return (SUP * TAG) | (col * EXT) | pos;
}

__device__ __host__ Ptr Op2(u64 ope, u64 pos) {
  return (OP2 * TAG) | (ope * EXT) | pos;
}

__device__ __host__ Ptr Num(u64 val) {
  return (NUM * TAG) | (val & NUM_MASK);
}

__device__ __host__ Ptr Ctr(u64 fun, u64 pos) {
  return (CTR * TAG) | (fun * EXT) | pos;
}

__device__ __host__ Ptr Fun(u64 fun, u64 pos) {
  return (FUN * TAG) | (fun * EXT) | pos;
}

__device__ __host__ u64 get_tag(Ptr ptr) {
  return ptr / TAG;
}

__device__ __host__ u64 get_ext(Ptr ptr) {
  return (ptr / EXT) & 0xFFFFFFFull;
}

__device__ __host__ u64 get_val(Ptr ptr) {
  return ptr & 0xFFFFFFFFull;
}

__device__ __host__ u64 get_num(Ptr ptr) {
  return ptr & 0xFFFFFFFFFFFFFFFull;
}

__device__ __host__ u64 get_loc(Ptr ptr, u64 arg) {
  return get_val(ptr) + arg;
}

__device__ Ptr load_ptr(Heap heap, u64 loc) {
  return atomicLoad(ref_nb_data(heap, loc));
}

__device__ Ptr load_arg(Heap heap, Ptr term, u64 arg) {
  return load_ptr(heap, get_loc(term, arg));
}

__device__ Ptr take_ptr(Heap heap, u64 loc) {
  return atomicExch(ref_nb_data(heap, loc), 0ull);
}

__device__ Ptr take_arg(Heap heap, Ptr term, u64 arg) {
  return take_ptr(heap, get_loc(term, arg));
}

__device__ void inc_cost(Heap heap, u64 tid) {
  *ref_cost(heap, tid) += 1;
}

__device__ u64 gen_dup(Heap heap, u64 tid) {
  u64* dups_ref = ref_dups(heap, tid);
  *dups_ref += 1;
  return *dups_ref & 0xFFFFFFF;
}

__device__ Ptr link(Heap heap, u64 loc, Ptr ptr) {
  atomicStore(ref_nb_data(heap, loc), ptr);
  if (get_tag(ptr) <= VAR) {
    u64 arg_loc = get_loc(ptr, get_tag(ptr) & 0x01ull);
    atomicStore(ref_nb_data(heap, arg_loc), Arg(loc));
  }
  return ptr;
}

// Program
// -------

__device__ __host__ u64 arity_of(u64 term) {
  switch (get_ext(term)) {
    case FOO: return 2;
    default: return 0;
  }
}

__device__ __host__ const char* name_of(u64 term) {
  switch (get_ext(term)) {
    case FOO: return "Foo";
    default: return "?";
  }
}

// Allocation
// ----------

// Returns the index of a tid's area on the RedexBag
__device__ __host__ u64 nb_area_index(u64 tid) {
  return NB_SIZE / MAX_TIDS * tid;
}

// Returns the length of a tid's area on the RedexBag
__device__ __host__ u64 nb_area_length(u64 tid) {
  return NB_SIZE / MAX_TIDS;
}
__device__ u64 alloc(Heap heap, u64 tid, u64 arity) {
  if (arity == 0ull) {
    return 0ull;
  } else {
    u64 length = 0ull;
    while (1) {
      u64* aloc_ref = ref_aloc(heap, tid);
      u64  aloc_max = nb_area_length(tid);
      // Loads value on cursor
      u64 val = atomicLoad(ref_nb_data(heap, nb_area_index(tid) + *aloc_ref));
      // If it is empty, increment length
      if (val == 0ull) {
        length += 1ull;
      // Otherwise, reset length
      } else {
        length = 0ull;
      }
      // Moves cursor right
      *aloc_ref += 1ull;
      // If it is out of bounds, warp around
      if (*aloc_ref >= aloc_max) {
        length = 0ull;
        *aloc_ref = 0;
        //printf("[%04llu] loop\n", tid);
      }
      // If length equals arity, allocate that space
      if (length == arity) {
        //printf("[%04llu] alloc %llu at %llx\n", tid, arity, nb_area_index(tid) + *aloc_ref - length);
        return nb_area_index(tid) + *aloc_ref - length;
      }
    }
  }
}

__device__ u64 free(Heap heap, u64 loc, u64 arity) {
  for (u64 i = 0ull; i < arity; ++i) {
    atomicStore(ref_nb_data(heap, loc + i), 0ull);
  }
}

// Garbage Collection
// ------------------

__device__ void collect_stack_push(Heap heap, u64 tid, u64 val) {
  *ref_cs_data(heap, (*ref_coll(heap, tid))++) = val;
}

__device__ u64 collect_stack_pop(Heap heap, u64 tid) {
  if (*ref_coll(heap, tid) > 0) {
    return *ref_cs_data(heap, --(*ref_coll(heap, tid)));
  } else {
    return -1ull;
  }
}

__device__ void collect(Heap heap, u64 tid, Ptr term) {
  u64 next = term;
  while (1) {
    u64 term = next;
    switch (get_tag(term)) {
      case DP0: {
        link(heap, get_loc(term, 0ull), Era());
        break;
      }
      case DP1: {
        link(heap, get_loc(term, 1ull), Era());
        break;
      }
      case VAR: {
        link(heap, get_loc(term, 0ull), Era());
        break;
      }
      case LAM: {
        next = take_arg(heap, term, 1ull);
        free(heap, get_loc(term, 0ull), 2ull);
        continue;
      }
      case APP: {
        collect_stack_push(heap, tid, take_arg(heap, term, 0ull));
        next = take_arg(heap, term, 1ull);
        free(heap, get_loc(term, 0ull), 2ull);
        continue;
      }
      case SUP: {
        collect_stack_push(heap, tid, take_arg(heap, term, 0ull));
        next = take_arg(heap, term, 1ull);
        free(heap, get_loc(term, 0ull), 2ull);
        continue;
      }
      case OP2: {
        collect_stack_push(heap, tid, take_arg(heap, term, 0ull));
        next = take_arg(heap, term, 1ull);
        free(heap, get_loc(term, 0ull), 2ull);
        continue;
      }
      case NUM: {
        break;
      }
      case CTR:
      case FUN: {
        u64 arity = arity_of(term);
        for (u64 i = 0ull; i < arity; ++i) {
          if (i < arity - 1) {
            collect_stack_push(heap, tid, take_arg(heap, term, i));
          } else {
            next = take_arg(heap, term, i);
          }
        }
        free(heap, get_loc(term, 0), arity);
        if (arity > 0) {
          continue;
        }
        break;
      }
    }
    u64 got = collect_stack_pop(heap, tid);
    if (got != -1ull) {
      next = got;
    } else {
      break;
    }
  }
}

// Substitution
// ------------

__device__ bool atomic_relink(Heap heap, u64 loc, Ptr old, Ptr neo) {
  u64 got = atomicCAS(ref_nb_data(heap, loc), old, neo);
  if (got != old) {
    return false;
  }
  if (get_tag(neo) <= VAR) {
    u64 arg_loc = get_loc(neo, get_tag(neo) & 0x01ull);
    atomicStore(ref_nb_data(heap, arg_loc), Arg(loc));
  }
  return true;
}

__device__ void atomic_subst(Heap heap, u64 tid, u64 var, u64 val) {
  while (1) {
    u64 arg_ptr = load_ptr(heap, get_loc(var, get_tag(var) & 0x01ull));
    if (get_tag(arg_ptr) == ARG) {
      if (atomic_relink(heap, get_loc(arg_ptr, 0ull), var, val)) {
        return;
      } else {
        continue;
      }
    }
    if (get_tag(arg_ptr) == ERA) {
      collect(heap, tid, val);
      return;
    }
  }
}

// Rewrite rules
// -------------

__device__ void app_lam(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0) {
  inc_cost(heap, tid);
  atomic_subst(heap, tid, Var(get_loc(arg0, 0)), take_arg(heap, term, 1));
  link(heap, host, take_arg(heap, arg0, 1));
  free(heap, get_loc(term, 0), 2);
  free(heap, get_loc(arg0, 0), 2);
}

__device__ void app_sup(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0) {
  inc_cost(heap, tid);
  u64 app0 = get_loc(term, 0);
  u64 app1 = get_loc(arg0, 0);
  u64 let0 = alloc(heap, tid, 3);
  u64 par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, term, 1));
  link(heap, app0 + 1, Dp0(get_ext(arg0), let0));
  link(heap, app0 + 0, take_arg(heap, arg0, 0));
  link(heap, app1 + 0, take_arg(heap, arg0, 1));
  link(heap, app1 + 1, Dp1(get_ext(arg0), let0));
  link(heap, par0 + 0, App(app0));
  link(heap, par0 + 1, App(app1));
  u64 done = Sup(get_ext(arg0), par0);
  link(heap, host, done);
}

__device__ void dup_lam(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  u64 let0 = alloc(heap, tid, 3);
  u64 par0 = alloc(heap, tid, 2);
  u64 lam0 = alloc(heap, tid, 2);
  u64 lam1 = alloc(heap, tid, 2);
  link(heap, let0 + 2, take_arg(heap, arg0, 1));
  link(heap, par0 + 1, Var(lam1));
  link(heap, par0 + 0, Var(lam0));
  link(heap, lam0 + 1, Dp0(get_ext(term), let0));
  link(heap, lam1 + 1, Dp1(get_ext(term), let0));
  atomic_subst(heap, tid, Var(get_loc(arg0, 0)), Sup(get_ext(term), par0));
  atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), Lam(lam0));
  atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), Lam(lam1));
  u64 done = Lam(get_tag(term) == DP0 ? lam0 : lam1);
  link(heap, host, done);
  free(heap, get_loc(term, 0), 3);
  free(heap, get_loc(arg0, 0), 2);
}

__device__ void dup_sup_0(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), take_arg(heap, arg0, 0));
  atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), take_arg(heap, arg0, 1));
  free(heap, get_loc(term, 0), 3);
  free(heap, get_loc(arg0, 0), 2);
}

__device__ void dup_sup_1(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  u64 par0 = alloc(heap, tid, 2);
  u64 let0 = alloc(heap, tid, 3);
  u64 par1 = get_loc(arg0, 0);
  u64 let1 = alloc(heap, tid, 3);
  link(heap, let0 + 2, take_arg(heap, arg0, 0));
  link(heap, let1 + 2, take_arg(heap, arg0, 1));
  link(heap, par1 + 0, Dp1(tcol, let0));
  link(heap, par1 + 1, Dp1(tcol, let1));
  link(heap, par0 + 0, Dp0(tcol, let0));
  link(heap, par0 + 1, Dp0(tcol, let1));
  atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par0));
  atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), Sup(get_ext(arg0), par1));
  free(heap, get_loc(term, 0), 3);
}

__device__ void dup_num(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), arg0);
  atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), arg0);
  free(heap, get_loc(term, 0), 3);
}

__device__ void dup_ctr(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  u64 fnid = get_ext(arg0);
  u64 arit = arity_of(arg0);
  if (arit == 0) {
    atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnid, 0));
    atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnid, 0));
    link(heap, host, Ctr(fnid, 0));
    free(heap, get_loc(term, 0), 3);
  } else {
    u64 ctr0 = get_loc(arg0, 0);
    u64 ctr1 = alloc(heap, tid, arit);
    for (u64 i = 0; i < arit - 1; ++i) {
      u64 leti = alloc(heap, tid, 3);
      link(heap, leti + 2, take_arg(heap, arg0, i));
      link(heap, ctr0 + i, Dp0(get_ext(term), leti));
      link(heap, ctr1 + i, Dp1(get_ext(term), leti));
    }
    u64 leti = alloc(heap, tid, 3);
    link(heap, leti + 2, take_arg(heap, arg0, arit - 1));
    link(heap, ctr0 + arit - 1, Dp0(get_ext(term), leti));
    link(heap, ctr1 + arit - 1, Dp1(get_ext(term), leti));
    atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), Ctr(fnid, ctr0));
    atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), Ctr(fnid, ctr1));
    free(heap, get_loc(term, 0), 3);
  }
}

__device__ void dup_era(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, u64 tcol) {
  inc_cost(heap, tid);
  atomic_subst(heap, tid, Dp0(tcol, get_loc(term, 0)), Era());
  atomic_subst(heap, tid, Dp1(tcol, get_loc(term, 0)), Era());
  link(heap, host, Era());
  free(heap, get_loc(term, 0), 3);
}

__device__ void op2_num(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, Ptr arg1) {
  inc_cost(heap, tid);
  u64 a = get_num(arg0);
  u64 b = get_num(arg1);
  u64 c;
  switch (get_ext(term)) {
    case ADD: { c = (a + b) & NUM_MASK; break; }
    case SUB: { c = (a - b) & NUM_MASK; break; }
    case MUL: { c = (a * b) & NUM_MASK; break; }
    case DIV: { c = (a / b) & NUM_MASK; break; }
    case MOD: { c = (a % b) & NUM_MASK; break; }
    case AND: { c = (a & b) & NUM_MASK; break; }
    case OR:  { c = (a | b) & NUM_MASK; break; }
    case XOR: { c = (a ^ b) & NUM_MASK; break; }
    case SHL: { c = (a << b) & NUM_MASK; break; }
    case SHR: { c = (a >> b) & NUM_MASK; break; }
    case LTN: { c = a < b ? 1 : 0; break; }
    case LTE: { c = a <= b ? 1 : 0; break; }
    case EQL: { c = a == b ? 1 : 0; break; }
    case GTE: { c = a >= b ? 1 : 0; break; }
    case GTN: { c = a > b ? 1 : 0; break; }
    case NEQ: { c = a != b ? 1 : 0; break; }
    default:  { c = 0; break; }
  }
  u64 done = Num(c);
  link(heap, host, done);
  free(heap, get_loc(term, 0), 2);
}

__device__ void op2_sup_0(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, Ptr arg1) {
  inc_cost(heap, tid);
  u64 op20 = get_loc(term, 0);
  u64 op21 = get_loc(arg0, 0);
  u64 let0 = alloc(heap, tid, 3);
  u64 par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, arg1);
  link(heap, op20 + 1, Dp0(get_ext(arg0), let0));
  link(heap, op20 + 0, take_arg(heap, arg0, 0));
  link(heap, op21 + 0, take_arg(heap, arg0, 1));
  link(heap, op21 + 1, Dp1(get_ext(arg0), let0));
  link(heap, par0 + 0, Op2(get_ext(term), op20));
  link(heap, par0 + 1, Op2(get_ext(term), op21));
  u64 done = Sup(get_ext(arg0), par0);
  link(heap, host, done);
}

__device__ void op2_sup_1(Heap heap, u64 tid, u64 host, Ptr term, Ptr arg0, Ptr arg1) {
  inc_cost(heap, tid);
  u64 op20 = get_loc(term, 0);
  u64 op21 = get_loc(arg1, 0);
  u64 let0 = alloc(heap, tid, 3);
  u64 par0 = alloc(heap, tid, 2);
  link(heap, let0 + 2, arg0);
  link(heap, op20 + 0, Dp0(get_ext(arg1), let0));
  link(heap, op20 + 1, take_arg(heap, arg1, 0));
  link(heap, op21 + 1, take_arg(heap, arg1, 1));
  link(heap, op21 + 0, Dp1(get_ext(arg1), let0));
  link(heap, par0 + 0, Op2(get_ext(term), op20));
  link(heap, par0 + 1, Op2(get_ext(term), op21));
  u64 done = Sup(get_ext(arg1), par0);
  link(heap, host, done);
}

__device__ Ptr fun_sup(Heap heap, u64 tid, u64 host, Ptr term, Ptr argn, u64 n) {
  inc_cost(heap, tid);
  u64 arit = arity_of(term);
  u64 func = get_ext(term);
  u64 fun0 = get_loc(term, 0);
  u64 fun1 = alloc(heap, tid, arit);
  u64 par0 = get_loc(argn, 0);
  for (u64 i = 0; i < arit; ++i) {
    if (i != n) {
      u64 leti = alloc(heap, tid, 3);
      u64 argi = take_arg(heap, term, i);
      link(heap, fun0 + i, Dp0(get_ext(argn), leti));
      link(heap, fun1 + i, Dp1(get_ext(argn), leti));
      link(heap, leti + 2, argi);
    } else {
      link(heap, fun0 + i, take_arg(heap, argn, 0));
      link(heap, fun1 + i, take_arg(heap, argn, 1));
    }
  }
  link(heap, par0 + 0, Fun(func, fun0));
  link(heap, par0 + 1, Fun(func, fun1));
  u64 done = Sup(get_ext(argn), par0);
  link(heap, host, done);
  return done;
}

// Locks
// -----

__device__ bool acquire_lock(Heap heap, u64 idx, u64 tid) {
  return atomicCAS(ref_lk_data(heap, idx), (u16)0, (u16)tid) == 0;
}

__device__ bool release_lock(Heap heap, u64 idx) {
  atomicStore16(ref_lk_data(heap, idx), 0);
}

// Reduction
// ---------

__device__ __host__ void print_ptr(u64 tid, u64 loc, Ptr x);
__global__ void reduce_kernel(Heap heap, u64 root) {
  u64 tid = blockIdx.x * blockDim.x + threadIdx.x;

  bool work = tid == 0;
  bool init = tid == 0;
  u64  cont = tid == 0 ? RB_EXIT : 0;
  u64  host = tid == 0 ? root : 0;
  u64  dlen = 0;
  u64  dvis[4];

  while (1) {
    if (work) {
      u64 term = load_ptr(heap, host);
      //printf("[%04llu] work init=%d cont=%06llx host=%08llx tick=%08llu cost=%08llu\n", tid, init, cont, host, tick, *ref_cost(heap, tid));
      //print_ptr(tid, host, term);
      if (init) {
        switch (get_tag(term)) {
          case APP: {
            u64 goup = rb_insert(heap, tid, new_redex(host, cont, 1));
            work = true;
            init = true;
            cont = goup;
            host = get_loc(term, 0);
            continue;
          }
          case DP0:
          case DP1: {
            u64 lock = get_loc(term, 0);
            if (!acquire_lock(heap, lock, tid)) {
              dvis[dlen++] = new_visit(host, cont);
              work = false;
              init = true;
              continue;
            } else {
              // If the term changed, release lock and try again
              if (term != load_ptr(heap, host)) {
                release_lock(heap, lock);
                continue;
              }
              u64 goup = rb_insert(heap, tid, new_redex(host, cont, 1));
              work = true;
              init = true;
              cont = goup;
              host = get_loc(term, 2);
              continue;
            }
            break;
          }
          case OP2: {
            u64 goup = rb_insert(heap, tid, new_redex(host, cont, 2));
            vq_push(heap, tid, new_visit(get_loc(term, 1), goup));
            work = true;
            init = true;
            cont = goup;
            host = get_loc(term, 0);
            continue;
          }
          case FUN: {
            u64 fid = get_ext(term);
            switch (fid) {
              case FOO: {
                u64 goup = rb_insert(heap, tid, new_redex(host, cont, 2));
                vq_push(heap, tid, new_visit(get_loc(term, 0), goup));
                work = true;
                init = true;
                cont = goup;
                host = get_loc(term, 1);
                continue;
              }
            }
            break;
          }
        }
        work = true;
        init = false;
        continue;
      } else {
        switch (get_tag(term)) {
          case APP: {
            u64 arg0 = load_arg(heap, term, 0);
            if (get_tag(arg0) == LAM) {
              app_lam(heap, tid, host, term, arg0);
              work = true;
              init = true;
              continue;
            }
            if (get_tag(arg0) == SUP) {
              app_sup(heap, tid, host, term, arg0);
              break;
            }
          }
          case DP0:
          case DP1: {
            u64 arg0 = load_arg(heap, term, 2);
            u64 tcol = get_ext(term);
            u64 lock = get_loc(term, 0);
            if (get_tag(arg0) == LAM) {
              dup_lam(heap, tid, host, term, arg0, tcol);
              release_lock(heap, lock);
              work = true;
              init = true;
              continue;
            } else if (get_tag(arg0) == SUP) {
              if (tcol == get_ext(arg0)) {
                dup_sup_0(heap, tid, host, term, arg0, tcol);
                release_lock(heap, lock);
                work = true;
                init = true;
                continue;
              } else {
                dup_sup_1(heap, tid, host, term, arg0, tcol);
                release_lock(heap, lock);
                work = true;
                init = true;
                continue;
              }
            } else if (get_tag(arg0) == NUM) {
              dup_num(heap, tid, host, term, arg0, tcol);
              release_lock(heap, lock);
              work = true;
              init = true;
              continue;
            } else if (get_tag(arg0) == CTR) {
              dup_ctr(heap, tid, host, term, arg0, tcol);
              release_lock(heap, lock);
              work = true;
              init = true;
              continue;
            } else if (get_tag(arg0) == ERA) {
              dup_era(heap, tid, host, term, arg0, tcol);
              release_lock(heap, lock);
              work = true;
              init = true;
              continue;
            } else {
              release_lock(heap, lock);
              break;
            }
          }
          case OP2: {
            u64 arg0 = load_arg(heap, term, 0);
            u64 arg1 = load_arg(heap, term, 1);
            if (get_tag(arg0) == NUM && get_tag(arg1) == NUM) {
              op2_num(heap, tid, host, term, arg0, arg1);
              break;
            } else if (get_tag(arg0) == SUP) {
              op2_sup_0(heap, tid, host, term, arg0, arg1);
              break;
            } else if (get_tag(arg1) == SUP) {
              op2_sup_1(heap, tid, host, term, arg0, arg1);
              break;
            }
          }
          case FUN: {
            u64 fid = get_ext(term);
            switch (fid) {
              case FOO: {
                if (get_tag(load_arg(heap,term,0)) == SUP) {
                  fun_sup(heap, tid, host, term, load_arg(heap, term, 0), 0);
                  continue;
                }
                if (get_tag(load_arg(heap,term,1)) == SUP) {
                  fun_sup(heap, tid, host, term, load_arg(heap, term, 1), 1);
                  continue;
                }
                if ((get_tag(load_arg(heap, term, 0)) == NUM && get_num(load_arg(heap, term, 0)) == 0) && (get_tag(load_arg(heap, term, 1)) == NUM && get_num(load_arg(heap, term, 1)) == 0)) {
                  inc_cost(heap, tid);
                  u64 done = Num(1);
                  link(heap, host, done);
                  free(heap, get_loc(term, 0), 2);
                  init = true;
                  continue;
                }
                if ((get_tag(load_arg(heap, term, 0)) == NUM && get_num(load_arg(heap, term, 0)) == 0) && (get_tag(load_arg(heap, term, 1)) == CTR || get_tag(load_arg(heap, term, 1)) == NUM)) {
                  inc_cost(heap, tid);
                  u64 ret_0;
                  if (get_tag(load_arg(heap, term, 1)) == NUM && get_tag(Num(1)) == NUM) {
                    ret_0 = Num(get_num(load_arg(heap, term, 1)) - get_num(Num(1)));
                    inc_cost(heap, tid);
                  } else {
                    u64 op2_1 = alloc(heap, tid, 2);
                    link(heap, op2_1 + 0, load_arg(heap, term, 1));
                    link(heap, op2_1 + 1, Num(1));
                    ret_0 = Op2(SUB, op2_1);
                  }
                  u64 cal_2 = alloc(heap, tid, 2);
                  link(heap, cal_2 + 0, Num(0));
                  link(heap, cal_2 + 1, ret_0);
                  u64 done = Fun(FOO, cal_2);
                  link(heap, host, done);
                  free(heap, get_loc(term, 0), 2);
                  init = true;
                  continue;
                }
                if ((get_tag(load_arg(heap, term, 0)) == CTR || get_tag(load_arg(heap, term, 0)) == NUM) && (get_tag(load_arg(heap, term, 1)) == CTR || get_tag(load_arg(heap, term, 1)) == NUM)) {
                  inc_cost(heap, tid);
                  u64 cpy_0 = load_arg(heap, term, 0);
                  u64 dp0_1;
                  u64 dp1_2;
                  if (get_tag(cpy_0) == NUM) {
                    inc_cost(heap, tid);
                    dp0_1 = cpy_0;
                    dp1_2 = cpy_0;
                  } else {
                    u64 dup_3 = alloc(heap, tid, 3);
                    u64 col_4 = gen_dup(heap, tid);
                    link(heap, dup_3 + 2, cpy_0);
                    dp0_1 = Dp0(col_4, dup_3);
                    dp1_2 = Dp1(col_4, dup_3);
                  }
                  u64 cpy_5 = load_arg(heap, term, 1);
                  u64 dp0_6;
                  u64 dp1_7;
                  if (get_tag(cpy_5) == NUM) {
                    inc_cost(heap, tid);
                    dp0_6 = cpy_5;
                    dp1_7 = cpy_5;
                  } else {
                    u64 dup_8 = alloc(heap, tid, 3);
                    u64 col_9 = gen_dup(heap, tid);
                    link(heap, dup_8 + 2, cpy_5);
                    dp0_6 = Dp0(col_9, dup_8);
                    dp1_7 = Dp1(col_9, dup_8);
                  }
                  u64 ret_12;
                  if (get_tag(dp0_1) == NUM && get_tag(Num(1)) == NUM) {
                    ret_12 = Num(get_num(dp0_1) - get_num(Num(1)));
                    inc_cost(heap, tid);
                  } else {
                    u64 op2_13 = alloc(heap, tid, 2);
                    link(heap, op2_13 + 0, dp0_1);
                    link(heap, op2_13 + 1, Num(1));
                    ret_12 = Op2(SUB, op2_13);
                  }
                  u64 cal_14 = alloc(heap, tid, 2);
                  link(heap, cal_14 + 0, ret_12);
                  link(heap, cal_14 + 1, dp0_6);
                  u64 ret_15;
                  if (get_tag(dp1_2) == NUM && get_tag(Num(1)) == NUM) {
                    ret_15 = Num(get_num(dp1_2) - get_num(Num(1)));
                    inc_cost(heap, tid);
                  } else {
                    u64 op2_16 = alloc(heap, tid, 2);
                    link(heap, op2_16 + 0, dp1_2);
                    link(heap, op2_16 + 1, Num(1));
                    ret_15 = Op2(SUB, op2_16);
                  }
                  u64 cal_17 = alloc(heap, tid, 2);
                  link(heap, cal_17 + 0, ret_15);
                  link(heap, cal_17 + 1, dp1_7);
                  u64 ret_10;
                  if (get_tag(Fun(FOO, cal_14)) == NUM && get_tag(Fun(FOO, cal_17)) == NUM) {
                    ret_10 = Num(get_num(Fun(FOO, cal_14)) + get_num(Fun(FOO, cal_17)));
                    inc_cost(heap, tid);
                  } else {
                    u64 op2_11 = alloc(heap, tid, 2);
                    link(heap, op2_11 + 0, Fun(FOO, cal_14));
                    link(heap, op2_11 + 1, Fun(FOO, cal_17));
                    ret_10 = Op2(ADD, op2_11);
                  }
                  u64 done = ret_10;
                  link(heap, host, done);
                  free(heap, get_loc(term, 0), 2);
                  init = true;
                  continue;
                }
              }
            }
            break;
          }
        }

        // If root is on WHNF, halt
        if (cont == RB_EXIT) {
          //printf("[%04llu] completed all\n", tid);
          atomicStore(ref_stop(heap), true);
          break;
        }

        // Otherwise, try reducing the parent redex
        u64 redex = rb_complete(heap, tid, cont);
        //printf("[%04llu] completing %llu\n", tid, cont);
        if (redex != -1ull) {
          //printf("[%04llu] completed %llu\n", tid, cont);
          work = true;
          init = false;
          host = get_redex_host(redex);
          cont = get_redex_cont(redex);
          continue;
        }

        // Otherwise, visit next pointer
        //printf("[%04llu] must visit next\n", tid);
        work = false;
        init = true;
        continue;
      }
    } else {
      //if (debug) {
        //printf("[%04llu] idle init=%d cont=%06llx host=%08llx tick=%08llu cost=%08llu\n", tid, init, cont, host, tick, *ref_cost(heap, tid));
      //}
      if (init) {
        // If available, visit a new location
        u64 visit = vq_pop(heap, tid);
        if (visit != -1ull) {
          work = true;
          init = true;
          host = get_visit_host(visit);
          cont = get_visit_cont(visit);
          continue;
        }
        // If available, visit a delayed location
        if (dlen > 0) {
          for (u64 i = 0; i < dlen; ++i) {
            vq_push(heap, tid, dvis[dlen - i - 1]);
          }
          dlen = 0;
          work = false;
          init = true;
          continue;
        }
        // Otherwise, we have nothing to do
        work = false;
        init = false;
        continue;
      } else {
        //println!("[{}] idle locks={}", tid, locks);
        //println!("[{}] will try to steal...", tid);
        if (atomicLoad(ref_stop(heap))) {
          break;
        } else {
          for (u64 i = 0; i < TIDS; ++i) {
            u64 victim_tid = (tid + i) % TIDS;
            if (victim_tid != tid) {
              u64 stolen = vq_steal(heap, victim_tid);
              if (stolen != -1ull) {
                work = true;
                init = true;
                host = get_visit_host(stolen);
                cont = get_visit_cont(stolen);
                //printf("[%04llu] stole task %llu (host=%llu, cont=%llu) from %llu\n", tid, host, stolen, cont, victim_tid);
                break;
              }
            }
            //__nanosleep(8);
          }
          continue;
        }
      }
    }
  }
}

// Debug
// -----

__device__ __host__ void print_ptr(u64 tid, u64 loc, Ptr x) {
  if (x == 0) {
    printf("[%04llu] ~\n", tid);
  } else {
    u64 tag = get_tag(x);
    u64 ext = get_ext(x);
    u64 val = get_val(x);
    switch (tag) {
      case DP0: { printf("[%04llu] %08llx | Dp0(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case DP1: { printf("[%04llu] %08llx | Dp1(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case VAR: { printf("[%04llu] %08llx | Var(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case ARG: { printf("[%04llu] %08llx | Arg(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case ERA: { printf("[%04llu] %08llx | Era(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case LAM: { printf("[%04llu] %08llx | Lam(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case APP: { printf("[%04llu] %08llx | App(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case SUP: { printf("[%04llu] %08llx | Sup(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case CTR: { printf("[%04llu] %08llx | Ctr(%s, %08llx)\n", tid, loc, name_of(x), val); break; }
      case FUN: { printf("[%04llu] %08llx | Fun(%s, %08llx)\n", tid, loc, name_of(x), val); break; }
      case OP2: { printf("[%04llu] %08llx | Op2(%llx, %08llx)\n", tid, loc, ext, val); break; }
      case NUM: { printf("[%04llu] %08llx | Num(%llx, %08llx)\n", tid, loc, ext, val); break; }
      default : { printf("[%04llu] %08llx ?\n", tid, loc); break; }
    };
  }
}

// Main
// ----

__global__ void test_kernel(u64* data) {
  u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("bloco %d, thread %d, tid %llu, data %llu\n", blockIdx.x, threadIdx.x, tid, data[tid]);
  data[tid] += tid;
}

int main() {
  int nDevices;

  printf("starting...\n");

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    // prints cuda runtime version and driver version
    printf("\n");
  }

  // Creates heap
  u64 mini = MINI_SIZE * sizeof(u64);
  u64 size = HEAP_SIZE * sizeof(u64);
  printf("allocated cpu heap\n");

  u64* heap_cpu = (u64*)malloc(mini);
  *ref_nb_data(heap_cpu, 0) = Fun(FOO, 1);
  *ref_nb_data(heap_cpu, 1) = Num(D);
  *ref_nb_data(heap_cpu, 2) = Num(L);

  // Moves heap to GPU
  u64* heap_gpu;
  cudaMalloc(&heap_gpu, size);
  printf("allocated gpu heap\n");
  cudaMemcpy(heap_gpu, heap_cpu, mini, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  printf("moved data to gpu\n");

  // Runs kernel
  double ini_time = now();
  reduce_kernel<<<H, W>>>(heap_gpu, 0);
  cudaDeviceSynchronize();
  double end_time = now();
  printf("kernels completed\n");

  // Moves heap to CPU
  cudaMemcpy(heap_cpu, heap_gpu, mini, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("moved data to cpu\n");

  // Frees GPU memory
  cudaFree(heap_gpu);

  // Print and compute tid costs
  printf("cost per thread:\n");
  u64 sum_cost = 0;
  for (u64 tid = 0; tid < TIDS; ++tid) {
    u64 tid_cost = *ref_cost(heap_cpu, tid);
    sum_cost += tid_cost;
    if (tid_cost > 0) {
      printf("[%04llu] cost: %llu\n", tid, tid_cost);
    }
  }

  // Prints results
  printf("computation result:\n");
  for (u64 i = 0; i < 4; ++i) {
    print_ptr(0, i, *ref_nb_data(heap_cpu, i));
  }

  // Print stats
  printf("cost: %llu\n", sum_cost);
  printf("time: %f s\n", end_time - ini_time);
  printf("perf: %f MR/s\n", (double)sum_cost / (end_time - ini_time) / 1000000.0);

  //printf("maxa: %llu\n", *ref_maxa(heap_cpu));
  //printf("maxr: %llu\n", *ref_maxr(heap_cpu));
  //printf("maxv: %llu\n", *ref_maxv(heap_cpu));
}
