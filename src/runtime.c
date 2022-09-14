// This is HVM's C runtime template. HVM files generate a copy of this file,
// modified to also include user-defined rules. It then can be compiled to run
// in parallel with -lpthreads.

#include <hvm-api.h>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*! GENERATED_PARALLEL_FLAG !*/

#ifdef PARALLEL
#include <pthread.h>
#include <stdatomic.h>
#endif

#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)

#ifdef PARALLEL
typedef pthread_t Thd;
#endif

// Consts
// ------

// When the program starts, we pre-alloc a big chunk of memory (set by cli flag).
// This will be replaced by a proper arena allocator soon (see the Issues)!
#define HEAP_SIZE /*! GENERATED_HEAP_SIZE */ 1 /* GENERATED_HEAP_SIZE !*/

#ifdef PARALLEL
#define MAX_WORKERS (/*! GENERATED_NUM_THREADS */ 1 /* GENERATED_NUM_THREADS !*/)
#else
#define MAX_WORKERS (1)
#endif

#define MAX_DUPS (16777216)
#define MAX_DYNFUNS (65536)
#define MAX_ARITY (256)

// Each worker has a fraction of the total.
#define MEM_SPACE (HEAP_SIZE/sizeof(u64)/MAX_WORKERS)
#define NORMAL_SEEN_MCAP (HEAP_SIZE/sizeof(u64)/(sizeof(u64)*8))

// Max different colors we're able to readback
#define DIRS_MCAP (0x10000)

// Terms
// -----

//GENERATED_CONSTRUCTOR_IDS_START//
/*! GENERATED_CONSTRUCTOR_IDS !*/
//GENERATED_CONSTRUCTOR_IDS_END//

#ifndef _MAIN_
#define _MAIN_ (0)
#endif

// Threads
// -------

typedef struct {
  u64 size;
  u64* data;
} Arr;

typedef struct {
  u64* data;
  u64  size;
  u64  mcap;
} Stk;

typedef struct {
  u64  tid;
  Ptr* node;
  u64  size;
  Stk  free[MAX_ARITY];
  u64  cost;
  u64  dups;
  u64* aris;
  u64  funs;

  #ifdef PARALLEL
  u64             has_work;
  pthread_mutex_t has_work_mutex;
  pthread_cond_t  has_work_signal;

  u64             has_result;
  pthread_mutex_t has_result_mutex;
  pthread_cond_t  has_result_signal;

  Thd  thread;
  #endif
} Worker;

// Globals
// -------

Worker workers[MAX_WORKERS];

// Array
// -----
// Some array utils

void array_write(Arr* arr, u64 idx, u64 value) {
  arr->data[idx] = value;
}

u64 array_read(Arr* arr, u64 idx) {
  return arr->data[idx];
}

// Stack
// -----
// Some stack utils.

u64 stk_growth_factor = 16;

void stk_init(Stk* stack) {
  stack->size = 0;
  stack->mcap = stk_growth_factor;
  stack->data = malloc(stack->mcap * sizeof(u64));
  assert(stack->data);
}

void stk_free(Stk* stack) {
  free(stack->data);
}

void stk_push(Stk* stack, u64 val) {
  if (UNLIKELY(stack->size == stack->mcap)) {
    stack->mcap = stack->mcap * stk_growth_factor;
    stack->data = realloc(stack->data, stack->mcap * sizeof(u64));
  }
  stack->data[stack->size++] = val;
}

u64 stk_pop(Stk* stack) {
  if (LIKELY(stack->size > 0)) {
    // TODO: shrink? -- impacts performance considerably
    //if (stack->size == stack->mcap / stk_growth_factor) {
      //stack->mcap = stack->mcap / stk_growth_factor;
      //stack->data = realloc(stack->data, stack->mcap * sizeof(u64));
      //printf("shrink %llu\n", stack->mcap);
    //}
    return stack->data[--stack->size];
  } else {
    return -1;
  }
}

u64 stk_find(Stk* stk, u64 val) {
  for (u64 i = 0; i < stk->size; ++i) {
    if (stk->data[i] == val) {
      return i;
    }
  }
  return -1;
}

// Memory
// ------
// Creating, storing and reading Ptrs, allocating and freeing memory.

Ptr Var(u64 pos) {
  return (VAR * TAG) | pos;
}

Ptr Dp0(u64 col, u64 pos) {
  return (DP0 * TAG) | (col * EXT) | pos;
}

Ptr Dp1(u64 col, u64 pos) {
  return (DP1 * TAG) | (col * EXT) | pos;
}

Ptr Arg(u64 pos) {
  return (ARG * TAG) | pos;
}

Ptr Era(void) {
  return (ERA * TAG);
}

Ptr Lam(u64 pos) {
  return (LAM * TAG) | pos;
}

Ptr App(u64 pos) {
  return (APP * TAG) | pos;
}

Ptr Par(u64 col, u64 pos) {
  return (SUP * TAG) | (col * EXT) | pos;
}

Ptr Op2(u64 ope, u64 pos) {
  return (OP2 * TAG) | (ope * EXT) | pos;
}

Ptr Num(u64 val) {
  return (NUM * TAG) | (val & NUM_MASK);
}

Ptr Nil(void) {
  return NIL * TAG;
}

Ptr Ctr(u64 ari, u64 fun, u64 pos) {
  return (CTR * TAG) | (fun * EXT) | pos;
}

// FIXME: update name to Fun
Ptr Cal(u64 ari, u64 fun, u64 pos) {
  return (FUN * TAG) | (fun * EXT) | pos;
}

u64 get_tag(Ptr lnk) {
  return lnk / TAG;
}

u64 get_ext(Ptr lnk) {
  return (lnk / EXT) & 0xFFFFFF;
}

u64 get_val(Ptr lnk) {
  return lnk & 0xFFFFFFFF;
}

u64 get_num(Ptr lnk) {
  return lnk & 0xFFFFFFFFFFFFFFF;
}

//u64 get_ari(Ptr lnk) {
  //return (lnk / ARI) & 0xF;
//}

u64 get_loc(Ptr lnk, u64 arg) {
  return get_val(lnk) + arg;
}

u64 ask_ari(Worker* mem, Ptr lnk) {
  u64 fid = get_ext(lnk);
  u64 got = fid < mem->funs ? mem->aris[fid] : 0;
  // TODO: remove this in a future update where ari will be removed from the lnk
  //if (get_ari(lnk) != got) {
    //printf("[ERROR] arity inconsistency\n");
    //exit(1);
  //}
  return got;
}

// Dereferences a Ptr, getting what is stored on its target position
Ptr ask_lnk(Worker* mem, u64 loc) {
  return mem->node[loc];
}

// Dereferences the nth argument of the Term represented by this Ptr
Ptr ask_arg(Worker* mem, Ptr term, u64 arg) {
  return ask_lnk(mem, get_loc(term, arg));
}

// This inserts a value in another. It just writes a position in memory if
// `value` is a constructor. If it is VAR, DP0 or DP1, it also updates the
// corresponding λ or dup binder.
u64 link(Worker* mem, u64 loc, Ptr lnk) {
  mem->node[loc] = lnk;
  //array_write(mem->nodes, loc, lnk);
  if (get_tag(lnk) <= VAR) {
    mem->node[get_loc(lnk, get_tag(lnk) == DP1 ? 1 : 0)] = Arg(loc);
    //array_write(mem->nodes, get_loc(lnk, get_tag(lnk) == DP1 ? 1 : 0), Arg(loc));
  }
  return lnk;
}

// Allocates a block of memory, up to 16 words long
u64 alloc(Worker* mem, u64 size) {
  if (UNLIKELY(size == 0)) {
    return 0;
  } else {
    u64 reuse = stk_pop(&mem->free[size]);
    if (reuse != -1) {
      return reuse;
    }
    u64 loc = mem->size;
    mem->size += size;
    return mem->tid * MEM_SPACE + loc;
    //return __atomic_fetch_add(&mem->nodes->size, size, __ATOMIC_RELAXED);
  }
}

// Frees a block of memory by adding its position a freelist
void clear(Worker* mem, u64 loc, u64 size) {
  stk_push(&mem->free[size], loc);
}

// Garbage Collection
// ------------------

// This clears the memory used by a term that became unreachable. It just frees
// all its nodes recursively. This is called as soon as a term goes out of
// scope. No global GC pass is necessary to find unreachable terms!
// HVM can still produce some garbage in very uncommon situations that are
// mostly irrelevant in practice. Absolute GC-freedom, though, requires
// uncommenting the `reduce` lines below, but this would make HVM not 100% lazy
// in some cases, so it should be called in a separate thread.
void collect(Worker* mem, Ptr term) {
  switch (get_tag(term)) {
    case DP0: {
      link(mem, get_loc(term,0), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,1),0));
      break;
    }
    case DP1: {
      link(mem, get_loc(term,1), Era());
      //reduce(mem, get_loc(ask_arg(mem,term,0),0));
      break;
    }
    case VAR: {
      link(mem, get_loc(term,0), Era());
      break;
    }
    case LAM: {
      if (get_tag(ask_arg(mem,term,0)) != ERA) {
        link(mem, get_loc(ask_arg(mem,term,0),0), Era());
      }
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }
    case APP: {
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }
    case SUP: {
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }
    case OP2: {
      collect(mem, ask_arg(mem,term,0));
      collect(mem, ask_arg(mem,term,1));
      clear(mem, get_loc(term,0), 2);
      break;
    }
    case NUM: {
      break;
    }
    case CTR: case FUN: {
      u64 arity = ask_ari(mem, term);
      for (u64 i = 0; i < arity; ++i) {
        collect(mem, ask_arg(mem,term,i));
      }
      clear(mem, get_loc(term,0), arity);
      break;
    }
  }
}

// Terms
// -----

void inc_cost(Worker* mem) {
  mem->cost++;
}

u64 gen_dupk(Worker* mem) {
  return mem->dups++ & 0xFFFFFF;
}

// Performs a `x <- value` substitution. It just calls link if the substituted
// value is a term. If it is an ERA node, that means `value` is now unreachable,
// so we just call the collector.
void subst(Worker* mem, Ptr lnk, Ptr val) {
  if (get_tag(lnk) != ERA) {
    link(mem, get_loc(lnk,0), val);
  } else {
    collect(mem, val);
  }
}

// (F {a0 a1} b c ...)
// ------------------- FUN-SUP
// dup b0 b1 = b
// dup c0 c1 = c
// ...
// {(F a0 b0 c0 ...) (F a1 b1 c1 ...)}
Ptr cal_par(Worker* mem, u64 host, Ptr term, Ptr argn, u64 n) {
  inc_cost(mem);
  u64 arit = ask_ari(mem, term);
  u64 func = get_ext(term);
  u64 fun0 = get_loc(term, 0);
  u64 fun1 = alloc(mem, arit);
  u64 par0 = get_loc(argn, 0);
  for (u64 i = 0; i < arit; ++i) {
    if (i != n) {
      u64 leti = alloc(mem, 3);
      u64 argi = ask_arg(mem, term, i);
      link(mem, fun0+i, Dp0(get_ext(argn), leti));
      link(mem, fun1+i, Dp1(get_ext(argn), leti));
      link(mem, leti+2, argi);
    } else {
      link(mem, fun0+i, ask_arg(mem, argn, 0));
      link(mem, fun1+i, ask_arg(mem, argn, 1));
    }
  }
  link(mem, par0+0, Cal(arit, func, fun0));
  link(mem, par0+1, Cal(arit, func, fun1));
  u64 done = Par(get_ext(argn), par0);
  link(mem, host, done);
  return done;
}

// Reduces a term to weak head normal form.
Ptr reduce(Worker* mem, u64 root, u64 slen) {
  Stk stack;
  stk_init(&stack);

  u64 init = 1;
  u32 host = (u32)root;

  while (1) {

    u64 term = ask_lnk(mem, host);

    //printf("reduce "); debug_print_lnk(term); printf("\n");
    //printf("------\n");
    //printf("reducing: host=%d size=%llu init=%llu ", host, stack.size, init); debug_print_lnk(term); printf("\n");
    //for (u64 i = 0; i < 256; ++i) {
      //printf("- %llx ", i); debug_print_lnk(mem->node[i]); printf("\n");
    //}

    if (init == 1) {
      switch (get_tag(term)) {
        case APP: {
          stk_push(&stack, host);
          //stack[size++] = host;
          init = 1;
          host = get_loc(term, 0);
          continue;
        }
        case DP0:
        case DP1: {
          #ifdef PARALLEL
          // TODO: reason about this, comment
          atomic_flag* flag = ((atomic_flag*)(mem->node + get_loc(term,0))) + 6;
          if (atomic_flag_test_and_set(flag) != 0) {
            continue;
          }

          // Term changed before we locked
          if (term != ask_lnk(mem, host)) {
            atomic_flag_clear(((atomic_flag*)(mem->node + get_loc(term,0))) + 6);
            continue;
          }
          #endif

          stk_push(&stack, host);
          host = get_loc(term, 2);
          continue;
        }
        case OP2: {
          if (slen == 1 || stack.size > 0) {
            stk_push(&stack, host);
            stk_push(&stack, get_loc(term, 0) | 0x80000000);
            //stack[size++] = host;
            //stack[size++] = get_loc(term, 0) | 0x80000000;
            host = get_loc(term, 1);
            continue;
          }
          break;
        }
        case FUN: {
          u64 fun = get_ext(term);
          u64 ari = ask_ari(mem, term);

          switch (fun)
          //GENERATED_REWRITE_RULES_STEP_0_START//
          {
/*! GENERATED_REWRITE_RULES_STEP_0 !*/
          }
          //GENERATED_REWRITE_RULES_STEP_0_END//

          break;
        }
      }

    } else {

      switch (get_tag(term)) {
        case APP: {
          u64 arg0 = ask_arg(mem, term, 0);
          switch (get_tag(arg0)) {

            // (λx(body) a)
            // ------------ APP-LAM
            // x <- a
            // body
            case LAM: {
              //printf("app-lam\n");
              inc_cost(mem);
              subst(mem, ask_arg(mem, arg0, 0), ask_arg(mem, term, 1));
              u64 done = link(mem, host, ask_arg(mem, arg0, 1));
              clear(mem, get_loc(term,0), 2);
              clear(mem, get_loc(arg0,0), 2);
              init = 1;
              continue;
            }

            // ({a b} c)
            // ----------------- APP-SUP
            // dup x0 x1 = c
            // {(a x0) (b x1)}
            case SUP: {
              //printf("app-sup\n");
              inc_cost(mem);
              u64 app0 = get_loc(term, 0);
              u64 app1 = get_loc(arg0, 0);
              u64 let0 = alloc(mem, 3);
              u64 par0 = alloc(mem, 2);
              link(mem, let0+2, ask_arg(mem, term, 1));
              link(mem, app0+1, Dp0(get_ext(arg0), let0));
              link(mem, app0+0, ask_arg(mem, arg0, 0));
              link(mem, app1+0, ask_arg(mem, arg0, 1));
              link(mem, app1+1, Dp1(get_ext(arg0), let0));
              link(mem, par0+0, App(app0));
              link(mem, par0+1, App(app1));
              u64 done = Par(get_ext(arg0), par0);
              link(mem, host, done);
              break;
            }

          }
          break;
        }
        case DP0:
        case DP1: {
          u64 arg0 = ask_arg(mem, term, 2);
          switch (get_tag(arg0)) {

            // dup r s = λx(f)
            // --------------- DUP-LAM
            // dup f0 f1 = f
            // r <- λx0(f0)
            // s <- λx1(f1)
            // x <- {x0 x1}
            case LAM: {
              //printf("dup-lam\n");
              inc_cost(mem);
              u64 let0 = get_loc(term, 0);
              u64 par0 = get_loc(arg0, 0);
              u64 lam0 = alloc(mem, 2);
              u64 lam1 = alloc(mem, 2);
              link(mem, let0+2, ask_arg(mem, arg0, 1));
              link(mem, par0+1, Var(lam1));
              u64 arg0_arg_0 = ask_arg(mem, arg0, 0);
              link(mem, par0+0, Var(lam0));
              subst(mem, arg0_arg_0, Par(get_ext(term), par0));
              u64 term_arg_0 = ask_arg(mem,term,0);
              link(mem, lam0+1, Dp0(get_ext(term), let0));
              subst(mem, term_arg_0, Lam(lam0));
              u64 term_arg_1 = ask_arg(mem,term,1);
              link(mem, lam1+1, Dp1(get_ext(term), let0));
              subst(mem, term_arg_1, Lam(lam1));
              u64 done = Lam(get_tag(term) == DP0 ? lam0 : lam1);
              link(mem, host, done);
              init = 1;
              continue;
            }

            // dup x y = {a b}
            // --------------- DUP-SUP (equal)
            // x <- a
            // y <- b
            //
            // dup x y = {a b}
            // ----------------- DUP-SUP (different)
            // x <- {xA xB}
            // y <- {yA yB}
            // dup xA yA = a
            // dup xB yB = b
            case SUP: {
              //printf("dup-sup\n");
              if (get_ext(term) == get_ext(arg0)) {
                inc_cost(mem);
                subst(mem, ask_arg(mem,term,0), ask_arg(mem,arg0,0));
                subst(mem, ask_arg(mem,term,1), ask_arg(mem,arg0,1));
                u64 done = link(mem, host, ask_arg(mem, arg0, get_tag(term) == DP0 ? 0 : 1));
                clear(mem, get_loc(term,0), 3);
                clear(mem, get_loc(arg0,0), 2);
                init = 1;
                continue;
              } else {
                inc_cost(mem);
                u64 par0 = alloc(mem, 2);
                u64 let0 = get_loc(term,0);
                u64 par1 = get_loc(arg0,0);
                u64 let1 = alloc(mem, 3);
                link(mem, let0+2, ask_arg(mem,arg0,0));
                link(mem, let1+2, ask_arg(mem,arg0,1));
                u64 term_arg_0 = ask_arg(mem,term,0);
                u64 term_arg_1 = ask_arg(mem,term,1);
                link(mem, par1+0, Dp1(get_ext(term),let0));
                link(mem, par1+1, Dp1(get_ext(term),let1));
                link(mem, par0+0, Dp0(get_ext(term),let0));
                link(mem, par0+1, Dp0(get_ext(term),let1));
                subst(mem, term_arg_0, Par(get_ext(arg0),par0));
                subst(mem, term_arg_1, Par(get_ext(arg0),par1));
                u64 done = Par(get_ext(arg0), get_tag(term) == DP0 ? par0 : par1);
                link(mem, host, done);
                break;
              }
              break;
            }

            // dup x y = N
            // ----------- DUP-NUM
            // x <- N
            // y <- N
            // ~
            case NUM: {
              //printf("dup-u32\n");
              inc_cost(mem);
              subst(mem, ask_arg(mem,term,0), arg0);
              subst(mem, ask_arg(mem,term,1), arg0);
              clear(mem, get_loc(term,0), 3);
              u64 done = arg0;
              link(mem, host, arg0);
              break;
            }

            // dup x y = (K a b c ...)
            // ----------------------- DUP-CTR
            // dup a0 a1 = a
            // dup b0 b1 = b
            // dup c0 c1 = c
            // ...
            // x <- (K a0 b0 c0 ...)
            // y <- (K a1 b1 c1 ...)
            case CTR: {
              //printf("dup-ctr\n");
              inc_cost(mem);
              u64 func = get_ext(arg0);
              u64 arit = ask_ari(mem, arg0);
              if (arit == 0) {
                subst(mem, ask_arg(mem,term,0), Ctr(0, func, 0));
                subst(mem, ask_arg(mem,term,1), Ctr(0, func, 0));
                clear(mem, get_loc(term,0), 3);
                u64 done = link(mem, host, Ctr(0, func, 0));
              } else {
                u64 ctr0 = get_loc(arg0,0);
                u64 ctr1 = alloc(mem, arit);
                for (u64 i = 0; i < arit - 1; ++i) {
                  u64 leti = alloc(mem, 3);
                  link(mem, leti+2, ask_arg(mem, arg0, i));
                  link(mem, ctr0+i, Dp0(get_ext(term), leti));
                  link(mem, ctr1+i, Dp1(get_ext(term), leti));
                }
                u64 leti = get_loc(term, 0);
                link(mem, leti + 2, ask_arg(mem, arg0, arit - 1));
                u64 term_arg_0 = ask_arg(mem, term, 0);
                link(mem, ctr0 + arit - 1, Dp0(get_ext(term), leti));
                subst(mem, term_arg_0, Ctr(arit, func, ctr0));
                u64 term_arg_1 = ask_arg(mem, term, 1);
                link(mem, ctr1 + arit - 1, Dp1(get_ext(term), leti));
                subst(mem, term_arg_1, Ctr(arit, func, ctr1));
                u64 done = Ctr(arit, func, get_tag(term) == DP0 ? ctr0 : ctr1);
                link(mem, host, done);
              }
              break;
            }

            // dup x y = *
            // ----------- DUP-CTR
            // x <- *
            // y <- *
            case ERA: {
              inc_cost(mem);
              subst(mem, ask_arg(mem, term, 0), Era());
              subst(mem, ask_arg(mem, term, 1), Era());
              link(mem, host, Era());
              clear(mem, get_loc(term, 0), 3);
              init = 1;
              continue;
            }

          }
          #ifdef PARALLEL
          atomic_flag* flag = ((atomic_flag*)(mem->node + get_loc(term,0))) + 6;
          atomic_flag_clear(flag);
          #endif
          break;
        }
        case OP2: {
          u64 arg0 = ask_arg(mem, term, 0);
          u64 arg1 = ask_arg(mem, term, 1);

          // (+ a b)
          // --------- OP2-NUM
          // add(a, b)
          if (get_tag(arg0) == NUM && get_tag(arg1) == NUM) {
            //printf("op2-u32\n");
            inc_cost(mem);
            u64 a = get_num(arg0);
            u64 b = get_num(arg1);
            u64 c = 0;
            switch (get_ext(term)) {
              case ADD: c = (a +  b) & NUM_MASK; break;
              case SUB: c = (a -  b) & NUM_MASK; break;
              case MUL: c = (a *  b) & NUM_MASK; break;
              case DIV: c = (a /  b) & NUM_MASK; break;
              case MOD: c = (a %  b) & NUM_MASK; break;
              case AND: c = (a &  b) & NUM_MASK; break;
              case OR : c = (a |  b) & NUM_MASK; break;
              case XOR: c = (a ^  b) & NUM_MASK; break;
              case SHL: c = (a << b) & NUM_MASK; break;
              case SHR: c = (a >> b) & NUM_MASK; break;
              case LTN: c = (a <  b) ? 1 : 0;    break;
              case LTE: c = (a <= b) ? 1 : 0;    break;
              case EQL: c = (a == b) ? 1 : 0;    break;
              case GTE: c = (a >= b) ? 1 : 0;    break;
              case GTN: c = (a >  b) ? 1 : 0;    break;
              case NEQ: c = (a != b) ? 1 : 0;    break;
            }
            u64 done = Num(c);
            clear(mem, get_loc(term,0), 2);
            link(mem, host, done);
          }

          // (+ {a0 a1} b)
          // --------------------- OP2-SUP-0
          // let b0 b1 = b
          // {(+ a0 b0) (+ a1 b1)}
          else if (get_tag(arg0) == SUP) {
            //printf("op2-sup-0\n");
            inc_cost(mem);
            u64 op20 = get_loc(term, 0);
            u64 op21 = get_loc(arg0, 0);
            u64 let0 = alloc(mem, 3);
            u64 par0 = alloc(mem, 2);
            link(mem, let0+2, arg1);
            link(mem, op20+1, Dp0(get_ext(arg0), let0));
            link(mem, op20+0, ask_arg(mem, arg0, 0));
            link(mem, op21+0, ask_arg(mem, arg0, 1));
            link(mem, op21+1, Dp1(get_ext(arg0), let0));
            link(mem, par0+0, Op2(get_ext(term), op20));
            link(mem, par0+1, Op2(get_ext(term), op21));
            u64 done = Par(get_ext(arg0), par0);
            link(mem, host, done);
          }

          // (+ a {b0 b1})
          // --------------- OP2-SUP-1
          // dup a0 a1 = a
          // {(+ a0 b0) (+ a1 b1)}
          else if (get_tag(arg1) == SUP) {
            //printf("op2-sup-1\n");
            inc_cost(mem);
            u64 op20 = get_loc(term, 0);
            u64 op21 = get_loc(arg1, 0);
            u64 let0 = alloc(mem, 3);
            u64 par0 = alloc(mem, 2);
            link(mem, let0+2, arg0);
            link(mem, op20+0, Dp0(get_ext(arg1), let0));
            link(mem, op20+1, ask_arg(mem, arg1, 0));
            link(mem, op21+1, ask_arg(mem, arg1, 1));
            link(mem, op21+0, Dp1(get_ext(arg1), let0));
            link(mem, par0+0, Op2(get_ext(term), op20));
            link(mem, par0+1, Op2(get_ext(term), op21));
            u64 done = Par(get_ext(arg1), par0);
            link(mem, host, done);
          }

          break;
        }
        case FUN: {
          u64 fun = get_ext(term);
          u64 ari = ask_ari(mem, term);

          switch (fun)
          //GENERATED_REWRITE_RULES_STEP_1_START//
          {
/*! GENERATED_REWRITE_RULES_STEP_1 !*/
          }
          //GENERATED_REWRITE_RULES_STEP_1_END//

          break;
        }
      }
    }

    u64 item = stk_pop(&stack);
    if (item == -1) {
      break;
    } else {
      init = item >> 31;
      host = item & 0x7FFFFFFF;
      continue;
    }

  }

  return ask_lnk(mem, root);
}

// sets the nth bit of a bit-array represented as a u64 array
void set_bit(u64* bits, u64 bit) {
  bits[bit >> 6] |= (1ULL << (bit & 0x3f));
}

// gets the nth bit of a bit-array represented as a u64 array
u8 get_bit(u64* bits, u64 bit) {
  return (bits[bit >> 6] >> (bit & 0x3F)) & 1;
}

#ifdef PARALLEL
void normal_fork(u64 tid, u64 host, u64 sidx, u64 slen);
u64  normal_join(u64 tid);
#endif

u64 normal_seen_data[NORMAL_SEEN_MCAP];

void normal_init(void) {
  for (u64 i = 0; i < NORMAL_SEEN_MCAP; ++i) {
    normal_seen_data[i] = 0;
  }
}

Ptr normal_go(Worker* mem, u64 host, u64 sidx, u64 slen) {
  Ptr term = ask_lnk(mem, host);
  //printf("normal %llu %llu | ", sidx, slen); debug_print_lnk(term); printf("\n");
  if (get_bit(normal_seen_data, host)) {
    return term;
  } else {
    term = reduce(mem, host, slen);
    set_bit(normal_seen_data, host);
    u64 rec_size = 0;
    u64 rec_locs[16];
    switch (get_tag(term)) {
      case LAM: {
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }
      case APP: {
        rec_locs[rec_size++] = get_loc(term,0);
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }
      case SUP: {
        rec_locs[rec_size++] = get_loc(term,0);
        rec_locs[rec_size++] = get_loc(term,1);
        break;
      }
      case DP0: {
        rec_locs[rec_size++] = get_loc(term,2);
        break;
      }
      case DP1: {
        rec_locs[rec_size++] = get_loc(term,2);
        break;
      }
      case OP2: {
        if (slen > 1) {
          rec_locs[rec_size++] = get_loc(term,0);
          rec_locs[rec_size++] = get_loc(term,1);
          break;
        }
      }
      case CTR: case FUN: {
        u64 arity = (u64)ask_ari(mem, term);
        for (u64 i = 0; i < arity; ++i) {
          rec_locs[rec_size++] = get_loc(term,i);
        }
        break;
      }
    }
    #ifdef PARALLEL

    //printf("ue %llu %llu\n", rec_size, slen);

    if (rec_size >= 2 && slen >= rec_size) {

      u64 space = slen / rec_size;

      for (u64 i = 1; i < rec_size; ++i) {
        //printf("spawn %llu %llu\n", sidx + i * space, space);
        normal_fork(sidx + i * space, rec_locs[i], sidx + i * space, space);
      }

      link(mem, rec_locs[0], normal_go(mem, rec_locs[0], sidx, space));

      for (u64 i = 1; i < rec_size; ++i) {
        link(mem, get_loc(term, i), normal_join(sidx + i * space));
      }

    } else {

      for (u64 i = 0; i < rec_size; ++i) {
        link(mem, rec_locs[i], normal_go(mem, rec_locs[i], sidx, slen));
      }

    }
    #else

    for (u64 i = 0; i < rec_size; ++i) {
      link(mem, rec_locs[i], normal_go(mem, rec_locs[i], sidx, slen));
    }

    #endif

    return term;
  }
}

Ptr normal(Worker* mem, u64 host, u64 sidx, u64 slen) {
  // In order to allow parallelization of numeric operations, reduce() will treat OP2 as a CTR if
  // there is enough thread space. So, for example, normalizing a recursive "sum" function with 4
  // threads might return something like `(+ (+ 64 64) (+ 64 64))`. reduce() will treat the first
  // 2 layers as CTRs, allowing normal() to parallelize them. So, in order to finish the reduction,
  // we call `normal_go()` a second time, with no thread space, to eliminate lasting redexes.
  normal_init();
  normal_go(mem, host, sidx, slen);
  u64 done;
  u64 cost = mem->cost;
  while (1) {
    normal_init();
    done = normal_go(mem, host, 0, 1);
    if (mem->cost != cost) {
      cost = mem->cost;
    } else {
      break;
    }
  }
  return done;
}


#ifdef PARALLEL

// Normalizes in a separate thread
// Note that, right now, the allocator will just partition the space of the
// normal form equally among threads, which will not fully use the CPU cores in
// many cases. A better task scheduler should be implemented. See Issues.
void normal_fork(u64 tid, u64 host, u64 sidx, u64 slen) {
  pthread_mutex_lock(&workers[tid].has_work_mutex);
  workers[tid].has_work = (sidx << 48) | (slen << 32) | host;
  pthread_cond_signal(&workers[tid].has_work_signal);
  pthread_mutex_unlock(&workers[tid].has_work_mutex);
}

// Waits the result of a forked normalizer
u64 normal_join(u64 tid) {
  while (1) {
    pthread_mutex_lock(&workers[tid].has_result_mutex);
    while (workers[tid].has_result == -1) {
      pthread_cond_wait(&workers[tid].has_result_signal, &workers[tid].has_result_mutex);
    }
    u64 done = workers[tid].has_result;
    workers[tid].has_result = -1;
    pthread_mutex_unlock(&workers[tid].has_result_mutex);
    return done;
  }
}

// Stops a worker
void worker_stop(u64 tid) {
  pthread_mutex_lock(&workers[tid].has_work_mutex);
  workers[tid].has_work = -2;
  pthread_cond_signal(&workers[tid].has_work_signal);
  pthread_mutex_unlock(&workers[tid].has_work_mutex);
}

// The normalizer worker
void *worker(void *arg) {
  u64 tid = (u64)arg;
  while (1) {
    pthread_mutex_lock(&workers[tid].has_work_mutex);
    while (workers[tid].has_work == -1) {
      pthread_cond_wait(&workers[tid].has_work_signal, &workers[tid].has_work_mutex);
    }
    u64 work = workers[tid].has_work;
    if (work == -2) {
      break;
    } else {
      u64 sidx = (work >> 48) & 0xFFFF;
      u64 slen = (work >> 32) & 0xFFFF;
      u64 host = (work >>  0) & 0xFFFFFFFF;
      workers[tid].has_result = normal_go(&workers[tid], host, sidx, slen);
      workers[tid].has_work = -1;
      pthread_cond_signal(&workers[tid].has_result_signal);
      pthread_mutex_unlock(&workers[tid].has_work_mutex);
    }
  }
  return 0;
}

#endif

u64 ffi_cost;
u64 ffi_size;

void ffi_normal(u8* mem_data, u32 mem_size, u32 host) {

  // Init thread objects
  for (u64 t = 0; t < MAX_WORKERS; ++t) {
    workers[t].tid = t;
    workers[t].size = t == 0 ? (u64)mem_size : 0l;
    workers[t].node = (u64*)mem_data;
    for (u64 a = 0; a < MAX_ARITY; ++a) {
      stk_init(&workers[t].free[a]);
    }
    workers[t].cost = 0;
    workers[t].dups = MAX_DUPS * t / MAX_WORKERS;
    #ifdef PARALLEL
    workers[t].has_work = -1;
    pthread_mutex_init(&workers[t].has_work_mutex, NULL);
    pthread_cond_init(&workers[t].has_work_signal, NULL);
    workers[t].has_result = -1;
    pthread_mutex_init(&workers[t].has_result_mutex, NULL);
    pthread_cond_init(&workers[t].has_result_signal, NULL);
    // workers[t].thread = NULL;
    #endif
  }

  // Spawns threads
  #ifdef PARALLEL
  for (u64 tid = 1; tid < MAX_WORKERS; ++tid) {
    pthread_create(&workers[tid].thread, NULL, &worker, (void*)tid);
  }
  #endif

  // Normalizes trm
  normal(&workers[0], (u64) host, 0, MAX_WORKERS);

  // Computes total cost and size
  ffi_cost = 0;
  ffi_size = 0;
  for (u64 tid = 0; tid < MAX_WORKERS; ++tid) {
    ffi_cost += workers[tid].cost;
    ffi_size += workers[tid].size;
  }

  #ifdef PARALLEL

  // Asks workers to stop
  for (u64 tid = 1; tid < MAX_WORKERS; ++tid) {
    worker_stop(tid);
  }

  // Waits workers to stop
  for (u64 tid = 1; tid < MAX_WORKERS; ++tid) {
    pthread_join(workers[tid].thread, NULL);
  }

  #endif

  // Clears workers
  for (u64 tid = 0; tid < MAX_WORKERS; ++tid) {
    for (u64 a = 0; a < MAX_ARITY; ++a) {
      stk_free(&workers[tid].free[a]);
    }
    #ifdef PARALLEL
    pthread_mutex_destroy(&workers[tid].has_work_mutex);
    pthread_cond_destroy(&workers[tid].has_work_signal);
    pthread_mutex_destroy(&workers[tid].has_result_mutex);
    pthread_cond_destroy(&workers[tid].has_result_signal);
    #endif
  }
}

// Readback
// --------

void readback_vars(Stk* vars, Worker* mem, Ptr term, Stk* seen) {
  //printf("- readback_vars %llu ", get_loc(term,0)); debug_print_lnk(term); printf("\n");
  if (stk_find(seen, term) != -1) { // FIXME: probably very slow, change to a proper hashmap
    return;
  } else {
    stk_push(seen, term);
    switch (get_tag(term)) {
      case LAM: {
        u64 argm = ask_arg(mem, term, 0);
        u64 body = ask_arg(mem, term, 1);
        if (get_tag(argm) != ERA) {
          stk_push(vars, Var(get_loc(term, 0)));
        };
        readback_vars(vars, mem, body, seen);
        break;
      }
      case APP: {
        u64 lam = ask_arg(mem, term, 0);
        u64 arg = ask_arg(mem, term, 1);
        readback_vars(vars, mem, lam, seen);
        readback_vars(vars, mem, arg, seen);
        break;
      }
      case SUP: {
        u64 arg0 = ask_arg(mem, term, 0);
        u64 arg1 = ask_arg(mem, term, 1);
        readback_vars(vars, mem, arg0, seen);
        readback_vars(vars, mem, arg1, seen);
        break;
      }
      case DP0: {
        u64 arg = ask_arg(mem, term, 2);
        readback_vars(vars, mem, arg, seen);
        break;
      }
      case DP1: {
        u64 arg = ask_arg(mem, term, 2);
        readback_vars(vars, mem, arg, seen);
        break;
      }
      case OP2: {
        u64 arg0 = ask_arg(mem, term, 0);
        u64 arg1 = ask_arg(mem, term, 1);
        readback_vars(vars, mem, arg0, seen);
        readback_vars(vars, mem, arg1, seen);
        break;
      }
      case CTR: case FUN: {
        u64 arity = ask_ari(mem, term);
        for (u64 i = 0; i < arity; ++i) {
          readback_vars(vars, mem, ask_arg(mem, term, i), seen);
        }
        break;
      }
    }
  }
}

void readback_decimal_go(Stk* chrs, u64 n) {
  //printf("--- A %llu\n", n);
  if (n > 0) {
    readback_decimal_go(chrs, n / 10);
    stk_push(chrs, '0' + (n % 10));
  }
}

void readback_decimal(Stk* chrs, u64 n) {
  if (n == 0) {
    stk_push(chrs, '0');
  } else {
    readback_decimal_go(chrs, n);
  }
}

void readback_term(Stk* chrs, Worker* mem, Ptr term, Stk* vars, Stk* dirs, char** id_to_name_data, u64 id_to_name_mcap) {
  //printf("- readback_term: "); debug_print_lnk(term); printf("\n");
  switch (get_tag(term)) {
    case LAM: {
      stk_push(chrs, '@');
      if (get_tag(ask_arg(mem, term, 0)) == ERA) {
        stk_push(chrs, '_');
      } else {
        stk_push(chrs, 'x');
        readback_decimal(chrs, stk_find(vars, Var(get_loc(term, 0))));
      };
      stk_push(chrs, ' ');
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      break;
    }
    case APP: {
      stk_push(chrs, '(');
      readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ' ');
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ')');
      break;
    }
    case SUP: {
      u64 col = get_ext(term);
      if (dirs[col].size > 0) {
        u64 head = stk_pop(&dirs[col]);
        if (head == 0) {
          readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
          stk_push(&dirs[col], head);
        } else {
          readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
          stk_push(&dirs[col], head);
        }
      } else {
        stk_push(chrs, '<');
        readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
        stk_push(chrs, ' ');
        readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
        stk_push(chrs, '>');
      }
      break;
    }
    case DP0: case DP1: {
      u64 col = get_ext(term);
      u64 val = ask_arg(mem, term, 2);
      stk_push(&dirs[col], get_tag(term) == DP0 ? 0 : 1);
      readback_term(chrs, mem, ask_arg(mem, term, 2), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_pop(&dirs[col]);
      break;
    }
    case OP2: {
      stk_push(chrs, '(');
      readback_term(chrs, mem, ask_arg(mem, term, 0), vars, dirs, id_to_name_data, id_to_name_mcap);
      switch (get_ext(term)) {
        case ADD: { stk_push(chrs, '+'); break; }
        case SUB: { stk_push(chrs, '-'); break; }
        case MUL: { stk_push(chrs, '*'); break; }
        case DIV: { stk_push(chrs, '/'); break; }
        case MOD: { stk_push(chrs, '%'); break; }
        case AND: { stk_push(chrs, '&'); break; }
        case OR: { stk_push(chrs, '|'); break; }
        case XOR: { stk_push(chrs, '^'); break; }
        case SHL: { stk_push(chrs, '<'); stk_push(chrs, '<'); break; }
        case SHR: { stk_push(chrs, '>'); stk_push(chrs, '>'); break; }
        case LTN: { stk_push(chrs, '<'); break; }
        case LTE: { stk_push(chrs, '<'); stk_push(chrs, '='); break; }
        case EQL: { stk_push(chrs, '='); stk_push(chrs, '='); break; }
        case GTE: { stk_push(chrs, '>'); stk_push(chrs, '='); break; }
        case GTN: { stk_push(chrs, '>'); break; }
        case NEQ: { stk_push(chrs, '!'); stk_push(chrs, '='); break; }
      }
      readback_term(chrs, mem, ask_arg(mem, term, 1), vars, dirs, id_to_name_data, id_to_name_mcap);
      stk_push(chrs, ')');
      break;
    }
    case NUM: {
      //printf("- u32\n");
      readback_decimal(chrs, get_num(term));
      //printf("- u32 done\n");
      break;
    }
    case CTR: case FUN: {
      u64 func = get_ext(term);
      u64 arit = ask_ari(mem, term);
      stk_push(chrs, '(');
      if (func < id_to_name_mcap && id_to_name_data[func] != NULL) {
        for (u64 i = 0; id_to_name_data[func][i] != '\0'; ++i) {
          stk_push(chrs, id_to_name_data[func][i]);
        }
      } else {
        stk_push(chrs, '$');
        readback_decimal(chrs, func); // TODO: function names
      }
      for (u64 i = 0; i < arit; ++i) {
        stk_push(chrs, ' ');
        readback_term(chrs, mem, ask_arg(mem, term, i), vars, dirs, id_to_name_data, id_to_name_mcap);
      }
      stk_push(chrs, ')');
      break;
    }
    case VAR: {
      stk_push(chrs, 'x');
      readback_decimal(chrs, stk_find(vars, term));
      break;
    }
    default: {
      stk_push(chrs, '?');
      break;
    }
  }
}

void readback(char* code_data, u64 code_mcap, Worker* mem, Ptr term, char** id_to_name_data, u64 id_to_name_mcap) {
  //printf("reading back\n");

  // Used vars
  Stk seen;
  Stk chrs;
  Stk vars;
  Stk* dirs;

  // Initialization
  stk_init(&seen);
  stk_init(&chrs);
  stk_init(&vars);
  dirs = (Stk*)malloc(sizeof(Stk) * DIRS_MCAP);
  assert(dirs);
  for (u64 i = 0; i < DIRS_MCAP; ++i) {
    stk_init(&dirs[i]);
  }

  // Readback
  readback_vars(&vars, mem, term, &seen);
  readback_term(&chrs, mem, term, &vars, dirs, id_to_name_data, id_to_name_mcap);

  // Generates C string
  for (u64 i = 0; i < chrs.size && i < code_mcap; ++i) {
    code_data[i] = chrs.data[i];
  }
  code_data[chrs.size < code_mcap ? chrs.size : code_mcap] = '\0';

  // Cleanup
  stk_free(&seen);
  stk_free(&chrs);
  stk_free(&vars);
  for (u64 i = 0; i < DIRS_MCAP; ++i) {
    stk_free(&dirs[i]);
  }
}

// Debug
// -----

// void debug_print_lnk(Ptr x) {
//   u64 tag = get_tag(x);
//   u64 ext = get_ext(x);
//   u64 val = get_val(x);
//   switch (tag) {
//     case DP0: printf("DP0"); break;
//     case DP1: printf("DP1"); break;
//     case VAR: printf("VAR"); break;
//     case ARG: printf("ARG"); break;
//     case ERA: printf("ERA"); break;
//     case LAM: printf("LAM"); break;
//     case APP: printf("APP"); break;
//     case SUP: printf("SUP"); break;
//     case CTR: printf("CTR"); break;
//     case FUN: printf("FUN"); break;
//     case OP2: printf("OP2"); break;
//     case NUM: printf("NUM"); break;
//     case FLO: printf("FLO"); break;
//     case NIL: printf("NIL"); break;
//     default : printf("???"); break;
//   }
//   printf(":%"PRIx64":%"PRIx64"", ext, val);
// }

// Main
// ----

Ptr parse_arg(char* code, char** id_to_name_data, u64 id_to_name_size) {
  if (code[0] >= '0' && code[0] <= '9') {
    return Num(strtol(code, 0, 10));
  } else {
    return Num(0);
  }
}

int main(int argc, char* argv[]) {

  Worker mem;

  // Id-to-Name map
  const u64 id_to_name_size = /*! GENERATED_NAME_COUNT */ 1 /* GENERATED_NAME_COUNT !*/;
  char* id_to_name_data[id_to_name_size];
/*! GENERATED_ID_TO_NAME_DATA !*/

  // Id-to-Arity map
  const u64 id_to_arity_size = /*! GENERATED_ARITY_COUNT */ 1 /* GENERATED_ARITY_COUNT !*/;
  u64 id_to_arity_data[id_to_arity_size];
/*! GENERATED_ID_TO_ARITY_DATA !*/

  // Builds main term
  mem.size = 0;
  mem.node = (u64*)malloc(HEAP_SIZE);
  mem.aris = id_to_arity_data;
  mem.funs = id_to_arity_size;
  assert(mem.node);
  if (argc <= 1) {
    mem.node[mem.size++] = Cal(0, _MAIN_, 0);
  } else {
    mem.node[mem.size++] = Cal(argc - 1, _MAIN_, 1);
    for (u64 i = 1; i < argc; ++i) {
      mem.node[mem.size++] = parse_arg(argv[i], id_to_name_data, id_to_name_size);
    }
  }

  for (u64 tid = 0; tid < MAX_WORKERS; ++tid) {
    workers[tid].aris = id_to_arity_data;
    workers[tid].funs = id_to_arity_size;
  }

  // Reduces, performing IO when the program asks for it
  
  void * io_state = io_setup();
  do {
    ffi_normal((u8*)mem.node, mem.size, 0);
  } while (io_step (io_state, mem.node));

  // Prints result normal form
  const u64 code_mcap = 256 * 256 * 256; // max code size = 16 MB
  char* code_data = (char*)malloc(code_mcap * sizeof(char));
  assert(code_data);
  readback(code_data, code_mcap, &mem, mem.node[0], id_to_name_data, id_to_name_size);
  printf("%s\n", code_data);

  // Cleanup
  free(code_data);
  free(mem.node);
}
