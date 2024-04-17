// HVM-CUDA: an Interaction Combinator evaluator in CUDA.
// 
// # Format
// 
// An HVM net is a graph with 8 node types:
// - *       ::= ERAser node.
// - #N      ::= NUMber node.
// - @def    ::= REFerence node.
// - x       ::= VARiable node.
// - (a b)   ::= CONstructor node.
// - {a b}   ::= DUPlicator node.
// - <+ a b> ::= OPErator node.
// - ?<a b>  ::= SWItch node.
// 
// Nodes form a tree-like structure in memory. For example:
// 
//     ((* x) {x (y y)})
// 
// Represents a tree with 3 CON nodes, 1 ERA node and 4 VAR nodes.
// 
// A net consists of a root tree, plus list of redexes. Example:
// 
//     (a b)
//     & (b a) ~ (x (y *)) 
//     & {y x} ~ @foo
// 
// The net above has a root and 2 redexes (in the shape `& A ~ B`).
// 
// # Interactions 
// 
// Redexes are reduced via *interaction rules*:
// 
// ## 0. LINK
// 
//     a ~ b
//     ------ LINK
//     a ~> b
//
// ## 1. CALL
// 
//     @foo ~ B
//     ---------------
//     deref(@foo) ~ B
// 
// ## 2. VOID
// 
//     * ~ *
//     -----
//     void
// 
// ## 3. ERAS
// 
//     (A1 A2) ~ *
//     -----------
//     A1 ~ *
//     A2 ~ *
//     
// ## 4. ANNI (https://i.imgur.com/ASdOzbg.png)
//
//     (A1 A2) ~ (B1 B2)
//     -----------------
//     A1 ~ B1
//     A2 ~ B2
//     
// ## 5. COMM (https://i.imgur.com/gyJrtAF.png)
// 
//     (A1 A2) ~ {B1 B2}
//     -----------------
//     A1 ~ {x y}
//     A2 ~ {z w}
//     B1 ~ (x z)
//     B2 ~ (y w)
// 
// ## 6. OPER
// 
//     #A ~ <+ B1 B2>
//     --------------
//     if B1 is #B:
//       B2 ~ #A+B
//     else:
//       B1 ~ <+ #A B2>
//
// ## 7. SWIT
// 
//     #A ~ ?<B1 B2>
//     -------------
//     if A == 0:
//       B1 ~ (B2 *)
//     else:
//       B1 ~ (* (#A-1 B2))
// 
// # Interaction Table
// 
// | A\B |  VAR |  REF |  ERA |  NUM |  CON |  DUP |  OPR |  SWI |
// |-----|------|------|------|------|------|------|------|------|
// | VAR | LINK | CALL | LINK | LINK | LINK | LINK | LINK | LINK |
// | REF | CALL | VOID | VOID | VOID | CALL | CALL | CALL | CALL |
// | ERA | LINK | VOID | VOID | VOID | ERAS | ERAS | ERAS | ERAS |
// | NUM | LINK | VOID | VOID | VOID | ERAS | ERAS | OPER | CASE |
// | CON | LINK | CALL | ERAS | ERAS | ANNI | COMM | COMM | COMM |
// | DUP | LINK | CALL | ERAS | ERAS | COMM | ANNI | COMM | COMM |
// | OPR | LINK | CALL | ERAS | OPER | COMM | COMM | ANNI | COMM |
// | SWI | LINK | CALL | ERAS | CASE | COMM | COMM | COMM | ANNI |
// 
// # Definitions
// 
// A top-level definition is just a statically known closed net, also called a
// package. It is represented like a net, with a root and linked trees:
// 
//     @foo = (a b)
//     & @tic ~ (x a)
//     & @tac ~ (x b)
// 
// The statement above represents a definition, @foo, with the `(a b)` tree as
// the root, and two linked trees: `@tic ~ (x a)` and `@tac ~ (x b)`. When a
// REF is part of a redex, it expands to its complete value. For example:
// 
//     & @foo ~ ((* a) (* a))
// 
// Expands to:
// 
//     & (a0 b0) ~ ((* a) (* a))
//     & @tic ~ (x0 a0)
//     & @tac ~ (x0 b0)
// 
// As an optimization, `@foo ~ {a b}` and `@foo ~ *` will NOT expand; instead,
// it will copy or erase when it is safe to do so.
// 
// # Example Reduction
// 
// Consider the first example, which had 2 redexes. HVM is strongly confluent,
// thus, we can reduce them in any order, even in parallel, with no effect on
// the total work done. Below is its complete reduction:
// 
//     (a b) & (b a) ~ (x (y *)) & {y x} ~ @foo
//     ----------------------------------------------- ANNI
//     (a b) & b ~ x & a ~ (y *) & {y x} ~ @foo
//     ----------------------------------------------- COMM
//     (a b) & b ~ x & a ~ (y *) & y ~ @foo & x ~ @foo
//     ----------------------------------------------- LINK `y` and `x`
//     (a b) & b ~ @foo & a ~ (@foo *)
//     ----------------------------------------------- LINK `b` and `a`
//     ((@foo *) @foo)
//     ----------------------------------------------- CALL `@foo` (optional)
//     ((a0 b0) (a1 b1))
//     & @tic ~ (x0 a0) & @tac ~ (x0 b0)
//     & @tic ~ (x1 a1) & @tac ~ (x1 b1)
//     ----------------------------------------------- CALL `@tic` and `@tac`
//     ((a0 b0) (a1 b1))
//     & (k0 k0) ~ (x0 a0) & (k1 k1) ~ (x0 b0)
//     & (k2 k2) ~ (x1 a1) & (k3 k3) ~ (x1 b1)
//     ----------------------------------------------- ANNI (many in parallel)
//     ((a0 b0) (a1 b1))
//     & k0 ~ x0 & k0 ~ a0 & k1 ~ x0 & k1 ~ b0
//     & k2 ~ x1 & k2 ~ a1 & k3 ~ x1 & k3 ~ b1
//     ----------------------------------------------- LINK `kN`
//     ((a0 b0) (a1 b1))
//     & x0 ~ a0 & x0 ~ b0 & x1 ~ a1 & x1 ~ b1
//     ----------------------------------------------- LINK `xN`
//     ((a0 b0) (a1 b1)) & a0 ~ b0 & a1 ~ b1
//     ----------------------------------------------- LINK `aN`
//     ((b0 b0) (b1 b1))
// 
// # Memory Layout
// 
// An HVM-CUDA net includes a redex bag, a node buffer and a vars buffer:
//
//     LNet ::= { RBAG: [Pair], NODE: [Pair], VARS: [Port] }
// 
// A Pair consists of two Ports, representing a either a redex or a node:
// 
//     Pair ::= (Port, Port)
// 
// A Port consists of a tag and a value:
// 
//     Port ::= 3-bit tag + 13-bit val
// 
// There are 8 Tags:
// 
//     Tag ::=
//       | VAR ::= a variable
//       | REF ::= a reference
//       | ERA ::= an eraser
//       | NUM ::= numeric literal
//       | CON ::= a constructor
//       | DUP ::= a duplicator
//       | OPR ::= numeric binary op
//       | SWI ::= numeric switch
// 
// ## Memory Layout Example
// 
// Consider, again, the following net:
// 
//     (a b)
//     & (b a) ~ (x (y *)) 
//     & {y x} ~ @foo
// 
// In memory, it could be represented as, for example:
// 
// - RBAG | FST-TREE | SND-TREE
// - ---- | -------- | --------
// - 0800 | CON 0001 | CON 0002 // '& (b a) ~ (x (y *))'
// - 1800 | DUP 0005 | REF 0000 // '& {x y} ~ @foo'
// - ---- | -------- | --------
// - NODE | PORT-1   | PORT-2
// - ---- | -------- | --------
// - 0000 | CON 0001 |          // points to root node
// - 0001 | VAR 0000 | VAR 0001 // '(a b)' node (root)
// - 0002 | VAR 0001 | VAR 0000 // '(b a)' node
// - 0003 | VAR 0002 | CON 0004 // '(x (y *))' node
// - 0004 | VAR 0003 | DUP 0000 // '(y *)' node
// - 0005 | VAR 0003 | VAR 0002 // '{y x}' node
// - ---- | -------- | --------

#include <stdint.h>
#include <stdio.h>

// Integers
// --------

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;

// Configuration
// -------------

// Clocks per Second
const u64 S = 2520000000;

// Threads per Block
const u32 TPB_L2 = 8;
const u32 TPB    = 1 << TPB_L2;

// Blocks per GPU
const u32 BPG_L2 = 7;
const u32 BPG    = 1 << BPG_L2;

// Types
// -----

// Local Types
typedef u8  Tag;  // Tag  ::= 3-bit (rounded up to u8)
typedef u32 Val;  // Val  ::= 29-bit (rounded up to u32)
typedef u32 Port; // Port ::= Tag + Val (fits a u32)
typedef u64 Pair; // Pair ::= Port + Port (fits a u64)

// Rules
typedef u8 Rule; // Rule ::= 3-bit (rounded up to 8)

// Tags
const Tag VAR = 0x0; // variable
const Tag REF = 0x1; // reference
const Tag ERA = 0x2; // eraser
const Tag NUM = 0x3; // number
const Tag CON = 0x4; // constructor
const Tag DUP = 0x5; // duplicator
const Tag OPR = 0x6; // operator
const Tag SWI = 0x7; // switch

// Port Number
const Tag P1 = 0; // PORT-1
const Tag P2 = 1; // PORT-2

// Interaction Rule Values
const Rule LINK = 0x0;
const Rule CALL = 0x1;
const Rule VOID = 0x2;
const Rule ERAS = 0x3;
const Rule ANNI = 0x4;
const Rule COMM = 0x5;
const Rule OPER = 0x6;
const Rule SWIT = 0x7;

// Thread Redex Bag Length
const u32 RLEN = 32; // max 32 redexes

// Thread Redex Bag
// It uses the same space to store two stacks: 
// - HI: a high-priotity stack, for shrinking reductions
// - LO: a low-priority stack, for growing reductions
struct RBag {
  u32 lo_idx; // high-priority stack push-index
  u32 hi_idx; // low-priority stack push-index
  Pair buf[RLEN]; // a shared buffer for both stacks
};

// Local Net
const u32 L_NODE_LEN = 0x2000; // max 8196 nodes
const u32 L_VARS_LEN = 0x2000; // max 8196 vars
struct LNet {
  Pair node_buf[L_NODE_LEN];
  Port vars_buf[L_VARS_LEN];
};

// Global Net
const u32 G_PAGE_MAX = 0x10000; // max 65536 pages
const u32 G_NODE_LEN = G_PAGE_MAX * L_NODE_LEN; // max 536m nodes 
const u32 G_VARS_LEN = G_PAGE_MAX * L_VARS_LEN; // max 536m vars 
const u32 G_RBAG_LEN = TPB * BPG * RLEN; // max 2m redexes
struct GNet {
  Pair rbag_buf[G_RBAG_LEN]; // global redex bag
  Pair node_buf[G_NODE_LEN]; // global node buffer
  Port vars_buf[G_VARS_LEN]; // global vars buffer
  u32  page_len[G_PAGE_MAX]; // node count of each page
  u32  free_buf[G_PAGE_MAX]; // set of free pages
  u32  free_pop; // index to reserve a page
  u32  free_put; // index to release a page
  u64 itrs; // interaction count
};

// View Net: includes both GNet and LNet
struct VNet {
  u32   l_node_len; // local node buffer length
  Pair *l_node_buf; // local node buffer values
  u32   l_vars_len; // local vars buffer length
  Port *l_vars_buf; // local vars buffer values
  u32   g_node_len; // global node buffer length
  Pair *g_node_buf; // global node buffer values
  u32   g_vars_len; // global vars buffer length
  Port *g_vars_buf; // global vars buffer values
  u32   g_page_idx; // selected page index
  u32  *g_page_len; // usage counter of pages
  u32  *g_free_buf; // free pages indexes
  u32  *g_free_pop; // index to reserve a page
  u32  *g_free_put; // index to release a page
};

// Thread Memory
struct TMem {
  RBag rbag; // tmem redex bag
  u32  tick; // tick counter
  u32  page; // page index
  u32  newN; // next node allocation attempt index
  u32  newV; // next vars allocation attempt index
  u32  node_loc[256]; // node allocation indices
  u32  vars_loc[256]; // vars allocation indices
  u32  itrs; // interaction count
};

// Top-Level Definition
struct Def {
  u32  rbag_len;
  Pair rbag_buf[32];
  u32  node_len;
  Pair node_buf[32];
  u32  vars_len;
};

// Book of Definitions
struct Book {
  u32 DEFS_LEN;
  Def DEFS_BUF[64];
};

// Static Book
__constant__ Book D_BOOK;

// Debugger
// --------

struct Show {
  char x[13];
};

__device__ __host__ void put_u16(char* B, u16 val);
__device__ __host__ Show show_port(Port port);
__device__ Show show_rule(Rule rule);
__device__ void print_rbag(RBag* rbag);
__device__ __host__ void print_net(VNet* net);
__device__ void pretty_print_port(VNet* net, Port port);
__device__ void pretty_print_rbag(VNet* net, RBag* rbag);
__device__ u32 count_rbag(RBag* rbag);
__device__ u32 count_node(VNet* net);
__device__ u32 count_vars(VNet* net);

// Port: Constructor and Getters
// -----------------------------

__device__ __host__ inline Port new_port(Tag tag, Val val) {
  return (val << 3) | tag;
}

__device__ __host__ inline Port new_var(u32 val) {
  return new_port(VAR, val);
}

__device__ __host__ inline Port new_ref(u32 val) {
  return new_port(REF, val);
}

__device__ __host__ inline Port new_num(u32 val) {
  return new_port(NUM, val);
}

__device__ __host__ inline Port new_era() {
  return new_port(ERA, 0);
}

__device__ __host__ inline Port new_con(u32 val) {
  return new_port(CON, val);
}

__device__ __host__ inline Port new_dup(u32 val) {
  return new_port(DUP, val);
}

__device__ __host__ inline Port new_opr(u32 val) {
  return new_port(OPR, val);
}

__device__ __host__ inline Port new_swi(u32 val) {
  return new_port(SWI, val);
}

__device__ __host__ inline Port new_nil() {
  return 0;
}

__device__ __host__ inline Tag get_tag(Port port) {
  return port & 7;
}

__device__ __host__ inline Val get_val(Port port) {
  return port >> 3;
}

__device__ __host__ inline Val get_page(Val val) {
  return val / L_NODE_LEN;
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

__device__ __host__ inline Port get_nth(Pair pair, u32 n) {
  return (pair >> (32 * n)) & 0xFFFFFFFF;
}

__device__ __host__ inline void set_nth(Pair* pair, Port val, u32 n) {
  *(((u32*)pair)+n) = val;
}

// Utils
// -----

// Swaps two ports.
__device__ __host__ inline void swap(Port *a, Port *b) {
  Port x = *a; *a = *b; *b = x;
}

// ID of peer to share redex with.
__device__ u32 peer_id(u32 id, u32 log2_len, u32 tick) {
  u32 side = (id >> (log2_len - 1 - (tick % log2_len))) & 1;
  u32 diff = (1 << (log2_len - 1)) >> (tick % log2_len);
  return side ? id - diff : id + diff;
}

// Index on the steal redex buffer for this peer pair.
__device__ u32 buck_id(u32 id, u32 log2_len, u32 tick) {
  u32 fid = peer_id(id, log2_len, tick);
  u32 itv = log2_len - (tick % log2_len);
  u32 val = (id >> itv) << (itv - 1);
  return (id < fid ? id : fid) - val;
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
__device__ inline bool block_all(bool x) {
  __shared__ bool res;
  u32 tid = threadIdx.x;
  if (tid == 0) res = true;
  __syncthreads();
  if (!x) res = false;
  __syncthreads();
  return res;
}

// Returns true if any 'x' is true, block-wise
__device__ inline bool block_any(bool x) {
  __shared__ bool res;
  u32 tid = threadIdx.x;
  if (tid == 0) res = false;
  __syncthreads();
  if (x) res = true;
  __syncthreads();
  return res;
}

// Returns the sum of a value, block-wise
__device__ inline u32 block_sum(u32 x) {
  __shared__ u32 res;
  u32 tid = threadIdx.x;
  if (tid == 0) res = 0;
  __syncthreads();
  atomicAdd(&res, x);
  __syncthreads();
  return res;
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

// Given two tags, gets their interaction rule. Uses a u64mask lookup table.
__device__ __host__ inline Rule get_rule(Port A, Port B) {
  const u64 x = 0b0111111010110110110111101110111010110000111100001111000000000000;
  const u64 y = 0b0000110000001100000011000000110011111110111111100000111000000000;
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
__device__ inline Port adjust_port(VNet* net, TMem* tm, Port port) {
  Tag tag = get_tag(port);
  Val val = get_val(port);
  if (is_nod(port)) return new_port(tag, tm->node_loc[val-1]);
  if (is_var(port)) return new_port(tag, tm->vars_loc[val]);
  return new_port(tag, val);
}

// Adjusts a newly allocated pair.
__device__ inline Pair adjust_pair(VNet* net, TMem* tm, Pair pair) {
  Port p1 = adjust_port(net, tm, get_fst(pair));
  Port p2 = adjust_port(net, tm, get_snd(pair));
  return new_pair(p1, p2);
}

// RBag
// ----

__device__ RBag rbag_new() {
  RBag rbag;
  rbag.lo_idx = 0;
  rbag.hi_idx = RLEN - 1;
  return rbag;
}

__device__ void push_redex(TMem* tm, Pair redex) {
  Rule rule = get_pair_rule(redex);
  if (is_high_priority(rule)) {
    tm->rbag.buf[tm->rbag.hi_idx--] = redex;
  } else {
    tm->rbag.buf[tm->rbag.lo_idx++] = redex;
  }
}

__device__ Pair pop_redex(TMem* tm) {
  if (tm->rbag.hi_idx < RLEN - 1) {
    return tm->rbag.buf[++tm->rbag.hi_idx];
  } else if (tm->rbag.lo_idx > 0) {
    return tm->rbag.buf[--tm->rbag.lo_idx];
  } else {
    return 0;
  }
}

__device__ u32 rbag_len(RBag* rbag) {
  return rbag->lo_idx + (RLEN - 1 - rbag->hi_idx);
}

__device__ u32 rbag_has_highs(RBag* rbag) {
  return rbag->hi_idx < RLEN-1;
}

// TMem
// ----

__device__ TMem tmem_new() {
  TMem tm;
  tm.rbag = rbag_new();
  tm.tick = 0;
  tm.newN = threadIdx.x;
  tm.newV = threadIdx.x;
  tm.itrs = 0;
  return tm;
}

// VNet
// ----

__device__ VNet vnet_new(GNet* gnet, void* smem) {
  VNet net;
  net.g_page_idx = 0;
  net.l_node_len = L_NODE_LEN;
  net.l_vars_len = L_VARS_LEN;
  net.l_node_buf = ((LNet*)smem)->node_buf;
  net.l_vars_buf = ((LNet*)smem)->vars_buf;
  net.g_node_len = G_NODE_LEN;
  net.g_vars_len = G_VARS_LEN;
  net.g_node_buf = gnet->node_buf;
  net.g_vars_buf = gnet->vars_buf;
  net.g_page_len = gnet->page_len;
  net.g_free_buf = gnet->free_buf;
  net.g_free_pop = &gnet->free_pop;
  net.g_free_put = &gnet->free_put;
  return net;
}

// Reserves a page.
__device__ u32 reserve_page(VNet* net) {
  u32 free_idx = atomicAdd(net->g_free_pop, 1);
  u32 page_idx = net->g_free_buf[free_idx % G_PAGE_MAX];
  return page_idx;
}

// Releases a page.
__device__ void release_page(VNet* net, u32 page) {
  u32 free_idx = atomicAdd(net->g_free_put, 1);
  net->g_free_buf[free_idx % G_PAGE_MAX] = page;
}

// If page is on global, decreases its length.
__device__ void decrease_page(VNet* net, u32 page) {
  if (page != net->g_page_idx) {
    u32 prev_len = atomicSub(&net->g_page_len[page], 1);
    if (prev_len == 1) {
      release_page(net, page);
    }
  }
}

// Gets the target of a given port.
__device__ __host__ inline Pair* node_ref(VNet* net, u32 loc) {
  if (get_page(loc) == net->g_page_idx) {
    return &net->l_node_buf[loc - L_NODE_LEN*net->g_page_idx];
  } else {
    return &net->g_node_buf[loc];
  }
}

// Gets the target of a given port.
__device__ __host__ inline Port* vars_ref(VNet* net, u32 var) {
  if (get_page(var) == net->g_page_idx) {
    return &net->l_vars_buf[var - L_VARS_LEN*net->g_page_idx];
  } else {
    return &net->g_vars_buf[var];
  }
}

// Takes a node.
__device__ inline Pair node_take(VNet* net, u32 loc) {
  decrease_page(net, get_page(loc));
  return atomicExch(node_ref(net,loc), 0);
}

// Takes a var.
__device__ inline Port vars_take(VNet* net, u32 var) {
  decrease_page(net, get_page(var));
  return atomicExch(vars_ref(net,var), 0);
}

// GNet
// ----

// Initializes a Global Net.
__global__ void gnet_init(GNet* gnet) {
  u32 gid = threadIdx.x + blockIdx.x * blockDim.x;
  // Adds all pages to the free buffer.
  if (gid < G_PAGE_MAX) {
    gnet->free_buf[gid] = gid;
  }
}

// Allocator
// ---------

// Allocates empty slots in an array.
template <typename A>
__device__ u32 alloc(u32* idx, u32* res, u32 num, A* arr, u32 len, u32 add) {
  u32 tid = threadIdx.x;
  u32 got = 0;
  u32 lps = 0;
  while (got < num) {
    u32 elem = arr[*idx];
    if (elem == 0 && *idx > 0) {
      res[got++] = *idx + add;
    }
    *idx = (*idx + TPB) % len;
    if (++lps >= len / TPB) {
      return 0;
    }
  }
  return got;
}

// Gets the necessary resources for an interaction.
__device__ bool get_resources(VNet* net, TMem* tm, u8 need_rbag, u8 need_node, u8 need_vars) {
  u32 got_rbag = RLEN - rbag_len(&tm->rbag);
  u32 got_node = alloc(&tm->newN, tm->node_loc, need_node, net->l_node_buf, net->l_node_len, L_NODE_LEN*net->g_page_idx);
  u32 got_vars = alloc(&tm->newV, tm->vars_loc, need_vars, net->l_vars_buf, net->l_vars_len, L_VARS_LEN*net->g_page_idx);
  return got_rbag >= need_rbag && got_node >= need_node && got_vars >= need_vars;
}

// Linking
// -------

// Atomically Links `A ~ B`.
__device__ void link(VNet* net, TMem* tm, Port A, Port B) {
  //printf("LINK %s ~> %s\n", show_port(A).x, show_port(B).x);
  u32 tid = threadIdx.x;
  u32 gid = tid + blockIdx.x * blockDim.x;

  // Attempts to directionally point `A ~> B`
  while (true) {
    // If `A` is PRI: swap `A` and `B`, and continue
    if (get_tag(A) != VAR) {
      Port X = A; A = B; B = X;
    }

    // If `A` is PRI: create the `A ~ B` redex
    if (get_tag(A) != VAR) {
      push_redex(tm, new_pair(A, B)); // TODO: move global ports to local
      break;
    }

    // While `B` is VAR: extend it (as an optimization)
    while (get_tag(B) == VAR) {
      // Takes the current `B` substitution as `B'`
      Port B_ = atomicExch(vars_ref(net, get_val(B)), B);
      // If there was no `B'`, stop, as there is no extension
      if (B_ == B || B_ == 0) {
        break;
      }
      // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
      vars_take(net, get_val(B));
      B = B_;
    }

    // Since `A` is VAR: point `A ~> B`.
    if (true) {
      // Stores `A -> B`, taking the current `A` subst as `A'`
      Port A_ = atomicExch(vars_ref(net, get_val(A)), B);
      // If there was no `A'`, stop, as we lost B's ownership
      if (A_ == A) {
        break;
      }
      //if (A_ == 0) { ??? } // FIXME: must handle on the move-to-global algo
      // Otherwise, delete `A` (we own both) and link `A' ~ B`
      vars_take(net, get_val(A));
      A = A_;
    }
  }
}

// Links `A ~ B` (as a pair).
__device__ void link_pair(VNet* net, TMem* tm, Pair AB) {
  link(net, tm, get_fst(AB), get_snd(AB));
}

// Sharing
// -------

// Sends redex to a friend local thread, when it is starving.
__device__ void share_redexes(TMem* tm, Pair* steal, u32 tid) {

  //// Gets the peer ID
  //u32 pid = peer_id(tid, TPB_L2, tm->tick);
  //u32 idx = buck_id(tid, TPB_L2, tm->tick);
  //bool lt = tid < pid;

  //// Sends my redex count to peer
  //u32 self_redex_count = tm->rbag.lo_idx;
  //set_nth(&steal[idx], self_redex_count, lt ? 0 : 1);
  //__syncthreads();

  //// Receives peer's redex count 
  //u32 peer_redex_count = (u32)get_nth(steal[idx], lt ? 1 : 0);
  //__syncthreads();

  //// Resets the stolen redex to none
  //steal[idx] = 0;
  //__syncthreads();

  //// Causes peer to steal a redex from me
  //if (self_redex_count > 1 && peer_redex_count == 0) {
    //steal[idx] = pop_redex(tm);
  //}
  //__syncthreads();

  //// If we stole a redex from them, add it
  //if (self_redex_count == 0 && steal[idx] != 0) {
    //push_redex(tm, steal[idx]);
  //}
  //__syncthreads();

  const u64 NEED = 0xFFFFFFFFFFFFFFFF;

  // Gets the peer ID
  u32 pid = peer_id(tid, TPB_L2, tm->tick);
  u32 idx = buck_id(tid, TPB_L2, tm->tick);

  // Asks a redex from parent peer
  if (tid > pid) {
    steal[idx] = tm->rbag.lo_idx == 0 ? NEED : 0;
  }
  __syncthreads();

  // Sends a redex to child peer
  if (tid < pid && tm->rbag.lo_idx > 1 && steal[idx] == NEED) {
    steal[idx] = pop_redex(tm);
    //printf("[%04x] send to %04x at %04x\n", tid, pid, idx);
  }
  __syncthreads();

  // Gets a redex from parent peer
  if (tid > pid && tm->rbag.lo_idx == 0 && steal[idx] != NEED) {
    push_redex(tm, steal[idx]);
    //printf("[%04x] rc from %04x at %04x\n", tid, pid, idx);
  }
  __syncthreads();

}


// Interactions
// ------------

// The Link Interaction.
__device__ bool interact_link(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 0, 0)) {
    return false;
  }

  // Links.
  link_pair(net, tm, new_pair(a, b));

  return true;
}

// The Call Interaction.
__device__ bool interact_call(VNet* net, TMem* tm, Port a, Port b) {
  Def* def = &D_BOOK.DEFS_BUF[get_val(a)];

  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, def->rbag_len + 1, def->node_len - 1, def->vars_len)) {
    return false;
  }

  // Stores new vars.
  for (u32 i = 0; i < def->vars_len; ++i) {
    *vars_ref(net, tm->vars_loc[i]) = new_var(tm->vars_loc[i]);
  }

  // Stores new nodes.  
  for (u32 i = 1; i < def->node_len; ++i) {
    *node_ref(net, tm->node_loc[i-1]) = adjust_pair(net, tm, def->node_buf[i]);
  }

  // Links.
  link_pair(net, tm, new_pair(b, adjust_port(net, tm, get_fst(def->node_buf[0]))));
  for (u32 i = 0; i < def->rbag_len; ++i) {
    link_pair(net, tm, adjust_pair(net, tm, def->rbag_buf[i]));
  }

  return true;
}

// The Void Interaction.
__device__ bool interact_void(VNet* net, TMem* tm, Port a, Port b) {
  return true;
}

// The Eras Interaction.
__device__ bool interact_eras(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return false;
  }

  // Checks availability
  if (*node_ref(net,get_val(b)) == 0) {
    //printf("[%04x] unavailable0: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);
    return false;
  }

  // Loads ports.
  Pair B  = atomicExch(node_ref(net,get_val(b)),0);
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  //if (B == 0) printf("[%04x] ERROR2: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);

  // Links.
  link_pair(net, tm, new_pair(a, B1));
  link_pair(net, tm, new_pair(a, B2));

  return true;
}

// The Anni Interaction.
__device__ bool interact_anni(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 2, 0, 0)) {
    return false;
  }

  // Checks availability
  if (*node_ref(net,get_val(a)) == 0 || *node_ref(net,get_val(b)) == 0) {
    //printf("[%04x] unavailable1: %s | %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(a).x, show_port(b).x);
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  //if (A == 0) printf("[%04x] ERROR3: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(a).x);
  //if (B == 0) printf("[%04x] ERROR4: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);

  // Links.
  link_pair(net, tm, new_pair(A1, B1));
  link_pair(net, tm, new_pair(A2, B2));

  return true;
}

// The Comm Interaction.
__device__ bool interact_comm(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 4, 4, 4)) {
    return false;
  }

  // Checks availability
  if (*node_ref(net,get_val(a)) == 0 || *node_ref(net,get_val(b)) == 0) {
    //printf("[%04x] unavailable2: %s | %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(a).x, show_port(b).x);
    return false;
  }

  // Loads ports.
  Pair A  = node_take(net, get_val(a));
  Port A1 = get_fst(A);
  Port A2 = get_snd(A);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  //if (A == 0) printf("[%04x] ERROR5: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(a).x);
  //if (B == 0) printf("[%04x] ERROR6: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);

  // Stores new vars.
  *vars_ref(net, tm->vars_loc[0]) = new_var(tm->vars_loc[0]);
  *vars_ref(net, tm->vars_loc[1]) = new_var(tm->vars_loc[1]);
  *vars_ref(net, tm->vars_loc[2]) = new_var(tm->vars_loc[2]);
  *vars_ref(net, tm->vars_loc[3]) = new_var(tm->vars_loc[3]);

  // Stores new nodes.
  *node_ref(net, tm->node_loc[0]) = new_pair(new_var(tm->vars_loc[0]), new_var(tm->vars_loc[1]));
  *node_ref(net, tm->node_loc[1]) = new_pair(new_var(tm->vars_loc[2]), new_var(tm->vars_loc[3]));
  *node_ref(net, tm->node_loc[2]) = new_pair(new_var(tm->vars_loc[0]), new_var(tm->vars_loc[2]));
  *node_ref(net, tm->node_loc[3]) = new_pair(new_var(tm->vars_loc[1]), new_var(tm->vars_loc[3]));

  // Links.
  link_pair(net, tm, new_pair(A1, new_port(get_tag(b), tm->node_loc[0])));
  link_pair(net, tm, new_pair(A2, new_port(get_tag(b), tm->node_loc[1])));
  link_pair(net, tm, new_pair(B1, new_port(get_tag(a), tm->node_loc[2])));
  link_pair(net, tm, new_pair(B2, new_port(get_tag(a), tm->node_loc[3])));

  return true;
}

// The Oper Interaction.
__device__ bool interact_oper(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.
  if (!get_resources(net, tm, 1, 1, 0)) {
    return false;
  }

  // Checks availability
  if (*node_ref(net,get_val(b)) == 0) {
    //printf("[%04x] unavailable3: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);
    return false;
  }

  // Loads ports.
  u32  av = get_val(a);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  //if (B == 0) printf("[%04x] ERROR8: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);

  // Performs operation.
  if (get_tag(B1) == NUM) {
    u32 bv = get_val(B1);
    u32 rv = av + bv;
    link_pair(net, tm, new_pair(B2, new_num(rv))); 
  } else {
    *node_ref(net, tm->node_loc[0]) = new_pair(a, B2);
    link_pair(net, tm, new_pair(B1, new_opr(tm->node_loc[0])));
  }

  return true;
}

// The Swit Interaction.
__device__ bool interact_swit(VNet* net, TMem* tm, Port a, Port b) {
  // Allocates needed nodes and vars.  
  if (!get_resources(net, tm, 1, 2, 0)) {
    return false;
  }

  // Checks availability
  if (*node_ref(net,get_val(b)) == 0) {
    //printf("[%04x] unavailable4: %s\n", threadIdx.x+blockIdx.x*blockDim.x, show_port(b).x);
    return false;
  }

  // Loads ports.
  u32  av = get_val(a);
  Pair B  = node_take(net, get_val(b));
  Port B1 = get_fst(B);
  Port B2 = get_snd(B);

  // Stores new nodes.  
  if (av == 0) {
    *node_ref(net, tm->node_loc[0]) = new_pair(B2, new_era());
    link_pair(net, tm, new_pair(new_port(CON, tm->node_loc[0]), B1));
  } else {
    *node_ref(net, tm->node_loc[0]) = new_pair(new_era(), new_port(CON, tm->node_loc[1]));
    *node_ref(net, tm->node_loc[1]) = new_pair(new_num(av-1), B2);
    link_pair(net, tm, new_pair(new_port(CON, tm->node_loc[0]), B1));
  }

  return true;
}

// Pops a local redex and performs a single interaction.
__device__ bool interact(VNet* net, TMem* tm) {
  u32 tid = threadIdx.x;

  // Pops a redex.
  Pair redex = pop_redex(tm);

  // If there is no redex, stop.
  if (redex != 0) {
    // Gets redex ports A and B.
    Port a = get_fst(redex);
    Port b = get_snd(redex);

    // Gets the rule type.
    Rule rule = get_rule(a, b);

    //if (tid == 0) {
      //printf("[%04x] REDUCE %s ~ %s | %s\n", tid, show_port(a).x, show_port(b).x, show_rule(rule).x);
    //}

    // Used for root redex.
    if (get_tag(a) == REF && get_tag(b) == VAR) {
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
    } else {
      tm->itrs += 1;
    }
  }

  return true;
}

// Evaluator
// ---------

__global__ void evaluator(GNet* gnet, u32 turn) {
  extern __shared__ char shared_mem[]; // 96 KB

  // Shared values
  __shared__ Pair steal[TPB/2 > 0 ? TPB/2 : 1];

  // Thread Index
  const u32 tid = threadIdx.x;
  const u32 bid = blockIdx.x;
  const u32 gid = blockDim.x * bid + tid;
  const u32 rbg = turn % 2 ? transpose(gid, TPB, BPG) : gid;

  // Thread TMem
  TMem tm = tmem_new();

  // Net (Local-Global View)
  VNet net = vnet_new(gnet, shared_mem);

  // Loads Redexes
  for (u32 i = 0; i < RLEN; ++i) {
    Pair redex = atomicExch(&gnet->rbag_buf[rbg * RLEN + i], 0);
    if (redex != 0) {
      push_redex(&tm, redex);
    } else {
      break;
    }
  }

  // Aborts if empty
  if (block_all(rbag_len(&tm.rbag) == 0)) {
    return;
  }

  // Allocates Page
  __shared__ u32 got_page;
  if (tid == 0) {
    got_page = reserve_page(&net);
  }
  __syncthreads();
  net.g_page_idx = got_page;
  if (net.g_page_idx >= G_PAGE_MAX) {
    return;
  }

  // Interaction Loop
  for (u32 i = 0; i < 1 << 14; ++i) {
    // Increments the tick
    tm.tick += 1;

    // Performs some interactions
    if (interact(&net, &tm)) {
      while (rbag_has_highs(&tm.rbag)) {
        if (!interact(&net, &tm)) break;
      }
    }

    // Shares a redex with neighbor thread
    if (TPB > 1) {
      share_redexes(&tm, steal, tid);
    }

    // If turn 0 and all threads are full, halt
    if (turn == 0 && block_all(rbag_len(&tm.rbag) > 0)) {
      break;
    }
  }

  // Reports heatmap (debug)
  //const u32 D = 2;
  //__shared__ u32 heat[TPB/D];
  //heat[tid/D] = 0;
  //__syncthreads();
  //atomicAdd(&heat[tid/D], tm.itrs);

  // Moves vars+node to global
  u32 count = 0;
  for (u32 i = tid; i < L_NODE_LEN; i += TPB) {
    Pair node = atomicExch(&net.l_node_buf[i], 0);
    if (node != 0) {
      net.g_node_buf[L_NODE_LEN*net.g_page_idx+i] = node;
      count += 1;
      //if (turn > 0) {
        //printf("[%04x] store turn=%d [%04x:%04x] <- (%s,%s)\n", gid, turn, net.g_page_idx, i, show_port(get_fst(node)).x, show_port(get_snd(node)).x);
      //}
    }
  }
  for (u32 i = tid; i < L_VARS_LEN; i += TPB) {
    // FIXME: handle var not being 0 (when other thread linked it).
    Port var = atomicExch(&net.l_vars_buf[i],0);
    if (var != 0) {
      net.g_vars_buf[L_VARS_LEN*net.g_page_idx+i] = var;
      count += 1;
    }
  }

  // Moves rbag to global
  u32 idx = 0;
  for (u32 i = 0; i < tm.rbag.lo_idx; ++i) {
    gnet->rbag_buf[rbg * RLEN + (idx++)] = tm.rbag.buf[i];
  }
  for (u32 i = RLEN-1; i > tm.rbag.hi_idx; --i) {
    gnet->rbag_buf[rbg * RLEN + (idx++)] = tm.rbag.buf[i];
  }

  // Stores count and rewrites
  atomicAdd(&gnet->page_len[net.g_page_idx], count);

  // Stores rewrites
  atomicAdd(&gnet->itrs, tm.itrs);
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

__device__ void print_rbag(RBag* rbag) {
  printf("RBAG | FST-TREE     | SND-TREE    \n");
  printf("---- | ------------ | ------------\n");
  for (u32 i = 0; i < rbag->lo_idx; ++i) {
    Pair redex = rbag->buf[i];
    printf("%04X | %s | %s\n", i, show_port((Port)get_fst(redex)).x, show_port((Port)get_snd(redex)).x);
  }

  for (u32 i = 15; i > rbag->hi_idx; --i) {
    Pair redex = rbag->buf[i];
    printf("%04X | %s | %s\n", i, show_port((Port)get_fst(redex)).x, show_port((Port)get_snd(redex)).x);
  }
  printf("==== | ============ | ============\n");
}

__device__ __host__ void print_net(VNet* net) {
  printf("NODE | PORT-1       | PORT-2      \n");
  printf("---- | ------------ | ------------\n");
  for (u32 i = 0; i < net->g_node_len; ++i) {
    Pair node = *node_ref(net, i);
    if (node != 0) {
      printf("%04X | %s | %s\n", i, show_port(get_fst(node)).x, show_port(get_snd(node)).x);
    }
  }
  printf("==== | ============ |\n");
  printf("VARS | VALUE        |\n");
  printf("---- | ------------ |\n");
  for (u32 i = 0; i < net->g_vars_len; ++i) {
    Port var = *vars_ref(net,i);
    if (var != 0) {
      printf("%04X | %s |\n", i, show_port(*vars_ref(net,i)).x);
    }
  }
  printf("==== | ============ |\n");
}

__device__ void pretty_print_port(VNet* net, Port port) {
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
        Pair node = *node_ref(net,get_val(cur));
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
        Port got = *vars_ref(net, get_val(cur));
        if (got != cur) {
          printf("=");
          stack[len++] = got;
        }
        break;
      }
      case NUM: {
        printf("#%d", get_val(cur));
        break;
      }
      case DUP: {
        Pair node = *node_ref(net,get_val(cur));
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
        Pair node = *node_ref(net,get_val(cur));
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
        Pair node = *node_ref(net,get_val(cur));
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

__device__ void pretty_print_rbag(VNet* net, RBag* rbag) {
  for (u32 i = 0; i < rbag->lo_idx; ++i) {
    Pair redex = rbag->buf[i];
    if (redex != 0) {
      pretty_print_port(net, get_fst(redex)); 
      printf(" ~ ");
      pretty_print_port(net, get_snd(redex));
      printf("\n");
    }
  }
  for (u32 i = RLEN-1; i > rbag->hi_idx; --i) {
    Pair redex = rbag->buf[i];
    if (redex != 0) {
      pretty_print_port(net, get_fst(redex));
      printf(" ~ ");
      pretty_print_port(net, get_snd(redex));
      printf("\n");
    }
  }
}

// Countest he number of non-none redexes in a bag.
__device__ u32 count_rbag(RBag* rbag) {
  u32 count = 0;
  for (u32 i = 0; i < rbag->lo_idx; ++i) {
    if (rbag->buf[i] != 0) {
      count++;
    }
  }
  for (u32 i = RLEN-1; i > rbag->hi_idx; --i) {
    if (rbag->buf[i] != 0) {
      count++;
    }
  }
  return count;
}

// Counts the number of non-zero nodes in a net.
__device__ __host__ u32 count_node(VNet* net) {
  u32 count = 0;
  for (u32 i = 0; i < net->g_node_len; ++i) {
    if (*node_ref(net,i) != 0) {
      count++;
    }
  }
  return count;
}

// Counts the number of non-zero vars in a net.
__device__ __host__ u32 count_vars(VNet* net) {
  u32 count = 0;
  for (u32 i = 0; i < net->g_vars_len; ++i) {
    if (*vars_ref(net,i) != 0) {
      count++;
    }
  }
  return count;
}

// Example Books
// -------------

// TPB=8 - LOOPS=65536 - DEPTH=10 => IT=469783543 | 1.35s
const u32 DEPTH = 2;
const u32 LOOPS = 65536;

//const u32 LOOPS = 8192;
//const u32 DEPTH = 22;

const Book BOOK = {
  6,
  {
    { // fun
      0, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      4, { 0x000000000000000C, 0x000000000000001F, 0x0000001100000009, 0x0000000000000014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      1,
    },
    { // fun$C0
      1, { 0x0000000C00000019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      2, { 0x0000000000000000, new_pair(new_num(LOOPS), new_var(0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      1,
    },
    { // fun$C1
      2, { 0x0000001C00000001, 0x0000002C00000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      6, { 0x000000000000000C, 0x0000001000000015, 0x0000000800000000, 0x0000002600000000, 0x0000001000000018, 0x0000001800000008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      4,
    },
    { // loop
      0, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      4, { 0x000000000000000C, 0x000000000000001F, 0x0000002100000003, 0x0000000000000014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      1,
    },
    { // loop$C0
      1, { 0x0000001400000019, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      3, { 0x000000000000000C, 0x0000000800000000, 0x0000000800000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      2,
    },
    { // main
      1, { 0x0000000C00000001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      2, { 0x0000000000000000, new_pair(new_num(DEPTH), new_var(0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
      1,
    },
  }
};

// Main
// ----

int main() {

  // GMem
  GNet *d_gnet;
  cudaMalloc((void**)&d_gnet, sizeof(GNet));
  cudaMemset(&d_gnet, 0, sizeof(GNet));

  // Set the initial redex
  Pair pair = new_pair(new_ref(5), new_var(0));
  cudaMemcpy(&d_gnet->rbag_buf[0], &pair, sizeof(Pair), cudaMemcpyHostToDevice);

  //printf("GNet size: %lu MB\n", sizeof(GNet) / (1024 * 1024));

  // Shows some info about the evaluator
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, evaluator);
  printf("Shared size: %d\n", (int)attr.sharedSizeBytes);
  printf("Constant size: %d\n", (int)attr.constSizeBytes);
  printf("Local size: %d\n", (int)attr.localSizeBytes);
  printf("Max threads per block: %d\n", attr.maxThreadsPerBlock);
  printf("Num regs: %d\n", attr.numRegs);
  printf("PTX version: %d\n", attr.ptxVersion);
  printf("Binary version: %d\n", attr.binaryVersion);

  // Copy the Book to the constant memory before launching the kernel
  cudaMemcpyToSymbol(D_BOOK, &BOOK, sizeof(Book));

  // Configures Shared Memory Sinze
  cudaFuncSetAttribute(evaluator, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(LNet));

  // Invokes the Kernel
  gnet_init<<<G_PAGE_MAX/TPB, TPB>>>(d_gnet);

  for (u32 i = 0; i < 65536; ++i) {
    evaluator<<<BPG, TPB, sizeof(LNet)>>>(d_gnet, i);
  }

  //evaluator<<<  1, TPB, sizeof(LNet)>>>(d_gnet, 0);
  //evaluator<<<BPG, TPB, sizeof(LNet)>>>(d_gnet, 1);
  //evaluator<<<BPG, TPB, sizeof(LNet)>>>(d_gnet, 2);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch evaluator (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  u64 itrs;
  cudaMemcpy(&itrs, &d_gnet->itrs, sizeof(u64), cudaMemcpyDeviceToHost);
  printf("itrs: %llu\n", itrs);

  // TODO: copy the page_len buffer from d_gnet and print it
  u32 page_len[G_PAGE_MAX];
  cudaMemcpy(page_len, d_gnet->page_len, sizeof(u32)*G_PAGE_MAX, cudaMemcpyDeviceToHost);
  for (u32 i = 0; i < G_PAGE_MAX; ++i) {
    if (page_len[i] > 0) {
      //printf("page %04x: %d\n", i, page_len[i]);
    }
  }

  return 0;
}
