use crate::runtime::{*};

// (+ a b)
// --------- OP2-NUM
// add(a, b)
#[inline(always)]
pub fn apply(heap: &Heap, arit: &Arit, tid: usize, host: u64, term: Ptr, arg0: Ptr, arg1: Ptr) {
  inc_cost(heap, tid);
  let a = get_num(arg0);
  let b = get_num(arg1);
  let c = match get_ext(term) {
    ADD => a.wrapping_add(b) & NUM_MASK,
    SUB => a.wrapping_sub(b) & NUM_MASK,
    MUL => a.wrapping_mul(b) & NUM_MASK,
    DIV => a.wrapping_div(b) & NUM_MASK,
    MOD => a.wrapping_rem(b) & NUM_MASK,
    AND => (a & b) & NUM_MASK,
    OR  => (a | b) & NUM_MASK,
    XOR => (a ^ b) & NUM_MASK,
    SHL => a.wrapping_shl(b as u32) & NUM_MASK,
    SHR => a.wrapping_shr(b as u32) & NUM_MASK,
    LTN => u64::from(a <  b),
    LTE => u64::from(a <= b),
    EQL => u64::from(a == b),
    GTE => u64::from(a >= b),
    GTN => u64::from(a >  b),
    NEQ => u64::from(a != b),
    _   => panic!("Invalid operation!"),
  };
  let done = Num(c);
  link(heap, host, done);
  free(heap, tid, get_loc(term, 0), 2);
}

