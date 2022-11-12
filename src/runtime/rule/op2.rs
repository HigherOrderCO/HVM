use crate::runtime::{*};

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 2));
  ctx.visit.push(new_visit(get_loc(ctx.term, 1), goup));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 0);
  return true;
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = load_arg(ctx.heap, ctx.term, 0);
  let arg1 = load_arg(ctx.heap, ctx.term, 1);

  // (+ a b)
  // --------- OP2-NUM
  // add(a, b)
  if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
    inc_cost(ctx.heap, ctx.tid);
    let a = get_num(arg0);
    let b = get_num(arg1);
    let c = match get_ext(ctx.term) {
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
    link(ctx.heap, *ctx.host, done);
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
    return false;
  }

  // (+ {a0 a1} b)
  // --------------------- OP2-SUP-0
  // dup b0 b1 = b
  // {(+ a0 b0) (+ a1 b1)}
  else if get_tag(arg0) == SUP {
    inc_cost(ctx.heap, ctx.tid);
    let op20 = get_loc(ctx.term, 0);
    let op21 = get_loc(arg0, 0);
    let let0 = alloc(ctx.heap, ctx.tid, 3);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, arg1);
    link(ctx.heap, op20 + 1, Dp0(get_ext(arg0), let0));
    link(ctx.heap, op20 + 0, take_arg(ctx.heap, arg0, 0));
    link(ctx.heap, op21 + 0, take_arg(ctx.heap, arg0, 1));
    link(ctx.heap, op21 + 1, Dp1(get_ext(arg0), let0));
    link(ctx.heap, par0 + 0, Op2(get_ext(ctx.term), op20));
    link(ctx.heap, par0 + 1, Op2(get_ext(ctx.term), op21));
    let done = Sup(get_ext(arg0), par0);
    link(ctx.heap, *ctx.host, done);
    return false;
  }

  // (+ a {b0 b1})
  // --------------- OP2-SUP-1
  // dup a0 a1 = a
  // {(+ a0 b0) (+ a1 b1)}
  else if get_tag(arg1) == SUP {
    inc_cost(ctx.heap, ctx.tid);
    let op20 = get_loc(ctx.term, 0);
    let op21 = get_loc(arg1, 0);
    let let0 = alloc(ctx.heap, ctx.tid, 3);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, arg0);
    link(ctx.heap, op20 + 0, Dp0(get_ext(arg1), let0));
    link(ctx.heap, op20 + 1, take_arg(ctx.heap, arg1, 0));
    link(ctx.heap, op21 + 1, take_arg(ctx.heap, arg1, 1));
    link(ctx.heap, op21 + 0, Dp1(get_ext(arg1), let0));
    link(ctx.heap, par0 + 0, Op2(get_ext(ctx.term), op20));
    link(ctx.heap, par0 + 1, Op2(get_ext(ctx.term), op21));
    let done = Sup(get_ext(arg1), par0);
    link(ctx.heap, *ctx.host, done);
    return false;
  }

  return false;
}
