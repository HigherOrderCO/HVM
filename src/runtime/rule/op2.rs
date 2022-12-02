use crate::runtime::{*};

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 2));
  ctx.visit.push(new_visit(get_loc(ctx.term, 1), ctx.hold, goup));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 0);
  return true;
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = load_arg(ctx.heap, ctx.term, 0);
  let arg1 = load_arg(ctx.heap, ctx.term, 1);

  // (OP a b)
  // -------- OP2-U60
  // op(a, b)
  if get_tag(arg0) == U60 && get_tag(arg1) == U60 {
    //operate(ctx.heap, ctx.tid, ctx.term, arg0, arg1, *ctx.host);

    inc_cost(ctx.heap, ctx.tid);
    let a = get_num(arg0);
    let b = get_num(arg1);
    let c = match get_ext(ctx.term) {
      ADD => u60::add(a, b),
      SUB => u60::sub(a, b),
      MUL => u60::mul(a, b),
      DIV => u60::div(a, b),
      MOD => u60::mdl(a, b),
      AND => u60::and(a, b),
      OR  => u60::or(a, b),
      XOR => u60::xor(a, b),
      SHL => u60::shl(a, b),
      SHR => u60::shr(a, b),
      LTN => u60::ltn(a, b),
      LTE => u60::lte(a, b),
      EQL => u60::eql(a, b),
      GTE => u60::gte(a, b),
      GTN => u60::gtn(a, b),
      NEQ => u60::neq(a, b),
      _   => 0,
    };
    let done = U6O(c);
    link(ctx.heap, *ctx.host, done);
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);

    return false;
  }

  // (OP a b)
  // -------- OP2-F60
  // op(a, b)
  else if get_tag(arg0) == F60 && get_tag(arg1) == F60 {
    //operate(ctx.heap, ctx.tid, ctx.term, arg0, arg1, *ctx.host);

    inc_cost(ctx.heap, ctx.tid);
    let a = get_num(arg0);
    let b = get_num(arg1);
    let c = match get_ext(ctx.term) {
      ADD => f60::add(a, b),
      SUB => f60::sub(a, b),
      MUL => f60::mul(a, b),
      DIV => f60::div(a, b),
      MOD => f60::mdl(a, b),
      AND => f60::and(a, b),
      OR  => f60::or(a, b),
      XOR => f60::xor(a, b),
      SHL => f60::shl(a, b),
      SHR => f60::shr(a, b),
      LTN => f60::ltn(a, b),
      LTE => f60::lte(a, b),
      EQL => f60::eql(a, b),
      GTE => f60::gte(a, b),
      GTN => f60::gtn(a, b),
      NEQ => f60::neq(a, b),
      _   => 0,
    };
    let done = F6O(c);
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
    let let0 = alloc(ctx.heap, ctx.tid, 4);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, arg1);
    link(ctx.heap, let0 + 3, Lck());
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
    let let0 = alloc(ctx.heap, ctx.tid, 4);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, arg0);
    link(ctx.heap, let0 + 3, Lck());
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
