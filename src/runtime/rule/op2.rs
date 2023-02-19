use crate::runtime::*;

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 2));
  ctx.visit.push(new_visit(get_loc(ctx.term, 1), ctx.hold, goup));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 0);
  true
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = ctx.heap.load_arg(ctx.term, 0);
  let arg1 = ctx.heap.load_arg(ctx.term, 1);

  // (OP a b)
  // -------- OP2-U60
  // op(a, b)
  if arg0.tag() == Tag::U60 && arg1.tag() == Tag::U60 {
    //operate(ctx.heap, ctx.tid, ctx.term, arg0, arg1, *ctx.host);

    ctx.heap.inc_cost(ctx.tid);
    let a = arg0.num();
    let b = arg1.num();
    let c = match ctx.term.oper() {
      Oper::Add => u60::add(a, b),
      Oper::Sub => u60::sub(a, b),
      Oper::Mul => u60::mul(a, b),
      Oper::Div => u60::div(a, b),
      Oper::Mod => u60::mdl(a, b),
      Oper::And => u60::and(a, b),
      Oper::Or => u60::or(a, b),
      Oper::Xor => u60::xor(a, b),
      Oper::Shl => u60::shl(a, b),
      Oper::Shr => u60::shr(a, b),
      Oper::Ltn => u60::ltn(a, b),
      Oper::Lte => u60::lte(a, b),
      Oper::Eql => u60::eql(a, b),
      Oper::Gte => u60::gte(a, b),
      Oper::Gtn => u60::gtn(a, b),
      Oper::Neq => u60::neq(a, b),
    };
    let done = U6O(c);
    ctx.heap.link(*ctx.host, done);
    ctx.heap.free(ctx.tid, get_loc(ctx.term, 0), 2);

    return false;
  }
  // (OP a b)
  // -------- OP2-F60
  // op(a, b)
  else if arg0.tag() == Tag::F60 && arg1.tag() == Tag::F60 {
    //operate(ctx.heap, ctx.tid, ctx.term, arg0, arg1, *ctx.host);

    ctx.heap.inc_cost(ctx.tid);
    let a = arg0.num();
    let b = arg1.num();
    let c = match ctx.term.oper() {
      Oper::Add => f60::add(a, b),
      Oper::Sub => f60::sub(a, b),
      Oper::Mul => f60::mul(a, b),
      Oper::Div => f60::div(a, b),
      Oper::Mod => f60::mdl(a, b),
      Oper::And => f60::and(a, b),
      Oper::Or => f60::or(a, b),
      Oper::Xor => f60::xor(a, b),
      Oper::Shl => f60::shl(a, b),
      Oper::Shr => f60::shr(a, b),
      Oper::Ltn => f60::ltn(a, b),
      Oper::Lte => f60::lte(a, b),
      Oper::Eql => f60::eql(a, b),
      Oper::Gte => f60::gte(a, b),
      Oper::Gtn => f60::gtn(a, b),
      Oper::Neq => f60::neq(a, b),
    };
    let done = F6O(c);
    ctx.heap.link(*ctx.host, done);
    ctx.heap.free(ctx.tid, get_loc(ctx.term, 0), 2);

    return false;
  }
  // (+ {a0 a1} b)
  // --------------------- OP2-SUP-0
  // dup b0 b1 = b
  // {(+ a0 b0) (+ a1 b1)}
  else if arg0.tag() == Tag::SUP {
    ctx.heap.inc_cost(ctx.tid);
    let op20 = get_loc(ctx.term, 0);
    let op21 = get_loc(arg0, 0);
    let let0 = ctx.heap.alloc(ctx.tid, 3);
    let par0 = ctx.heap.alloc(ctx.tid, 2);
    ctx.heap.link(let0 + 2, arg1);
    ctx.heap.link(op20 + 1, Dp0(arg0.ext(), let0));
    ctx.heap.link(op20 + 0, ctx.heap.take_arg(arg0, 0));
    ctx.heap.link(op21 + 0, ctx.heap.take_arg(arg0, 1));
    ctx.heap.link(op21 + 1, Dp1(arg0.ext(), let0));
    ctx.heap.link(par0 + 0, Op2(ctx.term.ext(), op20));
    ctx.heap.link(par0 + 1, Op2(ctx.term.ext(), op21));
    let done = Sup(arg0.ext(), par0);
    ctx.heap.link(*ctx.host, done);
    return false;
  }
  // (+ a {b0 b1})
  // --------------- OP2-SUP-1
  // dup a0 a1 = a
  // {(+ a0 b0) (+ a1 b1)}
  else if arg1.tag() == Tag::SUP {
    ctx.heap.inc_cost(ctx.tid);
    let op20 = get_loc(ctx.term, 0);
    let op21 = get_loc(arg1, 0);
    let let0 = ctx.heap.alloc(ctx.tid, 3);
    let par0 = ctx.heap.alloc(ctx.tid, 2);
    ctx.heap.link(let0 + 2, arg0);
    ctx.heap.link(op20 + 0, Dp0(arg1.ext(), let0));
    ctx.heap.link(op20 + 1, ctx.heap.take_arg(arg1, 0));
    ctx.heap.link(op21 + 1, ctx.heap.take_arg(arg1, 1));
    ctx.heap.link(op21 + 0, Dp1(arg1.ext(), let0));
    ctx.heap.link(par0 + 0, Op2(ctx.term.ext(), op20));
    ctx.heap.link(par0 + 1, Op2(ctx.term.ext(), op21));
    let done = Sup(arg1.ext(), par0);
    ctx.heap.link(*ctx.host, done);
    return false;
  }

  false
}
