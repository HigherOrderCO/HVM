use crate::runtime::*;

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 0);
  true
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = ctx.heap.load_arg(ctx.term, 0);

  // (λx(body) a)
  // ------------ APP-LAM
  // x <- a
  // body
  if arg0.tag() == Tag::LAM {
    ctx.heap.inc_cost(ctx.tid);
    ctx.heap.atomic_subst(
      &ctx.prog.aris,
      ctx.tid,
      Var(get_loc(arg0, 0)),
      ctx.heap.take_arg(ctx.term, 1),
    );
    ctx.heap.link(*ctx.host, ctx.heap.take_arg(arg0, 1));
    ctx.heap.free(ctx.tid, get_loc(ctx.term, 0), 2);
    ctx.heap.free(ctx.tid, get_loc(arg0, 0), 2);
    return true;
  }

  // ({a b} c)
  // --------------- APP-SUP
  // dup x0 x1 = c
  // {(a x0) (b x1)}
  if arg0.tag() == Tag::SUP {
    ctx.heap.inc_cost(ctx.tid);
    let app0 = get_loc(ctx.term, 0);
    let app1 = get_loc(arg0, 0);
    let let0 = ctx.heap.alloc(ctx.tid, 3);
    let par0 = ctx.heap.alloc(ctx.tid, 2);
    ctx.heap.link(let0 + 2, ctx.heap.take_arg(ctx.term, 1));
    ctx.heap.link(app0 + 1, Dp0(get_ext(arg0), let0));
    ctx.heap.link(app0 + 0, ctx.heap.take_arg(arg0, 0));
    ctx.heap.link(app1 + 0, ctx.heap.take_arg(arg0, 1));
    ctx.heap.link(app1 + 1, Dp1(get_ext(arg0), let0));
    ctx.heap.link(par0 + 0, App(app0));
    ctx.heap.link(par0 + 1, App(app1));
    let done = Sup(get_ext(arg0), par0);
    ctx.heap.link(*ctx.host, done);
    return false;
  }

  false
}
