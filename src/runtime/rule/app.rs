use crate::runtime::*;

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 0);
  return true;
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = load_arg(ctx.heap, ctx.term, 0);

  // (λx(body) a)
  // ------------ APP-LAM
  // x <- a
  // body
  if get_tag(arg0) == LAM {
    inc_cost(ctx.heap, ctx.tid);
    atomic_subst(
      ctx.heap,
      &ctx.prog.aris,
      ctx.tid,
      Var(get_loc(arg0, 0)),
      take_arg(ctx.heap, ctx.term, 1),
    );
    link(ctx.heap, *ctx.host, take_arg(ctx.heap, arg0, 1));
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);
    free(ctx.heap, ctx.tid, get_loc(arg0, 0), 2);
    return true;
  }

  // ({a b} c)
  // --------------- APP-SUP
  // dup x0 x1 = c
  // {(a x0) (b x1)}
  if get_tag(arg0) == SUP {
    inc_cost(ctx.heap, ctx.tid);
    let app0 = get_loc(ctx.term, 0);
    let app1 = get_loc(arg0, 0);
    let let0 = alloc(ctx.heap, ctx.tid, 3);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, take_arg(ctx.heap, ctx.term, 1));
    link(ctx.heap, app0 + 1, Dp0(get_ext(arg0), let0));
    link(ctx.heap, app0 + 0, take_arg(ctx.heap, arg0, 0));
    link(ctx.heap, app1 + 0, take_arg(ctx.heap, arg0, 1));
    link(ctx.heap, app1 + 1, Dp1(get_ext(arg0), let0));
    link(ctx.heap, par0 + 0, App(app0));
    link(ctx.heap, par0 + 1, App(app1));
    let done = Sup(get_ext(arg0), par0);
    link(ctx.heap, *ctx.host, done);
    return false;
  }

  return false;
}
