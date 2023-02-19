use crate::runtime::*;

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
  *ctx.cont = goup;
  *ctx.host = ctx.term.loc(2);
  true
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {
  let arg0 = ctx.heap.load_arg(ctx.term, 2);
  let tcol = ctx.term.ext();

  // dup r s = λx(f)
  // --------------- DUP-LAM
  // dup f0 f1 = f
  // r <- λx0(f0)
  // s <- λx1(f1)
  // x <- {x0 x1}
  if arg0.tag() == Tag::LAM {
    ctx.heap.inc_cost(ctx.tid);
    let let0 = ctx.heap.alloc(ctx.tid, 3);
    let par0 = ctx.heap.alloc(ctx.tid, 2);
    let lam0 = ctx.heap.alloc(ctx.tid, 2);
    let lam1 = ctx.heap.alloc(ctx.tid, 2);
    ctx.heap.link(let0 + 2, ctx.heap.take_arg(arg0, 1));
    ctx.heap.link(par0 + 1, Var(lam1));
    ctx.heap.link(par0 + 0, Var(lam0));
    ctx.heap.link(lam0 + 1, Dp0(ctx.term.ext(), let0));
    ctx.heap.link(lam1 + 1, Dp1(ctx.term.ext(), let0));
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Var(arg0.loc(0)), Sup(ctx.term.ext(), par0));
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), Lam(lam0));
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), Lam(lam1));
    let done = Lam(if ctx.term.tag() == Tag::DP0 { lam0 } else { lam1 });
    ctx.heap.link(*ctx.host, done);
    ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    ctx.heap.free(ctx.tid, arg0.loc(0), 2);
    true
  }
  // dup x y = {a b}
  // --------------- DUP-SUP
  // if equal: | else:
  // x <- a    | x <- {xA xB}
  // y <- b    | y <- {yA yB}
  //           | dup xA yA = a
  //           | dup xB yB = b
  else if arg0.tag() == Tag::SUP {
    if tcol == arg0.ext() {
      ctx.heap.inc_cost(ctx.tid);
      ctx.heap.atomic_subst(
        &ctx.prog.aris,
        ctx.tid,
        Dp0(tcol, ctx.term.loc(0)),
        ctx.heap.take_arg(arg0, 0),
      );
      ctx.heap.atomic_subst(
        &ctx.prog.aris,
        ctx.tid,
        Dp1(tcol, ctx.term.loc(0)),
        ctx.heap.take_arg(arg0, 1),
      );
      ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
      ctx.heap.free(ctx.tid, arg0.loc(0), 2);
      return true;
    } else {
      ctx.heap.inc_cost(ctx.tid);
      let par0 = ctx.heap.alloc(ctx.tid, 2);
      let let0 = ctx.heap.alloc(ctx.tid, 3);
      let par1 = arg0.loc(0);
      let let1 = ctx.heap.alloc(ctx.tid, 3);
      ctx.heap.link(let0 + 2, ctx.heap.take_arg(arg0, 0));
      ctx.heap.link(let1 + 2, ctx.heap.take_arg(arg0, 1));
      ctx.heap.link(par1 + 0, Dp1(tcol, let0));
      ctx.heap.link(par1 + 1, Dp1(tcol, let1));
      ctx.heap.link(par0 + 0, Dp0(tcol, let0));
      ctx.heap.link(par0 + 1, Dp0(tcol, let1));
      ctx.heap.atomic_subst(
        &ctx.prog.aris,
        ctx.tid,
        Dp0(tcol, ctx.term.loc(0)),
        Sup(arg0.ext(), par0),
      );
      ctx.heap.atomic_subst(
        &ctx.prog.aris,
        ctx.tid,
        Dp1(tcol, ctx.term.loc(0)),
        Sup(arg0.ext(), par1),
      );
      ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
      return true;
    }
  }
  // dup x y = N
  // ----------- DUP-U60
  // x <- N
  // y <- N
  // ~
  else if arg0.tag() == Tag::U60 {
    ctx.heap.inc_cost(ctx.tid);
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), arg0);
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), arg0);
    ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    return true;
  }
  // dup x y = N
  // ----------- DUP-F60
  // x <- N
  // y <- N
  // ~
  else if arg0.tag() == Tag::F60 {
    ctx.heap.inc_cost(ctx.tid);
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), arg0);
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), arg0);
    ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    return true;
  }
  // dup x y = (K a b c ...)
  // ----------------------- DUP-CTR
  // dup a0 a1 = a
  // dup b0 b1 = b
  // dup c0 c1 = c
  // ...
  // x <- (K a0 b0 c0 ...)
  // y <- (K a1 b1 c1 ...)
  else if arg0.tag() == Tag::CTR {
    ctx.heap.inc_cost(ctx.tid);
    let fnum = arg0.ext();
    let fari = arity_of(&ctx.prog.aris, arg0);
    if fari == 0 {
      ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), Ctr(fnum, 0));
      ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), Ctr(fnum, 0));
      ctx.heap.link(*ctx.host, Ctr(fnum, 0));
      ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    } else {
      let ctr0 = arg0.loc(0);
      let ctr1 = ctx.heap.alloc(ctx.tid, fari);
      for i in 0..fari - 1 {
        let leti = ctx.heap.alloc(ctx.tid, 3);
        ctx.heap.link(leti + 2, ctx.heap.take_arg(arg0, i));
        ctx.heap.link(ctr0 + i, Dp0(ctx.term.ext(), leti));
        ctx.heap.link(ctr1 + i, Dp1(ctx.term.ext(), leti));
      }
      let leti = ctx.heap.alloc(ctx.tid, 3);
      ctx.heap.link(leti + 2, ctx.heap.take_arg(arg0, fari - 1));
      ctx.heap.link(ctr0 + fari - 1, Dp0(ctx.term.ext(), leti));
      ctx.heap.link(ctr1 + fari - 1, Dp1(ctx.term.ext(), leti));
      ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), Ctr(fnum, ctr0));
      ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), Ctr(fnum, ctr1));
      ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    }
    return true;
  }
  // dup x y = *
  // ----------- DUP-ERA
  // x <- *
  // y <- *
  else if arg0.tag() == Tag::ERA {
    ctx.heap.inc_cost(ctx.tid);
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp0(tcol, ctx.term.loc(0)), Era());
    ctx.heap.atomic_subst(&ctx.prog.aris, ctx.tid, Dp1(tcol, ctx.term.loc(0)), Era());
    ctx.heap.link(*ctx.host, Era());
    ctx.heap.free(ctx.tid, ctx.term.loc(0), 3);
    return true;
  } else {
    return false;
  }
}
