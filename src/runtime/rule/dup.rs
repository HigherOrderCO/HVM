use crate::runtime::{*};

#[inline(always)]
pub fn visit(ctx: ReduceCtx) -> bool {
  let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, 1));
  *ctx.cont = goup;
  *ctx.host = get_loc(ctx.term, 2);
  return true;
}

#[inline(always)]
pub fn apply(ctx: ReduceCtx) -> bool {

  let arg0 = load_arg(ctx.heap, ctx.term, 2);
  let tcol = get_ext(ctx.term);

  // dup r s = λx(f)
  // --------------- DUP-LAM
  // dup f0 f1 = f
  // r <- λx0(f0)
  // s <- λx1(f1)
  // x <- {x0 x1}
  if get_tag(arg0) == LAM {
    inc_cost(ctx.heap, ctx.tid);
    let let0 = alloc(ctx.heap, ctx.tid, 3);
    let par0 = alloc(ctx.heap, ctx.tid, 2);
    let lam0 = alloc(ctx.heap, ctx.tid, 2);
    let lam1 = alloc(ctx.heap, ctx.tid, 2);
    link(ctx.heap, let0 + 2, take_arg(ctx.heap, arg0, 1));
    link(ctx.heap, par0 + 1, Var(lam1));
    link(ctx.heap, par0 + 0, Var(lam0));
    link(ctx.heap, lam0 + 1, Dp0(get_ext(ctx.term), let0));
    link(ctx.heap, lam1 + 1, Dp1(get_ext(ctx.term), let0));
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Var(get_loc(arg0, 0)), Sup(get_ext(ctx.term), par0));
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), Lam(lam0));
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), Lam(lam1));
    let done = Lam(if get_tag(ctx.term) == DP0 { lam0 } else { lam1 });
    link(ctx.heap, *ctx.host, done);
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
    free(ctx.heap, ctx.tid, get_loc(arg0, 0), 2);
    return true;
  }

  // dup x y = {a b}              
  // --------------- DUP-SUP
  // if equal: | else:        
  // x <- a    | x <- {xA xB} 
  // y <- b    | y <- {yA yB} 
  //           | dup xA yA = a
  //           | dup xB yB = b
  else if get_tag(arg0) == SUP {

    if tcol == get_ext(arg0) {
      inc_cost(ctx.heap, ctx.tid);
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), take_arg(ctx.heap, arg0, 0));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), take_arg(ctx.heap, arg0, 1));
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      free(ctx.heap, ctx.tid, get_loc(arg0, 0), 2);
      return true;

    } else {
      inc_cost(ctx.heap, ctx.tid);
      let par0 = alloc(ctx.heap, ctx.tid, 2);
      let let0 = alloc(ctx.heap, ctx.tid, 3);
      let par1 = get_loc(arg0, 0);
      let let1 = alloc(ctx.heap, ctx.tid, 3);
      link(ctx.heap, let0 + 2, take_arg(ctx.heap, arg0, 0));
      link(ctx.heap, let1 + 2, take_arg(ctx.heap, arg0, 1));
      link(ctx.heap, par1 + 0, Dp1(tcol, let0));
      link(ctx.heap, par1 + 1, Dp1(tcol, let1));
      link(ctx.heap, par0 + 0, Dp0(tcol, let0));
      link(ctx.heap, par0 + 1, Dp0(tcol, let1));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), Sup(get_ext(arg0), par0));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), Sup(get_ext(arg0), par1));
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
      return true;
    }
  }

  // dup x y = N
  // ----------- DUP-U60
  // x <- N
  // y <- N
  // ~
  else if get_tag(arg0) == U60 {
    inc_cost(ctx.heap, ctx.tid);
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), arg0);
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), arg0);
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
    return true;
  }

  // dup x y = N
  // ----------- DUP-F60
  // x <- N
  // y <- N
  // ~
  else if get_tag(arg0) == F60 {
    inc_cost(ctx.heap, ctx.tid);
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), arg0);
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), arg0);
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
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
  else if get_tag(arg0) == CTR {
    inc_cost(ctx.heap, ctx.tid);
    let fnum = get_ext(arg0);
    let fari = arity_of(&ctx.prog.aris, arg0);
    if fari == 0 {
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), Ctr(fnum, 0));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), Ctr(fnum, 0));
      link(ctx.heap, *ctx.host, Ctr(fnum, 0));
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
    } else {
      let ctr0 = get_loc(arg0, 0);
      let ctr1 = alloc(ctx.heap, ctx.tid, fari);
      for i in 0 .. fari - 1 {
        let leti = alloc(ctx.heap, ctx.tid, 3);
        link(ctx.heap, leti + 2, take_arg(ctx.heap, arg0, i));
        link(ctx.heap, ctr0 + i, Dp0(get_ext(ctx.term), leti));
        link(ctx.heap, ctr1 + i, Dp1(get_ext(ctx.term), leti));
      }
      let leti = alloc(ctx.heap, ctx.tid, 3);
      link(ctx.heap, leti + 2, take_arg(ctx.heap, arg0, fari - 1));
      link(ctx.heap, ctr0 + fari - 1, Dp0(get_ext(ctx.term), leti));
      link(ctx.heap, ctr1 + fari - 1, Dp1(get_ext(ctx.term), leti));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), Ctr(fnum, ctr0));
      atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), Ctr(fnum, ctr1));
      free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
    }
    return true;
  }

  // dup x y = *
  // ----------- DUP-ERA
  // x <- *
  // y <- *
  else if get_tag(arg0) == ERA {
    inc_cost(ctx.heap, ctx.tid);
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp0(tcol, get_loc(ctx.term, 0)), Era());
    atomic_subst(ctx.heap, &ctx.prog.aris, ctx.tid, Dp1(tcol, get_loc(ctx.term, 0)), Era());
    link(ctx.heap, *ctx.host, Era());
    free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 3);
    return true;
  }

  else {
    return false;
  }
}
