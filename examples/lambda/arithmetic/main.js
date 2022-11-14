let End = e => o => i => e
let B0  = p => e => o => i => o(p)
let B1  = p => e => o => i => i(p)

// Applies `f` `xs` times to `x`
function app(xs, f, x) {
  let e = f => x => x;
  let o = p => f => x => app(p, (k) => f(f(k)), x);
  let i = p => f => x => app(p, (k) => f(f(k)), f(x));
  return xs(e)(o)(i)(f)(x);
}

let inc = (xs) => (ex) => (ox) => (ix) => {
  let e = ex;
  let o = ix;
  let i = p => ox(inc(p));
  return xs(e)(o)(i);
}

function add(xs, ys) {
  return app(xs, (x) => inc(x), ys)
}

function mul(xs, ys) {
  let e = End;
  let o = p => B0(mul(p, ys));
  let i = p => add(ys, B0(mul(p, ys)));
  return xs(e)(o)(i);
}

function toU32(ys) {
  let e = 0;
  let o = p => 0 + (2 * toU32(p));
  let i = p => 1 + (2 * toU32(p));
  let a = ys(e)(o)(i);
  return a;
}

function fromU32(s, i) {
  if (s == 0) {
    return End;
  } else {
    function fromU32Put(s, mod, i) {
      if (mod == 0) {
        return B0(fromU32(s, i));
      } else {
        return B1(fromU32(s, i));
      }
    }
    return fromU32Put(s - 1, i % 2, (i / 2) >> 0);
  }
}

function main() {
  const N = process.argv[2];
  let a = fromU32(32,100000 * N);
  let b = fromU32(32,100000 * N);
  console.log(toU32(mul(a, b)));
}

main();
