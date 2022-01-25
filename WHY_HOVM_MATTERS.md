What is HOVM, and why it matters?
=================================

Is it possible to have a high-order functional language, like Haskell, with the
memory efficiency of Rust, and the parallelism of CUDA? After years of research,
the High-Order Virtual Machine (HOVM) is my ultimate answer to that question. In
this post, I'll make my case as to why I think either HOVM, or something very
similar to it, will inevitably take over the market, all the way to a point
when CPUs will be designed aroud it.

What is HOVM?
-------------

HOVM is the next-gen successor of the old optimal runtime used by
[Kind-Lang](https://github.com/kind-lang/kind) (Formality). Due to a
breakthrough that made it up to 50 times faster, it was separated as a
general-purpose compile target. In essence, HOVM is just a machine that takes,
as its input, a functional program that looks like untyped Haskell, and outputs
its evaluated result.

For example, given the following input:

```javascript
// Doubles every number in the [1, 2, 3] list
(Fn (Nil))       = (Nil)
(Fn (Cons x xs)) = (Cons (* x 2) (Fn xs))
(Main)           = (Fn (Cons 1 (Cons 2 (Cons 3 Nil))))
```

HOVM outputs `(Cons 2 (Cons 4 (Cons 6 Nil)))`. **And that's it.** What makes it
special, though, is how it does that.

What makes HOVM special?
------------------------

HOVM is very different from most conventional runtimes, because it is lazy, like
Haskell. It is, though, very different from Haskell itself, because it is based
on a new model of computation, interaction nets, that give it outstanding
characteristics, such as:

- Being **beta-optimal**: it is exponentially faster for many inputs.

- Being **inherently parallel**: it can be evaluated in thousands of cores.

- Being **garbage-collection free**: no "stop the world" GC is required.

- Being **memory-efficient**: mutable datatypes don't compromise purity.

- Being **strongly confluent**: it has a solid "gas cost" model.

In a way, HOVM is to Haskell as Haskell is to Scheme. It can be seen as a "hyper
lazy" runtime, that is capable of doing novel things that simply weren't
possible before. A language compiled to HOVM can be as expressive as Haskell, as
memory-efficient as Rust, can be cost-measured like the EVM, can compute
algorithms that were previously unfeasible, all while having the potential to
run in thousands of cores, like CUDA. If that looks like an extraordinary claim,
it is. But if that possibility makes you dreamy, keep reading, as I'll pack this
post with an extraordinary amount of evidences to convince you that's feasible!

What is the recent breakthrough?
--------------------------------

While the above facts have always been true, up to until a few months ago, the
real-world (not theoretical) efficiency of existing implementations was still
20-30x behind GHC, V8, ChezScheme and similar, which negated all the advantages
of beta-optimality, parallelism, GC-freedom, etc. That's why
[Kind-Lang](https://github.com/Kind-Lang) had to use JavaScript as its main
target, until optimal runtimes matured. A few months ago, though, I started a
brand new implementation, based 2 major optimizations:

1. A new memory format, based on [SIC](https://github.com/VictorTaelin/Symmetric-Interaction-Calculus), that reduces footprint by 50%.

2. User-defined rewrite rules, similar to Haskell equations.

These improvements allowed further optimizations, such as inlining numeric
duplications and using pthreads. The result is mind-blowing: HOVM now peaks
at **2.5 billion rewrites per second** on my machine. The previous best
implementation barely reached **50 million**, on the same machine. That's a **50x
speedup**!

Note that I am NOT claiming HOVM is faster than GHC today. It is a prototype
that has been in developement for about a month. It is obvious that, for many
inputs, it won't beat a mature compiler. The main point, though, is that - and I
want to make it very clear - **HOVM's current design is ready to scale and
become the fastest runtime in the world. And I to convince YOU to join me on
this endeavour!**

Benchmarks
----------

I'll present benchmarks against Haskell's GHC, since it is the reference when it
comes to lazy functional evaluation. Notice that, in some of these benchmarks,
HOVM is **exponentially faster**. In others, it is **faster when
multi-threaded**. In others, it is **either faster or slower by a small constant
factor**. Note that it is never asymptotically slower, nor slower by an
unreasonably large constant factor. For each case, I'll explain why it happens.

Haskell was measured with:

```bash
ghc -O2 main.hs -o main
time ./main
```

HOVM was measured with:

```bash
hovm main.hovm
clang -O2 main.c -o main
time ./main
```

### compose_id

**Applies the identity function `2^N` times to 0.**

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Comp 0 f x) = (f x)
(Comp n f x) = (Comp (- n 1) λk(f (f k)) x)
(Main)       = (Comp N λx(x) 0)
```

</td>
<td>

```haskell
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x
main       = print$ comp n (\x->x) (0::Int)
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

Function composition is one of cornerstones of functional programming, and,
amazingly, it is one of the cases where GHC performs poorly. In general, if the
composition of a function `f` has a constant-size normal form, then `f^N(x)` is
constant-time (`O(L)`) on HOVM, and exponential-time (`O(2^L)`) on GHC, where
`L` is the bit-size of `N`. Since the normal form of `id . id` is just `id`,
this program is exponentially faster in HOVM.

### compose_inc_u32

**Applies the `λx -> x + 1` function `2^N` times to 0.**

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Pow 0 f x) = (f x)
(Pow n f x) = (Pow (- n 1) λk(f (f k)) x)
(Main)      = (Pow N λx(+ x 1) 0)
  ```

</td>
<td>

```haskell
pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print $ pow n (\x -> x + 1) (0 :: Int)
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

The composition of `u32_inc = λx. x + 1` does NOT have a constant-size normal
form. For example, `u32_inc^4(x) = λx. x + 1 + 1 + 1 + 1`, and the size grows as
`N` grows. Because of that, both HOVM and GHC have the same asymptotics here.

For some reason, though, HOVM is about 2 faster, even single-thread. To be
honest, this surprised me. I've annotated the Haskell benchmark with `Int` to
make sure `Integer` wasn't involved, but the result still holds. In general, I
expect HOVM (of today) to be slightly slower than GHC when there is no
parallelism nor better asymptotics, but in this case it is just faster.

### compose_inc_ctr

**Applies the `Inc` function `2^N` times to BitString datatype.**

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(N)           = 26
(Comp 0 f x)  = (f x)
(Comp n f x)  = (Comp (- n 1) λk(f (f k)) x)
(Inc (E))     = (E)
(Inc (O bs))  = (I bs)
(Inc (I bs))  = (O (Inc bs))
(Zero 0)      = (E)
(Zero n)      = (O (Zero (- n 1)))
(ToInt (E))   = 0
(ToInt (O n)) = (* (ToInt n) 2)
(ToInt (I n)) = (+ (* (ToInt n) 2) 1)
(Main)        = (ToInt (Comp N λx(Inc x) (Zero 32)))
```

</td>
<td>

```haskell
data Bits   = E | O Bits | I Bits deriving Show
comp 0 f x  = f x
comp n f x  = comp (n - 1) (\k -> f (f k)) x
inc E       = E
inc (O bs)  = I bs
inc (I bs)  = O (inc bs)
zero 0      = E
zero n      = O (zero (n - 1))
toInt (E)   = 0
toInt (O n) = toInt n * 2
toInt (I n) = toInt n * 2 + 1
main        = print $ toInt (comp n inc (zero 32))
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

This benchmark is similar to the previous, except that, instead of incrementing
a machine integer `2^N` times, we increment a BitString represented as an
algebraic datatype. The purpose of this benchmark is to stress-test how fast the
runtime can perform pattern-matching and recursion. There is NO asymptotical
gain on the HOVM side, it does the exact same work as GHC. Despite that, it is
still considerably faster; again, to my surprise.

### compose_inc_lam

**Applies the `Inc` function `2^N` times to λ-Encoded BitString datatype.**

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(N)          = 24
(Comp 0 f x) = (f x)
(Comp n f x) = (Comp (- n 1) λk(f (f k)) x)
(E)          = λe λo λi e
(O pred)     = λe λo λi (o pred)
(I pred)     = λe λo λi (i pred)
(Inc bs)     = λe λo λi (bs e i λpred(o (Inc pred)))
(Zero 0)     = (E)
(Zero n)     = (O (Zero (- n 1)))
(ToInt bs)   = (bs 0 λn(* (ToInt n) 2) λn(+ (* (ToInt n) 2) 1))
(Main)       = (ToInt (Comp N λx(Inc x) (Zero 64)))
```

</td>
<td>

```haskell
newtype BS = BS { get :: forall a. a -> (BS -> a) -> (BS -> a) -> a }
comp 0 f x = f x
comp n f x = comp (n - 1) (\k -> f (f k)) x
e          = BS (\e -> \o -> \i -> e)
o pred     = BS (\e -> \o -> \i -> o pred)
i pred     = BS (\e -> \o -> \i -> i pred)
inc bs     = BS (\e -> \o -> \i -> get bs e i (\pred -> o (inc pred)))
zero 0     = e
zero n     = o (zero (n - 1))
toInt bs   = get bs 0 (\n -> toInt n * 2) (\n -> toInt n * 2 + 1)
main       = print $ toInt (comp n (\x -> inc x) (zero 64))
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

This is again similar to the previous benchmark, except that, this time, instead
of using built-in datatypes to represent the BitString, we're using the [Scott
Encoding](https://kseo.github.io/posts/2016-12-13-scott-encoding.html), which is
a way to represent data using lambdas. Doing so causes HOVM to be exponentially
faster than GHC.

This result looks wrong; after all, we shouldn't be able to call `inc` a
quadrillion times instantaneously. But that's exactly what happens. What is
going on is that, since the composition of *inc* has a small normal form, this
causes it to be slowly morphed into an "add with carry function" as the program
executes. This effect is what I call "runtime fusion".

Haskell programmers aren't unfamiliar with λ-encodings being used for
optimization. For example, the widely used [Free Monads
library](https://hackage.haskell.org/package/free-5.1.7/docs/Control-Monad-Free-Church.html)
has an entire version that replaces native datatypes by λ-encodings, and it is
much faster that way. But, since GHC isn't optimal, the application of this
technique is very limited in Haskell. HOVM opens doors for an entire field of
unexplored algorithms based on runtime fusion. I'd not be surprised if
there are solutions to hard problems lying there.


TODOS
=====

Before we get started, let's dive into some benchmarks.

**[TODO]** benchmarks where HOVM greatly outperforms GHC due to beta-optimality:
function exponentiation, map fusion, lambda encoding arithmetic, etc. 'HOVM is
to GHC as GHC is to C"

**[TODO]** benchmarks where HOVM greatly outperforms GHC due to parallelism:
sorting, rendering, parsing, mendelbrot set, etc.

**[TODO]** benchmarks where HOVM still underperforms GHC: numeric loops,
in-place mutation, C-like code, etc.

**[TODO]** Remark that HOVM is still on its infancy, and there are still many
optimizations and features to add (like mutable arrays) before it is generally
good; but stress that, on the long term, the underlying computation model paves
the way for it to be faster than everything that exists, specially when we
consider that 1. functional / high-order algorithms are every day more common,
2. the future is parallel and the core count will explode as soon as we start
using them properly

**[TODO]** explain how HOVM got there, starting from optimal evaluators (absal,
optlam, etc., which weren't as efficient as GHC in practice), models of
computation (Turing Machines, Lambda Calculus, Interaction Nets), and how the
new memory format plus user-defineds rewrite rules allowed HOVM to become an
order of magnitude faster and finally get close to GHC in real-world cases

**[TODO]** explain how it is for that a high-level language not to be garbage
collected. trace parallels with Rust, explaining how both languages are similar
on their basis, yet differ when it comes to duplication, with Rust going the
reference/borrow route, while HOVM goes the novel "lazy cloning" route, which
allows structures to be lazily shared/copied, including lambdas

**[TODO]** explain how it is asymptotically more efficient than Haskell in many
cases, and why. begin by explaining why lazy evaluation is, intuitivelly,
superior, since it only computes what is necessary, but is actually inferior
since it can lead to computing the same expression twice. explain how Haskell
solves this problem via sharing (memoization of redexes). explain how it isn't
capable of sharing computations inside lambda binders. and that's when HOVM is
superior to Haskell, asymptotically
