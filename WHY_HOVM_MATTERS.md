What is HOVM, and why it matters?
=================================

Is it possible to have a high-order functional language, like Haskell, with the
memory efficiency of Rust, and the parallelism of CUDA? After years of research,
the High-Order Virtual Machine (HOVM) is my ultimate answer to that question. In
this post, I'll make my case as to why I think either HOVM, or something very
similar to it, will **inevitably** take over the market, all the way to a point
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

HOVM outputs `(Cons 2 (Cons 4 (Cons 6 Nil)))`. **And that's it.** HOVM just
evaluates expressions. What makes it special, though, is how it does that.

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
possible before.

A language compiled to HOVM can be as expressive as Haskell, as memory-efficient
as Rust, can be cost-measured like the EVM, can compute algorithms that were
previously unfeasible, all while having the potential to run in thousands of
cores, like CUDA. If that looks like an extraordinary claim, it is. But if that
possibility makes you dreamy, keep reading, as I'll pack this post with an
extraordinary amount of evidences to convince you that's feasible!

What is the recent breakthrough?
--------------------------------

While the above facts have always been true, up to until a few months ago, the
real-world (not theoretical) efficiency of existing implementations was still
20-30x behind GHC, V8, ChezScheme and similar, which negated all the advantages
of beta-optimality, parallelism, GC-freedom, etc. That's why
[Kind-Lang](https://github.com/Kind-Lang) had to use JavaScript as its main
target, until optimal runtimes matured. A few months ago, though, I started a
brand new implementation, based 2 major optimizations:

1. A new memory format, based on [SIC](https://github.com/VictorTaelin/Symmetric-Interaction-Calculus).

2. User-defined rewrite rules, similar to Haskell equations.

These improvements allowed further optimizations, such as inlining numeric
duplications and using pthreads, and the result is mind-blowing: HOVM now peaks
at *2.5 billion rewrites per second* on my machine. The previous best
implementation barely reachesd *50 million*, on the same machine. That's a 50x
speedup!

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
factor**.  Note that it is never asymptotically slower, nor slower by an
unreasonably large constant factor. In the upcoming section, I'll explain why
each case happens.

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

Applies the identity function `2^N` times to an input.

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Pow 0 f x) = (f x)
(Pow n f x) = (Pow (- n 1) λk(f (f k)) x)
(Main)      = (Pow 32 λx(x) 0)
  ```

</td>
<td>

```haskell
pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print$ pow (32::Int) (\x->x) (0::Int)
```

</td>
</tr>
</table>

// TODO: CHART HERE

### compose_u32_inc

Applies the `λx -> x + 1` function `2^N` times to an input.

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Pow 0 f x) = (f x)
(Pow n f x) = (Pow (- n 1) λk(f (f k)) x)
(Main)      = (Pow 26 λx(+ x 1) 0)
  ```

</td>
<td>

```haskell
pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print $ pow (26 :: Int) (\x -> x + 1) (0 :: Int)
```

</td>
</tr>
</table>

// TODO: CHART HERE

Explaining the results
======================

TODO: I'll reword/reduce this section, leaving here for now

What is beta-optimality, and why it matters?
--------------------------------------------

Beta-optimality is the characteristic of a functional runtime that does the
minimal amount of beta-reductions required to 

runtime is extremely helpful when designing functional
algorithms, because certain programs run *exponentially* faster than they would
in any alternative.

### Is it even useful in practice?

Some people have the impression that beta-optimality only helps in "artificial"
situations, like λ-encoded datatypes, that don't show up in practice. That is
not true! There are countless common programming patterns on which HOVM
outperforms traditional runtimes. For example, function composition is one of
the most important operations in functional programming, and, in some cases,
it is exponentially faster in HOVM!

Similarly, there are vastly complex fusion optimizations performed by GHC to
prevent wasteful intermediate structures in common high-order functions like
`map` and `filter`, but these optimizations only work for a bunch of hard-coded
datatypes like Lists and Vectors. In HOVM, fusion is a natural consequence of
the runtime! As such, it can be implemented as a library, for any user-defined
datatype, without compile-time rewrites. Even better, it can take place at
runtime, which is impossible in Haskell.

Composition
===========

This micro-benchmark computes `f^(2^n)(x)`. I.e., it applies a function `2^n`
times to an argument. This is a classic example where HOVM beats Haskell
*asymptotically*, depending on the function.

In general, if the composition of a function `f` has a constant-size normal
form, then `f^(2^n)(x)` is constant-time (`O(n)`) on HOVM, and exponential-time
(`O(2^n)`) on GHC.

For example, the composition of `id = λx. x` has a constant-size normal
form, since `id^N(x) = λx. x`, for any `N`. Because of that, `id^(2^30)(x)` is
instantaneous on HOVM, yet it takes about 5 seconds on GHC. 

The composition of `u32_inc = λx. x + 1`, though, doesn't have a constant-size
normal form. For example, `u32_inc^4(x) = λx. x + 1 + 1 + 1 + 1`, and the size
grows the higher `N` is. Because of that, 





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
