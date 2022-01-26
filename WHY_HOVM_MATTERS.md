What is HOVM, and why it matters?
=================================

In essence, HOVM is just a machine that takes, as its input, a functional
program that looks like untyped Haskell, and outputs its evaluated result. It
can be used as a compile target for functional languages, as a virtual machine
for decentralized computers, and even as a programming language. As an example,
given the following input:

```javascript
// Doubles every number in the [1, 2, 3] list
(Fn (Nil))       = (Nil)
(Fn (Cons x xs)) = (Cons (* x 2) (Fn xs))
(Main)           = (Fn (Cons 1 (Cons 2 (Cons 3 Nil))))
```

HOVM outputs `(Cons 2 (Cons 4 (Cons 6 Nil)))`. That's it. What makes it special,
though, is **how** it does that.

What makes HOVM special?
========================

HOVM is based on a new, mathematically beautiful model of computation, the
Interaction Net, which is like the perfert child of the Lambda Calculus with the
Turing Machine. In a way, it is very similar to Haskell's STG, but with key
differences that give it outstanding characteristics, such as:

- Being **beta-optimal**: it is exponentially faster for many inputs.

- Being **inherently parallel**: it can be evaluated in thousands of cores.

- Being **memory-efficient**: no garbage collection, pure mutable datatypes.

- Being **strongly confluent**: it has a solid "gas cost" model.

In other words, in theory, a language compiled to this model could be as
expressive as Haskell, as memory-efficient as Rust, all while having the
potential to run in thousands of cores, like CUDA. Up to a few months ago,
though, the best implementations were still much slower than GHC in practice,
which negated its theoretical advantages. Now, thanks to a recent memory layout
breakthrough , we were able to completely redesign the runtime, and reach a peak
speed of **2.5 billion rewrites per second** on common CPUs. That's **50x** more
than the previous implementation, and enough to compete with GHC today.

Benchmarks
==========

Before we get technical, let's see some benchmarks against Haskell's GHC. Note
that HOVM's current release is a proof-of-concept implemented in about 1 month
by 4 part-time devs. It obviously won't always beat the most mature functional
runtime in the world. 

Haskell measured with:

```bash
ghc -O2 main.hs -o main
time ./main
```

HOVM measured with:

```bash
hovm main.hovm
clang -O2 main.c -o main
time ./main
```

Fibonacci
---------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Fib 0) = 0
(Fib 1) = 1
(Fib n) = (+ (Fib (- n 1)) (Fib (- n 2)))
```

</td>
<td>

```haskell
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
```

</td>
</tr>
</table>

// TODO: GRAPH HERE

#### Comment

HOVM is still not on par with GHC on predominantly numeric computations, but it
is not too far.

Function composition
--------------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Comp 0 f x) = (f x)
(Comp n f x) = (Comp (- n 1) λk(f (f k)) x)
```

</td>
<td>

```haskell
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x
```

</td>
</tr>
</table>

// TODO: GRAPH HERE

#### Comment

HOVM is exponentially faster than GHC here, due to optimality. In general, if
the composition of a function `f` has a constant-size normal form, then `f^N(x)`
is constant-time (`O(L)`) on HOVM, and exponential-time (`O(2^L)`) on GHC, where
`L` is the bit-size of `N`. This has a surprising consequence: HOVM can apply
certain functions `2^N` times using only `N` calls. This can be highly useful
for algorithm design (ref the Lambda Arithmetic benchmark).

QuickSort
---------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Quicksort (Nil))       = (Nil)
(Quicksort (Cons x xs)) =
  let min = (Filter λn(< n x) xs)
  let max = (Filter λn(> n x) xs)
  (Concat (Quicksort min) (Cons x (Quicksort max)))
```

</td>
<td>

```haskell
quicksort :: List Word32 -> List Word32
quicksort Nil         = Nil
quicksort (Cons x xs) =
  let min = xfilter (\n -> if n < x then 1 else 0) xs
      max = xfilter (\n -> if n > x then 1 else 0) xs
  in xconcat (quicksort min) (Cons x (quicksort max))
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

GHC is slightly faster on this test. That's because quicksort doesn't benefit
from optimality nor parallelism, so this is basically testing the performance of
the allocator and pattern-matcher. As HOVM matures and gains more micro
optimizations, this small difference should disappear.

TODO: use MergeSort instead?

Binary Tree Sum
----------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Sum Tip)       = 1
(Sum (Bin a b)) = (+ (Sum a) (Sum b))
```

</td>
<td>

```haskell
sun Tip       = 1
sun (Bin a b) = sun a + sun b
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

HOVM outperforms Haskell, but mostly because this algorithm is embarassingly
parallel, and the used machine has 8 cores.

Lambda Arithmetic
-----------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(Mul xs ys) = 
  let e = End
  let o = λp (B0 (Mul p ys))
  let i = λp (Add ys (B0 (Mul p ys)))
  (xs e o i)
```

</td>
<td>

```haskell
mul xs ys = 
  let e = end
      o = \p -> b0 (mul p ys)
      i = \p -> add ys (b1 (mul p ys))
  in get xs e o i
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

Once again, HOVM is exponentially faster than GHC. In this test, we used fast
function composition to design a multiplication algorithm using
[Scott-Encoded](https://kseo.github.io/posts/2016-12-13-scott-encoding.html)
bit-strings. Doing so efficiently is simply impossible in GHC.

Encoding data as lambdas has deep applications. For example, Haskell's Lists are
optimized by converting them to lambdas (foldr/build), its Free Monads library
has an entire, faster version based on lambdas, and so on. Sadly, GHC has
trouble evaluating programs too high-order. HOVM's optimality opens up an entire
field of unexplored lambda encoded algorithms.


[Free
Monad](https://hackage.haskell.org/package/free-5.1.7/docs/Control-Monad-Free.html)
library has an alternative version based on lambda necodings, which is much
faster. HOVM's optimal runtime opens the doors for an entire field of unexplored
lambda encoded algorithms.

// TODO: MORE BENCHMARKS
========================

TODO!

How is that possible?
=====================

How can it not need a garbage collector?
----------------------------------------

**[TODO]** explain how it is for that a high-level language not to be garbage
collected. trace parallels with Rust, explaining how both languages are similar
on their basis, yet differ when it comes to duplication, with Rust going the
reference/borrow route, while HOVM goes the novel "lazy cloning" route, which
allows structures to be lazily shared/copied, including lambdas

When is it asymptotically superior?
-----------------------------------

**[TODO]** explain how it is asymptotically more efficient than Haskell in many
cases, and why. begin by explaining why lazy evaluation is, intuitivelly,
superior, since it only computes what is necessary, but is actually inferior
since it can lead to computing the same expression twice. explain how Haskell
solves this problem via sharing (memoization of redexes). explain how it isn't
capable of sharing computations inside lambda binders. and that's when HOVM is
superior to Haskell, asymptotically

