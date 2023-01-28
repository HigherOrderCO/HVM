High-order Virtual Machine (HVM)
=================================

**High-order Virtual Machine (HVM)** is a pure functional runtime that is **lazy**, **non-garbage-collected** and
**massively parallel**. It is also **beta-optimal**, meaning that, for higher-order computations, it can, in
some cases, be up to exponentially faster than alternatives, including Haskell's GHC.

That is possible due to a new model of computation, the **Interaction Net**, which supersedes the **Turing Machine** and
the **Lambda Calculus**. Previous implementations of this model have been inefficient in practice, however, a recent
breakthrough has drastically improved its efficiency, resulting in the HVM. Despite being relatively new, it already
beats mature compilers in many cases, and is set to scale towards uncharted levels of performance.

**Welcome to the massively parallel future of computers!**

Examples
========

Essentially, HVM is a minimalist functional language that is compiled to a novel runtime based on [Interaction
Nets](https://pdf.sciencedirectassets.com/272575/1-s2.0-S0890540100X00600/1-s2.0-S0890540197926432/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJGMEQCIDQQ1rA8jV9UQEKlaFzLfqINP%2B%2FrcTj4NCbI40n%2FrGbyAiBLG%2FpnaXmp6BGaG1Yplr3YYsHzxet6aQXc1qnkbb3W0irMBAhSEAUaDDA1OTAwMzU0Njg2NSIM1FOMCDHcoyvFFAU6KqkEgJPcH6B8%2BRYsdLtUERtVlwtcXZBW38xnvb%2FPRSkmxcNaK%2BmQTa7L3ZFuZt9rpNjrB3sJHg%2Bxc%2FqdAF%2FsthEb1NreHNze7LmbStuRufZCGEcxax%2FsyjnSb9bnrHuDEpnck1Dhk2YPqR8%2Bim%2BdQisDUp%2F4torZsCK1%2BPAEQkQAmGqinioexAr8dEE0BOlHgxBz5YRIkV9pjLoq%2FjWFqiUSO2bPdVi2AfpDbXI48ek6gQs%2F6VTIFRShfezfAr1HoDlQEoyyVYnVy6wI%2Fu1WVB%2FA0JJHK1B7rZFEYilPSAdUpVSOvjhNHN9elxIxlFX6hOZz3YJ4QDeLCPztfMClYYxAex6hoBBVzTkRzszs18hK1K%2FMUMwF4o%2FDy1i3WLeUmC36CL7WXDik%2BTZ7WjJNYGVRILH6cDsHrg17A0MVI5njvw7iM%2FrYKoOgBD2ESct4nO3mpRkKVq%2F9UyKScwVT5VrNpuLWLnrg29BDvE%2BDoFI6c71cisENjhIhGPNrBCQvZLNe1k%2BD54NyfqOe4a1DguuzxBnsNj6BBD2lM6TyDvCz9w36u194aN8oks9hLuTuKp7Rk05dTt6rj4pThkHA%2FQQymmx74MlQtTXTnD5v%2F%2BmGSUz6vHzqaV2Ft5xjWf9w9NJHfTkFkpxNEv8fTUUSMBEhL4nF8wj0wiNbSwp9NvPOj3YMIG2icNxdAZyNsJYJUowOCXi4JTwCkqb2WdNOi88pOSaAautZrBg7nzCKyuCbBjqqATOzXItndBn%2Be6oyH2l8sD%2B5v%2FjIqCz8%2Bx%2Bz%2FZA3dntddFac64iWFGPbJeRGw05BiPX5TKBnrR%2BmaqfO%2F7SxoYfTV4hl5Z2lmJcoiEd%2BWUmNK2wntMlGtFn%2FmFeeljKBeMxnfh8DN0qRz10NZAfxhvqxAEBu67G0ZXpECGxr8fAiBrdvnEac6rWfv8%2FT0VA%2Fu6xjIMIrrwU65xAuVuIG%2BXpsdC073VLm1%2BEW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221119T011901Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYRK5XVMW%2F20221119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=74892553e56ba432350974a6f4dbebfd97418e2187a5c4e183da61dd0e951609&hash=bc1de316d0b6ee58191106c1cdbc34d1eaeab536a9bbc02dfae09818a8cc2510&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0890540197926432&tid=spdf-00500b38-a41c-4d5b-98bb-4a2754da3953&sid=17532fa99b4522476f2b00d636dc838e7e36gxrqa&type=client&ua=515904515402570a0401&rr=76c51d7eea7b4d36).
This approach is not only memory-efficient (no GC needed), but also has two significant advantages: **automatic
parallelism** and **beta-optimality**. The idea is that you write a simple functional program, and HVM will turn it into
a massively parallel, beta-optimal executable. The examples below highlight these advantages in action.

Bubble Sort
-----------

<table>
<tr>
  <td>From: <a href="./examples/sort/bubble/main.hvm">HVM/examples/sort/bubble/main.hvm</a></td>
  <td>From: <a href="./examples/sort/bubble/main.hs" >HVM/examples/sort/bubble/main.hs</a></td>
</tr>
<tr>
<td>

```javascript
// sort : List -> List
(Sort Nil)         = Nil
(Sort (Cons x xs)) = (Insert x (Sort xs))

// Insert : U60 -> List -> List
(Insert v Nil)         = (Cons v Nil)
(Insert v (Cons x xs)) = (SwapGT (> v x) v x xs)

// SwapGT : U60 -> U60 -> U60 -> List -> List
(SwapGT 0 v x xs) = (Cons v (Cons x xs))
(SwapGT 1 v x xs) = (Cons x (Insert v xs))
```

</td>
<td>

```haskell
sort' :: List -> List
sort' Nil         = Nil
sort' (Cons x xs) = insert x (sort' xs)

insert :: Word64 -> List -> List
insert v Nil         = Cons v Nil
insert v (Cons x xs) = swapGT (if v > x then 1 else 0) v x xs

swapGT :: Word64 -> Word64 -> Word64 -> List -> List
swapGT 0 v x xs = Cons v (Cons x xs)
swapGT 1 v x xs = Cons x (insert v xs)
```

</td>
</tr>
</table>

![](bench/_results_/sort-bubble.png)

On this example, we run a simple, recursive [Bubble Sort](https://en.wikipedia.org/wiki/Bubble_sort) on both HVM and GHC
(Haskell's compiler). Notice the algorithms are identical. The chart shows how much time each runtime took to sort a
list of given size (the lower, the better). The purple line shows GHC (single-thread), the green lines show HVM (1, 2, 4
and 8 threads). As you can see, both perform similarly, with HVM having a small edge.  Sadly, here, its performance
doesn't improve with added cores. That's because Bubble Sort is an *inherently sequential* algorithm, so HVM can't
improve it.

Radix Sort
----------

<table>
<tr>
  <td>From: <a href="./examples/sort/radix/main.hvm">HVM/examples/sort/radix/main.hvm</a></td>
  <td>From: <a href="./examples/sort/radix/main.hs" >HVM/examples/sort/radix/main.hs</a></td>
</tr>
<tr>
<td>

```javascript
// Sort : Arr -> Arr
(Sort t) = (ToArr 0 (ToMap t))

// ToMap : Arr -> Map
(ToMap Null)       = Free
(ToMap (Leaf a))   = (Radix a)
(ToMap (Node a b)) =
  (Merge (ToMap a) (ToMap b))

// ToArr : Map -> Arr
(ToArr x Free) = Null
(ToArr x Used) = (Leaf x)
(ToArr x (Both a b)) =
  let a = (ToArr (+ (* x 2) 0) a)
  let b = (ToArr (+ (* x 2) 1) b)
  (Node a b)

// Merge : Map -> Map -> Map
(Merge Free       Free)       = Free
(Merge Free       Used)       = Used
(Merge Used       Free)       = Used
(Merge Used       Used)       = Used
(Merge Free       (Both c d)) = (Both c d)
(Merge (Both a b) Free)       = (Both a b)
(Merge (Both a b) (Both c d)) =
  (Both (Merge a c) (Merge b d))
```

</td>
<td>

```haskell
sort :: Arr -> Arr
sort t = toArr 0 (toMap t)

toMap :: Arr -> Map
toMap Null       = Free
toMap (Leaf a)   = radix a
toMap (Node a b) =
  merge (toMap a) (toMap b)

toArr :: Word64 -> Map -> Arr
toArr x Free       = Null
toArr x Used       = Leaf x
toArr x (Both a b) =
  let a' = toArr (x * 2 + 0) a
      b' = toArr (x * 2 + 1) b
  in Node a' b'

merge :: Map -> Map -> Map
merge Free       Free       = Free
merge Free       Used       = Used
merge Used       Free       = Used
merge Used       Used       = Used
merge Free       (Both c d) = (Both c d)
merge (Both a b) Free       = (Both a b)
merge (Both a b) (Both c d) =
  (Both (merge a c) (merge b d))
```

</td>
</tr>
</table>


![](bench/_results_/sort-radix.png)

On this example, we try a [Radix Sort](https://en.wikipedia.org/wiki/Radix_sort), based on merging immutable trees. In
this test, for now, single-thread performance was superior on GHC - and this is often the case, since GHC is much older
and has astronomically more micro-optimizations - yet, since this algorithm is *inherently parallel*, HVM was able to
outperform GHC given enough cores. With **8 threads**, HVM sorted a large list **2.5x faster** than GHC.

Keep in mind one could parallelize the Haskell version with `par` annotations, but that would demand time-consuming,
expensive refactoring - and, in some cases, it isn't even *possible* to use all the available parallelism with `par`
alone. HVM, on the other hands, will automatically distribute parallel workloads through all available cores, achieving
horizontal scalability. As HVM matures, the single-thread gap will decrease significantly.

Lambda Multiplication
---------------------

<table>
<tr>
  <td>From: <a href="./examples/lambda/multiplication/main.hvm">HVM/examples/lambda/multiplication/main.hvm </a></td>
  <td>From: <a href="./examples/lambda/multiplication/main.hs" >HVM/examples/lambda/multiplication/main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// Increments a Bits by 1
// Inc : Bits -> Bits
(Inc xs) = λex λox λix
  let e = ex
  let o = ix
  let i = λp (ox (Inc p))
  (xs e o i)

// Adds two Bits
// Add : Bits -> Bits -> Bits
(Add xs ys) = (App xs λx(Inc x) ys)

// Multiplies two Bits
// Mul : Bits -> Bits -> Bits
(Mul xs ys) =
  let e = End
  let o = λp (B0 (Mul p ys))
  let i = λp (Add ys (B0 (Mul p ys)))
  (xs e o i)
```

</td>
<td>

```haskell
-- Increments a Bits by 1
inc :: Bits -> Bits
inc xs = Bits $ \ex -> \ox -> \ix ->
  let e = ex
      o = ix
      i = \p -> ox (inc p)
  in get xs e o i

-- Adds two Bits
add :: Bits -> Bits -> Bits
add xs ys = app xs (\x -> inc x) ys

-- Muls two Bits
mul :: Bits -> Bits -> Bits
mul xs ys =
  let e = end
      o = \p -> b0 (mul p ys)
      i = \p -> add ys (b0 (mul p ys))
  in get xs e o i
```

</td>
</tr>
</table>

![](bench/_results_/lambda-multiplication.png)

This example implements bitwise multiplication using [λ-encodings](https://en.wikipedia.org/wiki/Church_encoding). Its
purpose is to show yet another important advantage of HVM: beta-optimality. This chart isn't wrong: HVM multiplies
λ-encoded numbers **exponentially faster** than GHC, since it can deal with very higher-order programs with optimal
asymptotics, while GHC can not. As esoteric as this technique may look, it can actually be very useful to design
efficient functional algorithms. One application, for example, is to implement [runtime
deforestation](https://github.com/Kindelia/HVM/issues/167#issuecomment-1314665474) for immutable datatypes. In general,
HVM is capable of applying any fusible function `2^n` times in linear time, which sounds impossible, but is indeed true.

*Charts made on [plotly.com](https://chart-studio.plotly.com/).*

Getting Started
===============

1. Install Rust nightly:

    ```
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    rustup toolchain install nightly
    ```

2. Install HVM:

    ```
    cargo +nightly install hvm
    ```

3. Run an HVM expression:

    ```
    hvm run "(@x(+ x 1) 41)"
    ```

That's it! For more advanced usage, check the [complete guide](guide/README.md).

More Information
================

- To learn more about the **underlying tech**, check [guide/HOW.md](guide/HOW.md).

- To ask questions and **join our community**, check our [Discord Server](https://discord.gg/kindelia).

- To **contact the author** directly, send an email to <taelin@kindelia.org>.

FAQ
===

### Is HVM faster than GHC in a single core today?

No. For now, HVM seems to be from 50% faster to 3x slower in single thread
performance, to even worse if the Haskell code exploits optimizations that
HVM doesn't have yet (ST Monad, mutable arrays, inlining, loops).

### Is HVM faster than Rust today?

No.

### Is HVM faster than C today?

No!

### Can HVM be faster than these one day? 

Hard question. Perhaps! The underlying model is very efficient. HVM shares the
same initial core as Rust (an affine λ-calculus), has great memory management
(no thunks, no garbage-collection). Some people think interaction nets are an
overhead, but that's not the case - they're the *lack* of overhead. For example,
a lambda on HVM uses only 2 64-bit pointers, which is about as lightweight as it
gets. Furthermore, every reduction rule of HVM is a lightweight, constant-time
operation that can be compiled to very fast machine code. As such, given enough
optimizations, from proper inlining, to real loops, to inner mutability
(FBIP-like?), I believe HVM could one day compare to GHC and even Rust or C. But
we're still far from that.

### Why do the benchmarks compare single-thread vs multi-core?

They do not! Notice all benchmarks include a line for single-threaded HVM
execution, which is usually 3x slower than GHC. We do include multi-core HVM
execution to let us visualize how its performance scales with added cores,
without any change of the code. We do not include multi-core GHC execution
because GHC doesn't support automatic parallelism, so it is not possible to make
use of threads without changing the code. Keep in mind, once again, the
benchmarks are NOT claiming that HVM is faster than GHC today.

### Does HVM support the full λ-Calculus, or System-F?

Not yet! HVM is an impementation of the bookkeeping-free version of the
reduction algorithm proposed on [TOIOFPL](https://www.researchgate.net/publication/235778993_The_optimal_implementation_of_functional_programming_languages)
book, up to page 40. As such, it doesn't support some λ-terms, such as:

```
(λx.(x x) λf.λx.(f (f x)))
```

It is, though, Turing complete, and covers a wide subset of the λ-calculus,
including terms such as the Y-combinator, church encodings (including algorithms
like addition, multiplication and exponentiation), as well as arbitrary
datatypes (both native and scott encoded) and recursion.

### Will HVM support the full λ-Calculus, or System-F?

Yes! We plan to, by implementing the full-algorithm described on the
[TOIOFPL](https://www.researchgate.net/publication/235778993_The_optimal_implementation_of_functional_programming_languages),
i.e., after page 40. Sadly, this results in an overhead that affects
the performance of beta-reduction by about 10x. As such, we want to
do so with caution to keep HVM efficient. Currently, the plan is:

1. Split lambdas into full-lambdas and light-lambdas

    - Light lambdas are what HVM has today. They're fast, but don't support the full λ-Calculus.

    - Full lambdas will be slower, but support the full λ-Calculus, via "internal brackets/croissants".

2. To decrease the overhead, convert full-lambdas to light-lambdas using EAL inference

    Elementary Affine Logic is a substructural logic that rejects the structural
    rule of contraction, replacing it by a controlled form of duplication. By
    extending HVM with EAL inference, we'll be able to convert most full-lambdas
    into lightweight lambdas, greatly reducing the associated slowdown.

Finally, keep in mind this only concerns lambdas. Low-order terms (constructors,
trees, recursion) aren't affected.

### Are unsupported terms "Undefined Behavior"?

No! Unsupported λ-terms like `λx.(x x) λf.λx.(f (f x))` don't cause HVM to
display undefined behavior. HVM will always behave deterministically, and give
you a correct result to any input, except it will be in terms of [Interaction
Calculus](https://github.com/Kindelia/Wikind/blob/master/IC/_.kind2) (IC)
semantics. The IC is an alternative to the Lambda Calculus (LC) which differs
slightly in how non-linear variables are treated. As such, these "unsupported"
terms are just cases where the LC and the IC evaluation disagree. In theory, you
could use the HVM as a Interaction Net runtime, and it would always give you
perfectly correct answers under relation to these semantics - but that's not
usual, so we don't talk about it often.

### What is HVM's main innovation, in simple terms?

In complex terms, HVM's main innovation is that it is an efficient
implementation of the Interaction Net, which is a concurrent model of
computation. But there is a way to translate it to more familiar terms. HVM's
performance, parallelism and GC-freedom all come from the fact it is based on a
linear core - just like Rust!  But, on top of it, instead of adding loops and
references (plus a "borrow checker"), HVM adds recursion and a *lazy,
incremental cloning primitive*. For example, the expression below:

```
let xs = (Cons 1 (Cons 2 (Cons 3 Nil))) in [xs, xs]
```

Computes to:

```
let xs = (Cons 2 (Cons 3 Nil)) in [(Cons 1 xs), (Cons 1 xs)]
```


Notice the first `Cons 1` layer was cloned incrementally. This makes cloning
essentially free, for the same reason Haskell's lazy evaluator allows you to
make infinite lists: there is no cost until you actually read the copy! That
lazy-cloning primitive is pervasive, and covers all primitives of HVM's runtime:
constructors, numbers and lambdas. This idea, though, breaks down for lambdas:
how do you incrementally copy a lambda?

```
let f = λx. (2 + x) in [f, f]
```

If you try it, you'll realize why that's not possible:

```
let f = (2 + x) in [λx. f, λx. f]
```

The solution to that question is the main insight that the Interaction Net model
bought to the table, and it is described in more details on the
[HOW.md](https://github.com/Kindelia/HVM/blob/master/guide/HOW.md) document.

### Is HVM always *asymptotically* faster than GHC?

No. In most common cases, it will have the same asymptotics. In some cases, it
is exponentially faster. In [this
issue](https://github.com/Kindelia/HVM/issues/60), an user noticed that HVM
displays quadratic asymptotics for certain functions that GHC computes in linear
time. That was a surprise to me, and, as far as I can tell, despite the
"optimal" brand, seems to be a limitation of the underlying theory. That said,
there are multiple ways to alleviate, or solve, this problem. One approach would
be to implement "safe pointers", also described on the book, which would reduce
the cloning overhead and make some quadratic cases linear. But that wouldn't
work for all cases. A complimentary approach would be to do linearity analysis,
converting problematic quadratic programs in faster, linear versions.  Finally,
in the worst case, we could add references just like Haskell, but that should be
made with a lot of caution, in order not to break the assumptions made by the
parallel execution engine.

### Is HVM's optimality only relevant for weird, academic λ-encoded terms?

No. HVM's optimality has some very practical benefits. For example, all the
"deforesting" techniques that Haskell employs as compile-time rewrite rules,
happen naturally, at runtime, on the HVM. For example, Haskell optimizes:

`map f .  map g`

Into:

`map (f . g)`

This is a hardcoded optimization. On HVM, that occurs naturally, at runtime,
in a very general and pervasive way. So, for example, if you have something
like:

```
foldr (.) id funcs :: [Int -> Int]
```

GHC won't be able to "fuse" the functions on the `funcs` list, since they're
not known at compile time. HVM will do that just fine.

Another practical application for λ-encodings is for monads. On Haskell, the
Free Monad library uses Church encodings as an important optimization. Without
it, the asymptotics of binding make free monads much less practical. HVM has
optimal asymptotics for Church encoded data, making it great for these problems.


### Why is HVM so parallelizable?

Because it is fully linear: every piece of data only occurs in one place at the
same time, which reduces need for synchronization. Furthermore, it is pure, so
there are no global side effects that demand communication. Because of that,
reducing HVM expressions in parallel is actually quite simple: we just keep a
work strealing queue of redexes, and let a pool of threads computing them. That
said, there are two places where HVM needs synchronization:

- On dup nodes, used by lazy cloning: a lock is needed to prevent threads from
  passing through, and, thus, accessing the same data

- On the substitution operation: that's because substitution could send data
  from one thread to another, so it must be done atomically

In theory, Haskell could be parallelized too, and GHC devs tried it at a point,
but I believe the non-linearity of the STG model would make the problem much
more complex than it is for the HVM, making it hard to not lose too much
performance due to synchronization overhead.

### How is the memory footprint of HVM, compared to other runtimes?

It is a common misconception that an "interactional" runtime would somehow
consume more memory than a "procedural" runtime like Rust's. That's not the
case. Interaction nets, as implemented on HVM, add no overhead, and HVM
instantly collects any piece of data that becomes unreachable, just like Rust,
so there are no accumulating thunks that result in world-stopping garbage
collection, as happens in Haskell currently.

That said, currently, HVM doesn't implement memory-efficient features like
references, loops and local mutability. As such, to do anything on HVM today,
you need to use immutable datatypes and recursion, which are naturally
memory-hungry. Thus, HVM programs today will have increased memory footprint, in
relation to C and Rust programs. Thankfully, there is no theoretical limitation
preventing us from adding loops and local mutability, and, once/if we do, one
can expect the same memory footprint as Rust. The only caveat, though, is shared
references: we're not sure if we want to add these, as they might impact
parallelism. As such, it is posible that we choose to let lazy clones to be the
only form of non-linearity, which would preserve parallelism, at the cost of
making some algorithms more memory-hungry.

### Is HVM meant to replace GHC?

No. GHC is actually a superb, glorious runtime that is very hard to match. HVM
is meant to be a lightweight, massively parallel runtime for functional, and
even imperative, languages, from Elm to JavaScript. That said, we do want to
support Haskell, but that will require HVM being in a much later stage of
maturity, as well as provide support for full lambdas, which it doesn't do yet.

### Is HVM production-ready?

No. HVM is still to be considered a prototype. Right now, I had less than
3 months to work on it directly. It is considerably less mature than other
compiler and runtimes like GHC and V8.

### I've ran an HVM program and it consumed 1950 GB and my computer exploded.

HVM is a prototype. Bugs are expected. Please, open an issue!

### I've used HVM in production and now my company is bankrupt

I quit.

Disclaimers
===========

(Removed in favor of the FAQ above!)

Related Work
============

- [Inpla](https://github.com/inpla/inpla) - a pure interaction net framework, without the "functional/calculus" style of HVM
