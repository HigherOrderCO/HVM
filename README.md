High-order Virtual Machine (HVM)
=================================

**High-order Virtual Machine (HVM)** is a pure functional runtime that is
**lazy**, **non-garbage-collected** and **massively parallel**. It is also
**beta-optimal**, meaning that, for higher-order computations, it can be
exponentially faster than alternatives, including Haskell's GHC.

That is possible due to a new model of computation, the **Interaction Net**, which
supersedes the **Turing Machine** and the **Lambda Calculus**. Previous implementations
of this model have been inefficient in practice, however, a recent breakthrough
has drastically improved its efficiency, resulting in the HVM. Despite being
relatively new, it already beats mature compilers in many cases, and is set to
scale towards uncharted levels of performance.

**Welcome to the massively parallel future of computers!**

*Note: this repository is, right now, being updated towards V1. Expect some
unstable bits, missing stuff, and many changes through this week!*

Usage
-----

### 1. Install it

First, install [Rust](https://www.rust-lang.org/). Then, type:

```sh
cargo install hvm
```

### 2. Create an HVM file

HVM files look like untyped Haskell. Save the file below as `main.hvm`:

```javascript
// Creates a tree with `2^n` elements
(Gen 0) = (Leaf 1)
(Gen n) = (Node (Gen(- n 1)) (Gen(- n 1)))

// Adds all elements of a tree
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))

// Performs 2^n additions in parallel
(Main n) = (Sum (Gen n))
```

### 3. Run it!

```sh
hvm run -f main.hvm "(Main 25)"
```

This will sum 2^25 numbers in parallel, using HVM's interpreter. To learn how to
compile it, and much more, see the [guide/README.md](guide/README.md). To
understand the technology behind it, check [guide/HOW.md](guide/HOW.md).

Benchmarks
==========

HVM has two main advantages over GHC: automatic parallelism and beta-optimality.
I've selected 5 common benchmarks to compare them. Below are the results:

List Fold (Sequential)
----------------------

<table>
<tr>
  <td><a href="./bench/ListFold/main.hvm"> main.hvm </a></td>
  <td><a href="./bench/ListFold/main.hs" > main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// Folds over a list
(Fold Nil         c n) = n
(Fold (Cons x xs) c n) = (c x (Fold xs c n))

// A list from 0 to n
(Range 0 xs) = xs
(Range n xs) =
  let m = (- n 1)
  (Range m (Cons m xs))

// Sums a big list with fold
(Main n) =
  let size = (* n 1000000)
  let list = (Range size Nil)
  (Fold list λaλb(+ a b) 0)
```

</td>
<td>

```haskell
-- Folds over a list
fold Nil         c n = n
fold (Cons x xs) c n = c x (fold xs c n)

-- A list from 0 to n
range 0 xs = xs
range n xs =
  let m = n - 1
  in range m (Cons m xs)

-- Sums a big list with fold
main = do
  n <- read.head <$> getArgs :: IO Word32
  let size = 1000000 * n
  let list = range size Nil
  print $ fold list (+) 0
```

</td>
</tr>
</table>

![](bench/_results_/ListFold.png)
<sub> *the lower the better </sub>

In this micro-benchmark, we just build a huge list of numbers, and fold over
it to sum them. Since lists are sequential, and since there are no higher-order
lambdas, HVM doesn't have any technical advantage over GHC. As such, both
runtimes perform very similarly.

Tree Sum (Parallel)
-------------------

<table>
<tr>
  <td><a href="./bench/TreeSum/main.hvm"> main.hvm </a></td>
  <td><a href="./bench/TreeSum/main.hs" > main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// Creates a tree with `2^n` elements
(Gen 0) = (Leaf 1)
(Gen n) = (Node (Gen(- n 1)) (Gen(- n 1)))

// Adds all elements of a tree
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))

// Performs 2^n additions
(Main n) = (Sum (Gen n))
```

</td>
<td>

```haskell
-- Creates a tree with 2^n elements
gen 0 = Leaf 1
gen n = Node (gen(n - 1)) (gen(n - 1))

-- Adds all elements of a tree
sun (Leaf x)   = 1
sun (Node a b) = sun a + sun b

-- Performs 2^n additions
main = do
  n <- read.head <$> getArgs :: IO Word32
  print $ sun (gen n)
```

</td>
</tr>
</table>

![](bench/_results_/TreeSum.png)

TreeSum recursively builds and sums all elements of a perfect binary tree. HVM
outperforms Haskell by a wide margin because this algorithm is embarrassingly
parallel, allowing it to fully use the available cores.

QuickSort (Parallel)
--------------------

<table>
<tr>
  <td><a href="./bench/QuickSort/main.hvm"> main.hvm </a></td>
  <td><a href="./bench/QuickSort/main.hs" > main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// QuickSort
(QSort p s Nil)          = Empty
(QSort p s (Cons x Nil)) = (Single x)
(QSort p s (Cons x xs))  =
  (Split p s (Cons x xs) Nil Nil)

// Splits list in two partitions
(Split p s Nil min max) =
  let s   = (>> s 1)
  let min = (QSort (- p s) s min)
  let max = (QSort (+ p s) s max)
  (Concat min max)
(Split p s (Cons x xs) min max) =
  (Place p s (< p x) x xs min max)

// Sorts and sums n random numbers
(Main n) =
  let list = (Randoms 1 (* 100000 n))
  (Sum (QSort Pivot Pivot list))
```

</td>
<td>

```haskell
-- QuickSort
qsort p s Nil          = Empty
qsort p s (Cons x Nil) = Single x
qsort p s (Cons x xs)  =
  split p s (Cons x xs) Nil Nil

-- Splits list in two partitions
split p s Nil min max =
  let s'   = shiftR s 1
      min' = qsort (p - s') s' min
      max' = qsort (p + s') s' max
  in  Concat min' max'
split p s (Cons x xs) min max =
  place p s (p < x) x xs min max

-- Sorts and sums n random numbers
main = do
  n <- read.head <$> getArgs :: IO Word32
  let list = randoms 1 (100000 * n)
  print $ sun $ qsort pivot pivot $ list
```

</td>
</tr>
</table>

![](bench/_results_/QuickSort.png)

This test modifies QuickSort to return a concatenation tree instead of a flat
list. This makes it embarrassingly parallel, allowing HVM to outperform GHC by a
wide margin again. It even beats Haskell's sort from Data.List! Note that
flattening the tree will make the algorithm sequential. That's why we didn't
choose MergeSort, as `merge` operates on lists. In general, trees should be
favoured over lists on HVM.

Composition (Optimal)
---------------------

<table>
<tr>
  <td><a href="./bench/Composition/main.hvm"> main.hvm </a></td>
  <td><a href="./bench/Composition/main.hs" > main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// Computes f^(2^n)
(Comp 0 f x) = (f x)
(Comp n f x) = (Comp (- n 1) λk(f (f k)) x)

// Performs 2^n compositions
(Main n) = (Comp n λx(x) 0)
```

</td>
<td>

```haskell
-- Computes f^(2^n)
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x

-- Performs 2^n compositions
main = do
  n <- read.head <$> getArgs :: IO Int
  print $ comp n (\x -> x) (0 :: Int)
```

</td>
</tr>
</table>

![](bench/_results_/Composition.png)

This chart isn't wrong: HVM is *exponentially* faster for function composition,
due to optimality, depending on the target function. There is no parallelism
involved here. In general, if the composition of a function `f` has a constant-
size normal form, then `f^(2^N)(x)` is linear-time (`O(N)`) on HVM, and
exponential-time (`O(2^N)`) on GHC. This can be taken advantage of to design
novel functional algorithms. I highly encourage you to try composing different
functions and watching how their complexity behaves. Can you tell if it will be
linear or exponential? Or how recursion will affect it? That's a very
insightful experience!

Lambda Arithmetic (Optimal)
---------------------------

<table>
<tr>
  <td><a href="./bench/LambdaArithmetic/main.hvm"> main.hvm </a></td>
  <td><a href="./bench/LambdaArithmetic/main.hs" > main.hs </a></td>
</tr>
<tr>
<td>

```javascript
// Increments a Bits by 1
(Inc xs) = λex λox λix
  let e = ex
  let o = ix
  let i = λp (ox (Inc p))
  (xs e o i)

// Adds two Bits
(Add xs ys) = (App xs λx(Inc x) ys)

// Multiplies two Bits
(Mul xs ys) =
  let e = End
  let o = λp (B0 (Mul p ys))
  let i = λp (Add ys (B0 (Mul p ys)))
  (xs e o i)

// Squares (n * 100k)
(Main n) =
  let a = (FromU32 32 (* 100000 n))
  let b = (FromU32 32 (* 100000 n))
  (ToU32 (Mul a b))
```

</td>
<td>

```haskell
-- Increments a Bits by 1
inc xs = Bits $ \ex -> \ox -> \ix ->
  let e = ex
      o = ix
      i = \p -> ox (inc p)
  in get xs e o i

-- Adds two Bits
add xs ys = app xs (\x -> inc x) ys

-- Multiplies two Bits
mul xs ys =
  let e = end
      o = \p -> b0 (mul p ys)
      i = \p -> add ys (b1 (mul p ys))
  in get xs e o i

-- Squares (n * 100k)
main = do
  n <- read.head <$> getArgs :: IO Word32
  let a = fromU32 32 (100000 * n)
  let b = fromU32 32 (100000 * n)
  print $ toU32 (mul a b)
```

</td>
</tr>
</table>

![](bench/_results_/LambdaArithmetic.png)

This example takes advantage of beta-optimality to implement multiplication
using lambda-encoded bitstrings. Once again, HVM halts instantly, while GHC
struggles to deal with all these lambdas. Lambda encodings have wide practical
applications. For example, Haskell's Lists are optimized by converting them to
lambdas (foldr/build), its Free Monads library has a faster version based on
lambdas, and so on. HVM's optimality open doors for an entire unexplored field
of lambda-encoded algorithms that were simply impossible before.

*Charts made on [plotly.com](https://chart-studio.plotly.com/).*

How is that possible?
=====================

Check [guide/HOW.md](guide/HOW.md).

Community
=========

To follow the project, please join our [Telegram Chat](https://t.me/kindelia),
the [Kindelia community on Discord](https://discord.gg/VV7ppaVWYn) or
[Matrix](https://matrix.to/#/#kindelia:kde.org)!
