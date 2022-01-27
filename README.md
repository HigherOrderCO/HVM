High-Order Virtual Machine (HOVM)
=================================

**High-Order Virtual Machine (HOVM)** is a pure functional compile target that is
**lazy**, **non-garbage-collected** and **massively parallel**. Not only that,
it is **beta-optimal**, which means it, in several cases, can be exponentially
faster than the every other functional runtime, including Haskell's GHC.

That is possible due to a new model of computation, the Interaction Net, a
natural combination of the Turing Machine with the Lambda Calculus. Up until
recently, that model, despite elegant, was not efficient in practice. Thanks to
a new breaktrough, HOVM can now beat mature compilers, despite being just a
prototype.

Have you ever dreamed of a future where developers wrote high-level code in
language that felt **as elegant as Haskell**, and that code was compiled to
executables **as memory-efficiency of Rust**, all while enjoying the **massive
parallelism of CUDA**? Wait no more, the future has arrived!

Usage
-----

#### 1. Install it

First, install [Rust](https://www.rust-lang.org/). Then, type:

```bash
git clone git@github.com:Kindelia/HOVM
cd HOVM
cargo install --path .
```

#### 2. Create a HOVM file

HOVM files look like untyped Haskell. Save the file below as `main.hovm`:

```javascript
// Creates a tree with `2^n` elements
(Gen 0) = (Leaf 1)
(Gen n) = (Node (Gen(- n 1)) (Gen(- n 1)))

// Adds all elemements of a tree
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))

// Performs 2^30 additions in parallel
(Main n) = (Sum (Gen 30))
```

#### 3. Test it with the interpreter

```bash
hovm run main.hovm
```

#### 4. Compile it to blazingly fast, parallel C

```bash
hovm c main.hovm                   # compiles hovm to C
clang -O2 main.c -o main -lpthread # compiles C to executable
./main                             # runs the executable
```

The program above runs in about **6.4 seconds** in a modern 8-core processor,
while the identical Haskell code takes about **19.2 seconds** in the same
machine with GHC. Notice how there are no parallelism annotations! And that's
just the tip of iceberg. 


Benchmarks
==========

HOVM is compared against Haskell GHC, because it is the reference lazy
functional compiler. Note HOVM is still an early prototype. It obviously won't
beat GHC in many cases. HOVM has a lot of room for improvements and is expected
to improve steadily as optimizations are implemented.

```bash
# GHC
ghc -O2 main.hs -o main
time ./main

# HOVM
hovm main.hovm
clang -O2 main.c -o main
time ./main
```

Parallel Tree Sum
-----------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
// Creates a tree with `2^n` elements
(Gen 0) = (Leaf 1)
(Gen n) = (Node (Gen(- n 1)) (Gen(- n 1)))

// Adds all elemements of a tree
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))
```

</td>
<td>

```haskell
-- Creates a tree with 2^n elements
gen :: Word32 -> Tree
gen 0 = Leaf 1
gen n = Node (gen(n - 1)) (gen(n - 1))

-- Adds all elements of a tree
sun :: Tree -> Word32
sun (Leaf x)   = 1
sun (Node a b) = sun a + sun b
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment

The example from the README, TreeSum recursively builds and sums all elements of
a perfect binary tree. HOVM outperforms Haskell because this algorithm is
embarassingly parallel, allowing it to use all the 8 cores available.

Parallel QuickSort
------------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
(QSort Nil)         = Empty
(QSort (Cons x xs)) = (QSortAux x xs)
(QSortAux p Nil)    = (Single p)
(QSortAux p xs)     = (Split p (Cons p xs) Nil Nil)
  (Split p Nil         min max) = (Concat (QSort min) (QSort max))
  (Split p (Cons x xs) min max) = (Place p (< p x) x xs min max)
  (Place p 0     x xs  min max) = (Split p xs (Cons x min) max)
  (Place p 1     x xs  min max) = (Split p xs min (Cons x max))
```

</td>
</tr>
<tr>
<td>

```haskell
qsort :: List Word32 -> Tree Word32
qsort Nil          = Empty
qsort (Cons x Nil) = Single x
qsort xs           = split p ls Nil Nil where
  split p Nil         min max = Concat (qsort min) (qsort max)
  split p (Cons x xs) min max = place p (p < x) x xs min max
  place p False x xs  min max = split p xs (Cons x min) max
  place p True  x xs  min max = split p xs min (Cons x max)
```

</td>
</tr>
</table>

// TODO: CHART HERE

#### Comment


This test once again takes advantage of automatic parallelism by modifying the
usual QuickSort implementation to return a concatenation tree instead of a flat
list. This, again, allows HOVM to use multiple cores, making it outperform GHC
by a wide margin.

Optimal Composition
-------------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
// Computes f^(2^n)
(Comp 0 f x) = (f x)
(Comp n f x) = (Comp (- n 1) λk(f (f k)) x)
```

</td>
<td>

```haskell
-- Computes f^(2^n)
comp :: Int -> (a -> a) -> a -> a
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x
```

</td>
</tr>
</table>

// TODO: GRAPH HERE

#### Comment

This is a micro benchmark that composes a function `2^N` times and applies it to
an argument. There is no parallelism involved here. Instead, HOVM beats GHC
because of beta-optimality. In general, if the composition of a function `f` has
a constant-size normal form, then `f^N(x)` is constant-time (`O(L)`) on HOVM,
and exponential-time (`O(2^L)`) on GHC.

Optimal Lambda Arithmetic
-------------------------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
// The Scott-Encoded Bits type
(End)  = λe λo λi e
(B0 p) = λe λo λi (o p)
(B1 p) = λe λo λi (i p)

// Applies the `f` `xs` times to `x`
(App xs f x) =
  let e = λf λx x
  let o = λpλfλx(App p λk(f (f k)) x)
  let i = λpλfλx(App p λk(f (f k)) (f x))
  (xs e o i f x)

// Increments a Bits by 1
(Inc xs) = λeλoλi(xs e i λp(o (Inc p)))

// Adds two Bits
(Add xs ys) = (App xs λx(Inc x) ys)

// Multiplies two Bits
(Mul xs ys) = 
  let e = End
  let o = λp (B0 (Mul p ys))
  let i = λp (Add ys (B0 (Mul p ys)))
  (xs e o i)
```

</td>
<td>

```haskell
-- The Scott-Encoded Bits type
newtype Bits = Bits { get :: forall a. a -> (Bits -> a) -> (Bits -> a) -> a }
end  = Bits (\e -> \o -> \i -> e)
b0 p = Bits (\e -> \o -> \i -> o p)
b1 p = Bits (\e -> \o -> \i -> i p)

-- Applies the `f` `xs` times to `x`
app :: Bits -> (a -> a) -> a -> a
app xs f x =
  let e = \f -> \x -> x
      o = \p -> \f -> \x -> app p (\k -> f (f k)) x
      i = \p -> \f -> \x -> app p (\k -> f (f k)) (f x)
  in get xs e o i f x

-- Increments a Bits by 1
inc :: Bits -> Bits
inc xs = Bits (\e -> \o -> \i -> get xs e i (\p -> o (inc p)))

-- Adds two Bits
add :: Bits -> Bits -> Bits
add xs ys = app xs (\x -> inc x) ys

-- Muls two Bits
mul :: Bits -> Bits -> Bits
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

This example takes advantage of beta-optimality to implement multiplication
using lambda-encoded bit-strings. As expected, HOVM is exponentially faster than
GHC, since this program is very high-order.

Lambda encodings have wide practical applications. For example, Haskell's Lists
are optimized by converting them to lambdas (foldr/build), its Free Monads
library has a faster version based on lambdas, and so on. HOVM's optimality open
doors for an entire unexplored field of lambda encoded algorithms that are
simply impossible on any other runtime.

How is that possible?
=====================

Check [HOW.md](https://github.com/Kindelia/HOVM/blob/master/HOW.md).

How can I help?
===============

Join us at the [Kindelia](https://discord.gg/QQ2jkxVj) community!
