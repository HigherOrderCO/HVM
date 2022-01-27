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

Tree Sum
--------

<table>
<tr> <td>HOVM</td> <td>Haskell</td> </tr>
<tr>
<td>

```javascript
// Creates a tree with `2^n` copies of `x`
(Gen 0 x) = (Leaf x)
(Gen n x) = (Node (Gen (- n 1) x) (Gen (- n 1) x))

// Sums a tree in parallel
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))
```

</td>
<td>

```haskell
-- Generates a binary tree
gen :: Word32 -> Word32 -> Tree
gen 0 x = Leaf x
gen n x = Node (gen (n - 1) x) (gen (n - 1) x)

-- Sums its elements
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
(Quicksort Nil)                 = Empty
(Quicksort (Cons x xs))         = (Quicksort_ x xs)
(Quicksort_ p Nil)              = (Single p)
(Quicksort_ p (Cons x xs))      = (Split p (Cons p (Cons x xs)) Nil Nil)
  (Split p Nil         min max) = (Concat (Quicksort min) (Quicksort max))
  (Split p (Cons x xs) min max) = (Place p (< p x) x xs min max)
  (Place p 0 x xs      min max) = (Split p xs (Cons x min) max)
  (Place p 1 x xs      min max) = (Split p xs min (Cons x max))
```

</td>
<td>

```haskell
quicksort :: List Word32 -> Tree Word32
quicksort Nil                    = Empty
quicksort (Cons x Nil)           = Single x
quicksort l@(Cons p (Cons x xs)) = split p l Nil Nil where
  split p Nil         min max    = Concat (quicksort min) (quicksort max)
  split p (Cons x xs) min max    = place p (p < x) x xs min max
  place p False x xs  min max    = split p xs (Cons x min) max
  place p True  x xs  min max    = split p xs min (Cons x max)
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

Repeated Composition
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

This is a micro benchmark that composes a function `2^N` times and applies it to
an argument. There is no parallelism involved here. Instead, HOVM beats GHC
because of beta-optimality. In general, if the composition of a function `f` has
a constant-size normal form, then `f^N(x)` is constant-time (`O(L)`) on HOVM,
and exponential-time (`O(2^L)`) on GHC.

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
