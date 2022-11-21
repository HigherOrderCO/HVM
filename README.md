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

Examples
========

Essentially, HVM is a minimalist functional language that is compiled to a
novel runtime based on [Interaction
Nets](https://pdf.sciencedirectassets.com/272575/1-s2.0-S0890540100X00600/1-s2.0-S0890540197926432/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJGMEQCIDQQ1rA8jV9UQEKlaFzLfqINP%2B%2FrcTj4NCbI40n%2FrGbyAiBLG%2FpnaXmp6BGaG1Yplr3YYsHzxet6aQXc1qnkbb3W0irMBAhSEAUaDDA1OTAwMzU0Njg2NSIM1FOMCDHcoyvFFAU6KqkEgJPcH6B8%2BRYsdLtUERtVlwtcXZBW38xnvb%2FPRSkmxcNaK%2BmQTa7L3ZFuZt9rpNjrB3sJHg%2Bxc%2FqdAF%2FsthEb1NreHNze7LmbStuRufZCGEcxax%2FsyjnSb9bnrHuDEpnck1Dhk2YPqR8%2Bim%2BdQisDUp%2F4torZsCK1%2BPAEQkQAmGqinioexAr8dEE0BOlHgxBz5YRIkV9pjLoq%2FjWFqiUSO2bPdVi2AfpDbXI48ek6gQs%2F6VTIFRShfezfAr1HoDlQEoyyVYnVy6wI%2Fu1WVB%2FA0JJHK1B7rZFEYilPSAdUpVSOvjhNHN9elxIxlFX6hOZz3YJ4QDeLCPztfMClYYxAex6hoBBVzTkRzszs18hK1K%2FMUMwF4o%2FDy1i3WLeUmC36CL7WXDik%2BTZ7WjJNYGVRILH6cDsHrg17A0MVI5njvw7iM%2FrYKoOgBD2ESct4nO3mpRkKVq%2F9UyKScwVT5VrNpuLWLnrg29BDvE%2BDoFI6c71cisENjhIhGPNrBCQvZLNe1k%2BD54NyfqOe4a1DguuzxBnsNj6BBD2lM6TyDvCz9w36u194aN8oks9hLuTuKp7Rk05dTt6rj4pThkHA%2FQQymmx74MlQtTXTnD5v%2F%2BmGSUz6vHzqaV2Ft5xjWf9w9NJHfTkFkpxNEv8fTUUSMBEhL4nF8wj0wiNbSwp9NvPOj3YMIG2icNxdAZyNsJYJUowOCXi4JTwCkqb2WdNOi88pOSaAautZrBg7nzCKyuCbBjqqATOzXItndBn%2Be6oyH2l8sD%2B5v%2FjIqCz8%2Bx%2Bz%2FZA3dntddFac64iWFGPbJeRGw05BiPX5TKBnrR%2BmaqfO%2F7SxoYfTV4hl5Z2lmJcoiEd%2BWUmNK2wntMlGtFn%2FmFeeljKBeMxnfh8DN0qRz10NZAfxhvqxAEBu67G0ZXpECGxr8fAiBrdvnEac6rWfv8%2FT0VA%2Fu6xjIMIrrwU65xAuVuIG%2BXpsdC073VLm1%2BEW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221119T011901Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYRK5XVMW%2F20221119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=74892553e56ba432350974a6f4dbebfd97418e2187a5c4e183da61dd0e951609&hash=bc1de316d0b6ee58191106c1cdbc34d1eaeab536a9bbc02dfae09818a8cc2510&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0890540197926432&tid=spdf-00500b38-a41c-4d5b-98bb-4a2754da3953&sid=17532fa99b4522476f2b00d636dc838e7e36gxrqa&type=client&ua=515904515402570a0401&rr=76c51d7eea7b4d36).
This approach is not only memory-efficient (no GC needed), but also has two
significant advantages: **automatic parallelism** and **beta-optimality**. The
idea is that you write a simple functional program, and HVM will turn it into a
massively parallel, beta-optimal executable. The examples below highlight these
advantages in action.

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

On this example, we run a simple, recursive [Bubble
Sort](https://en.wikipedia.org/wiki/Bubble_sort) on both HVM and GHC (Haskell's
compiler). Notice the algorithms are identical. The chart shows how much time
each runtime took to sort a list of given size (the lower, the better). The
purple line shows GHC (single-thread), the green lines show HVM (1, 2, 4 and 8
threads). As you can see, both perform similarly, with HVM having a small edge.
Sadly, here, its performance doesn't improve with added cores. That's because
Bubble Sort is an *inherently sequential* algorithm, so HVM can't improve it.

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

On this example, we try a [Radix
Sort](https://en.wikipedia.org/wiki/Radix_sort), based on merging immutable
trees. This time, HVM's performance improves proportionally to the number of
cores. As such, in this test, it was able to sort large lists **9x faster** than GHC! That's because
GHC is locked to a single thread, while HVM exploits the fact that tree-merging
is *inherently parallel*. Of course, one could parallelize the Haskell version
with `par` annotations, but that would require refactoring. Usually, doing so is
very hard and time-consuming. In some cases, it is even *impossible* to use all
the available parallelism with `par` alone. HVM, on the other hands, will
automatically distribute the workload evenly among all available cores, with
no added programmer effort.


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

This example implements bitwise multiplication using
[λ-encodings](https://en.wikipedia.org/wiki/Church_encoding). Its purpose is to
show yet another important advantage of HVM: beta-optimality. This chart isn't
wrong: HVM multiplies λ-encoded numbers **exponentially faster** than GHC, since
it can deal with very higher-order programs with optimal asymptotics, while GHC
can not. As esoteric as this technique may look, it can actually be very useful
to design efficient functional algorithms. One application, for example, is to
implement [runtime
deforestation](https://github.com/Kindelia/HVM/issues/167#issuecomment-1314665474)
for immutable datatypes. In general, HVM is capable of applying any fusible
function `2^n` times in linear time, which sounds impossible, but is indeed true.

*Charts made on [plotly.com](https://chart-studio.plotly.com/).*

Getting Started
===============

1. Install Rust nightly:

    ```
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    rustup default nightly
    ```

2. Install HVM:

    ```
    cargo install hvm
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
