Composition
===========

This micro-benchmark computes `f^(2^n)(x)`. I.e., it applies a function `2^n`
times to an argument. This is a classic example where HVM beats Haskell
*asymptotically*, depending on the function.

In general, if the composition of a function `f` has a constant-size normal
form, then `f^(2^n)(x)` is constant-time (`O(n)`) on HVM, and exponential-time
(`O(2^n)`) on GHC.

For example, the composition of `id = λx. x + 1` has a constant-size normal
form, since `id^N(x) = λx. x`, for any `N`. Because of that, `id^(2^30)(x)` is
instantaneous on HVM, yet it takes about 5 seconds on GHC. 

The composition of `u32_inc = λx. x + 1`, though, doesn't have a constant-size
normal form. For example, `u32_inc^4(x) = λx. x + 1 + 1 + 1 + 1`, and the size
grows the higher `N` is. Because of that, 
