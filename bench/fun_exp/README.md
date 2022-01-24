Function Exponentiation
=======================

This micro benchmark computes `FunExp n f = f ^ (2 ^ n)`. That is, given a
function `f`, and a number `n`, it computes `f . f . f . f . (...)`, for a total
of `n` compositions. For example, `FunExp 8 (λx -> x + 1)` is equivalent to `(λx
-> x + 256)`.

This is a classic example where HOVM beats Haskell **asymptotically*. For
example, `(xFunExp 30 (\x -> x)) 42` takes 5 seconds to compute Haskell, and is
instantaneous on HOVM.

This holds as long as the composed function doesn't grow in size when
self-composed. If it does grow in size, then the asymptotics will be the same.
Interestingly, HOVM still beats GHC even in these cases, probably because HOVM
lambdas are more lightweight than GHC lambdas.
