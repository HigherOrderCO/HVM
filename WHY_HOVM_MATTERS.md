What is HOVM, and why it matters?
=================================

Is it possible to have an expressive functional language, like Haskell, with the
memory efficiency of C, and the parallelism of CUDA? After years of research,
HOVM is my ultimate answer to that question. In this post, I'll teach you all
I've learned, and explain how HOVM will change computing as we know it.

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
