HVM Guide
=========

Introduction
------------

HVM, or High-order Virtual Machine, is a massively parallel runtime that lets
programmers write high-performance applications via the functional paradigm.

Before the HVM, developing multi-threaded software was hard and costly, since
the complexity of thread-safe synchronization demanded significant expertise and
time. Even though CPUs and GPUs have been shipping with increasingly more cores
over the years, programming languages have failed to catch up with that trend,
wasting a huge potential. HVM bridges that gap, decreasing the cost of parallel
software development drastically. Not only that, it also brings important
computational properties, such as being garbage collection free, making it very
memory-efficient, and beta-optimal, which allows it to perform higher-order
computations with minimal complexity. 

This guide is intended for all audiences. If you want to use HVM to solve a
scientific program, develop an app or service, compile another programming
language, or just to learn about the tech, this document has you covered.

Installation
------------

First [install Rust](https://www.rust-lang.org/tools/install). Then, enter:

```bash
cargo install hvm
```

This will install HVM's command-line interface. Make sure it worked with:

```bash
hvm --version
```

Basic Usage
-----------

In its simplest form, HVM is just a machine that receives a functional expression and
outputs its normal form. You can ask it to compute an expression with `hvm run`:

```bash
hvm run "(+ 2 3)"
```

This will adds `2` and `3`, and output `5`. Expressions can include lambdas,
which are created with `@`. For example:

```bash
hvm run "(@x(* x 2) 21)"
```

Here, `@x(* x 2)` creates an anonymous function that receives `x` and doubles
it. That function is applied to `21`, so the final output is `42`. Since lambdas
are so powerful, HVM's expressions are Turing-complete, so, in theory, we could
solve any problem with expressions alone. But to make use of HVM's full
potential, we need to write programs.

First program
-------------

HVM allows you to extend its machine with user-supplied functions, which are
defined in files, using an equational style that resembles other functional
languages like Haskelll. For example, the function below computes the
[BMI](https://en.wikipedia.org/wiki/Body_mass_index) of a person:

```javascript
(BMI weight height) = (/ weight (* height height))
```

Save this file as `BMI.hvm` and enter:

```javascript
hvm run -f BMI.hvm "(BMI 62.0 1.70)"
```

The `-f` option tells HVM to load all the functions defined on `BMI.hvm` before
running the expression. The command above outputs `21.453287197231816`, which is
my BMI.


A sequential function
---------------------

Functions can have multiple equations, pattern-match on arguments, and recurse.
For example, the function below sums a range of numbers recursively:

```javascript
(Sum 1 a b) = 0
(Sum 0 a b) = (+ a (Sum (== a b) (+ a 1) b))
```

Internally, HVM breaks down its computations into parallel atomic operations,
called *graph rewrites*. Since each graph rewrite is a lightweight const-time
operation, the total cost of a computation can be measured precisely by the
number of graph rewrites. You can ask HVM to display it with the `-c` option.
Save the program above as `summation.hvm` and run:

```bash
time hvm run -c -f summation.hvm "(Sum 0 0 5000000)"
```

This will make HVM output:

```
12500002500000

[TIME: 0.96s | COST: 35000007 | RPS: 36.38m]
```

There are 4 relevant values here. First, `12500002500000` is the output, i.e.,
the summation from 0 to 5 million. Second, `0.96s` is the time this computation
took to complete. Third, `35000007` is the total number of atomic operations
that HVM applied to reach this result. Last, `36.38m` is the number of rewrites
per second, i.e., HVM's performance.

The advantage of using `COST` instead of `TIME` to measure the complexity of an
algorithm is that it is machine-agnostic, making it more reliable. With a cost
of about 35 million rewrites, this was a fairly heavy computation. Sadly, we
only achieved 36.38 million rewrites per second, which isn't stellar. Why?

The problem is HVM is greedy for parallelism, yet, the algorithm above is
**inherently sequential**. To understand why, let's see how `Sum` unfolds,
omitting the halting argument:

```
(Sum 0 100)
---------
(+ 0 (Sum 1 100))
----------------
(+ 0 (+ 1 (Sum 2 100)))
-----------------------
(+ 0 (+ 1 (+ 2 ... (Sum 98 100)))))
----------------------------------
(+ 0 (+ 1 (+ 2 ... (+ 98 (Sum 99 100)))))
-----------------------------------------
(+ 0 (+ 1 (+ 2 ... (+ 98 (+ 99 100)))))
--------------------------------
(+ 0 (+ 1 (+ 2 ... (+ 98 199))))
-------------------------
(+ 0 (+ 1 (+ 2 ... 297)))
----------------------
(+ 0 (+ 1 (+ 2 5047)))
----------------------
(+ 0 (+ 1 5049))
----------------
(+ 0 5050)
----------
5050
```

As you can see, HVM must recurse it all the way down to the base case, before it
is able to perform the first addition. Then, additions are performed upwards,
one after the other, in order. There is no room for parallelism in the function
we wrote, so, HVM can't help us here.

A parallel function
-------------------

We can improve the program above using a divide-and-conquer approach:

```javascript
// Sums all the numbers in the (a .. b) range.
(Sum 1 a b) = a
(Sum 0 a b) =
  let m = (/ (+ a b) 2)
  let n = (+ m 1)
  let l = (Sum (== a m) a m)
  let r = (Sum (== n b) n b)
  (+ l r)
```

The idea is that `Sum` now receives the range it must add. Then, on each
recursive iteration, it splits the range in two halves. When the range length is
1, it halts. Omitting the halting argument, below is how it unfolds:

```
(Sum 0 100)
-----------
(+ (Sum 0 50) (Sum 51 100))
---------------------------
(+ (+ (Sum 0 25) (Sum 26 50)) (+ (Sum 51 75) (Sum 76 100)))
-----------------------------------------------------------
(+ (+ (+ ... ...) (+ ... ...)) (+ (+ ... ...) (+ ... ...))))
------------------------------------------------------------
(+ (+ (+ 78 247) (+ 416 534)) (+ (+ 741 834) (+ 1066 1134)))
------------------------------------------------------------
(+ (+ 325 950) (+ 1575 2200))
-----------------------------
(+ 1275 3775)
-------------
5050
```

The way this function branches generates independent additions: it is
**inherently parallel**. That allows HVM's built-in parallelism to kick in,
significantly boosting the performance. If we run it:

```
time hvm run -c -f summation.hvm "(Sum 0 0 5000000)"
```

It will output:

```
12500002500000

[TIME: 0.28s | COST: 75000001 | RPS: 268.82m]
```

The RPS becomes 268 million rewrites per second! That's an almost perfect 7.38x
improvement, in a 8-core CPU. In this case, the parallel version did way more
operations overall, which decreased the TIME gains, but that's only because the
original `Sum` was too simple. In general, one can improve a function's
performance proportionally to the number of cores by just writing its recursion
in a parallel-aware manner.

Take a moment to appreciate how we didn't do anything related to parallelism.
No manual thread spawning, no kernels, mutexes, locks, atomics nor any other
overwhelmingly complex, error-prone synchronization primitives. We just wrote a
pure recursive function, and HVM smoothly distributed the workload across all
available cores. That's the paradigm shift that makes HVM so powerful: it makes
parallelism trivial, finally allowing software to make full use of the
performance that increasingly parallel CPUs have to offer.

Of course, this function in particular was fairly trivial, and it wouldn't be
hard to parallelize it using threads, or even Haskell's `par`. Some other
algorithms, though, aren't so simple. For example, the Fibonacci function
doesn't recurse in a regular way: some branches are much deeper than others. As
such, threads or `par` would fail to make full use of cores. HVM does so in all
cases, as long as your program isn't inherently sequential.

```javascript
(Fib 0) = 1
(Fib 1) = 1
(Fib n) = (+ (Fib (- n 1)) (Fib (- n 2)))
```

Compiling a program
-------------------

The command we've used so far, `hvm run`, will evaluate the programs using HVM's
interpreter. To run an application in production, you want to compile it. To do
so, use the `compile` command as follows:

```
hvm compile summation.hvm
```

This will generate a Rust repository with a fresh new copy of HVM, plus all the
functions defined on `summation.hvm` **precompiled** on the reduction engine.
You can then publish that project on `cargo` and use it from inside other Rust
projects (more on that later), or you can install `summation` as an executable
in your system and run it from the command line. It will work exactly like the
`hvm` command, except you'll be able to call `Sum` without loading a file:

```
cd summation
cargo install --path .
summation run -c true "(Sum 0 0 100000000)"
```

Moreover, it will be much faster. On my computer, the command below outputs:

```
5000000050000000

[TIME: 0.82s | COST: 1500000001 | RPS: 1818.18m]
```

That's another massive 6.7x increase in performance. With parallelism and
compilation, we're now 49.97x faster than before! While this looks fantastic,
this program in particular isn't a great showoff of HVM's capabilities, since
any reasonable compiler would convert it into a tight numeric loop anyway. Keep
in mind that HVM is a runtime, not a compiler, and, as such, it will run exactly
what you give it, without doing any kind of transformation. It is the source
language compiler's job to do these optimizations.

Let's now, finally, see a real-world example where HVM can greatly simplify the
amount of work needed to develop a parallel algorithm: sorting!

Parallel sorting
----------------

While HVM greatly decreases the complexity of parallel algorithm design, that
doesn't mean you can throw anything at it and it will just work. Caution must be
taken to avoid 1. inherent sequentialism; 2. unecessary synchronization points.
We already covered the first point. For example, consider the program below:

```javascript
// Insertion Sort
(Sort Nil)         = Nil
(Sort (Cons x xs)) = (Insert x (Sort xs))
  // Inserts an element on its sorted position
  (Insert v Nil)         = (Cons v Nil)
  (Insert v (Cons x xs)) = (Insert.go (> v x) v x xs)
    (Insert.go 0 v x xs) = (Cons v (Cons x xs))
    (Insert.go 1 v x xs) = (Cons x (Insert v xs))

// Generates a list in descending order
(Gen 0) = Nil
(Gen n) = (Cons n (Gen (- n 1)))

// Sums a list
(Sum Nil)         = 0
(Sum (Cons x xs)) = (+ x (Sum xs))

// Sorts and sums a descending list
Main = (Sum (Sort (Gen 20000)))
```

That's a pure functional insertion sort. Running it on compiled mode, I get:

```
200010000

[TIME: 7.50s | COST: 1000090004 | RPS: 133.36m]
```

That's 7.5s to sort a list of 20000 elements, with a total of 1 trillion
rewrites, and an RPS of 133 million rewrites per second. As you can guess, there
is much to improve. First, its asymptotic complexity is quadratic (and, indeed,
building a `size x cost` chart is a great way to see it). Second, due to the
sequential nature of the algorithm, we don't achieve any parallelization,
reaching a RPS of just `133m`.

A Haskell programmer might be tempted to write the following algorithm instead:

... TODO ...
