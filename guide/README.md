HVM Guide
=========

Installation
------------

First, install [install Rust nightly](https://www.rust-lang.org/tools/install):

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install nightly
```

Then, install HVM:

```bash
cargo +nightly install --force --git https://github.com/HigherOrderCO/HVM.git
```

This will install HVM's command-line interface. Make sure it worked with:

```bash
hvm --version
```

You should see `hvm 1.0.VERSION`.

Basic Usage
-----------

In its simplest form, HVM is just a machine that receives a functional expression and
outputs its normal form. You can ask it to compute an expression with `hvm run`:

```bash
hvm run "(+ 2 3)"
```

This will add `2` and `3`, and output `5`. Expressions can include lambdas,
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
languages like Haskell. For example, the function below computes the
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
my BMI. Note that function names must start with an uppercase letter: that's how
HVM differentiates global functions from lambda-bound variables.

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
time hvm run -c true -f summation.hvm "(Sum 0 0 5000000)"
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
time hvm run -c true -f summation.hvm "(Sum 0 0 5000000)"
```

It will output:

```
12500002500000

[TIME: 0.28s | COST: 75000001 | RPS: 268.82m]
```

The RPS becomes 268 million rewrites per second! That's an almost perfect 7.38x
improvement, in a 8-core CPU. In general, one can improve a function's
performance proportionally to the number of cores by just writing its recursion
in a parallel-aware manner. No need for manual thread spawning, no kernels,
mutexes, locks, atomics nor any other overwhelmingly complex, error-prone
synchronization primitives. 

While the function above could be parallelized with some effort in other
languages; for example, using Haskell's `par`; this becomes considerably harder
as the recursion schemes become more complex. For example, the Fibonacci
function doesn't recurse in a regular way: some branches are much deeper than
others. As such, using all available parallelism with `par` alone would be very
hard. On HVM, you just write the function as it is, and HVM will smoothly
distribute the workload evenly across all available cores.

```javascript
(Fib 0) = 1
(Fib 1) = 1
(Fib n) = (+ (Fib (- n 1)) (Fib (- n 2)))
```

To learn more about parallel algorithm design on HVM, check [PARALLELISM](PARALLELISM.md).

Constructors
------------

If you do not write an equation for a function you use, it is considered a
constructor. That means you do not need to define datatypes with a `data` syntax
(as in Haskell). You can use any name starting with an uppercase, and it will
just work. For example, the program below extracts the first element of a pair:

```javascript
(First (Pair x y)) = x

Main = (First (Pair 1 2))
```

Notice that `Pair` is considered a constructor, because we didn't write an
equation to reduce it to some other expression. Another example would be
representing booleans:

```javascript
(And True  True)  = True
(And True  False) = False
(And False True)  = False
(And False False) = False

Main = (And True False)
```

HVM also has two pre-defined constructors, `String.cons` and `String.nil`, which
are meant to be used as UTF-32 strings. This just affects pretty printing. For
example:

```javascript
Main = (String.cons 104 (String.cons 105 String.nil))
```

If you run this, it will output the string `"hi"`, because `[104,105]` is the
UTF-32 encoding for it. HVM also has syntax sugars for Strings, so the program
above is equivalent to both programs below:

```javascript
Main = (String.cons 'h' (String.cons 'i' String.nil))
```

```javascript
Main = "hi"
```

HVM also has a syntax sugar for `List.cons` and `List.nil`, which are printed as
`[]` lists. For example:

```javascript
Main = (List.cons 1 (List.cons 2 (List.cons 3 List.nil)))
```

Running this will output `[1, 2, 3]`. As you can guess, you can also write `[1,
2, 3]` instead of `List.cons`. Both are equivalent.

Compiling a program
-------------------

The command we've used so far, `hvm run`, evaluates programs using an
interpreter. To run an application in production, you must compile it. To do so,
use the `compile` command, as follows:

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
compilation, we're now 49.97x faster than before.

Builtin Functions
-----------------

HVM has some useful pre-compiled functions.

### HVM.log (term: Term) (cont: Term)

Prints an arbitrary term to the terminal. It is very useful for debugging. Example:

```javascript
(Sum 0) = (HVM.log Done 0)
(Sum n) = (HVM.log (Call "Sum" n) (+ n (Sum (- n 1))))

Main = (Sum 4)
```

Will output:

```javascript
(Call "Sum" 4)
(Call "Sum" 3)
(Call "Sum" 2)
(Call "Sum" 1)
(Done)
10
```

Note that `10` is the result, and the other lines are the logged expressions.

### HVM.print (text: String) (cont: Term)

Prints a string to the terminal. The difference from `HVM.log` is that the text
is expected to be a string. Example:

```javascript
Main = (HVM.print "Hello" (+ 2 3))
```

This will output:

```
Hello
5
```

### HVM.query (cont: String -> Term)

Reads an user input from the terminal as a String. Example:

```javascript
(String.concat String.nil         ys) = ys
(String.concat (String.cons x xs) ys) = (String.cons x (String.concat xs ys))

Main =
  (HVM.print "What is your name?"
  (HVM.query λname
  (HVM.print (String.concat "Hello, " name)
  (Done))))
```

This will ask your name, then greet you.

### HVM.store (key: String) (val: String) (cont: Term)

Saves a text file on the working directory. Example:

```javascript
Main =
  (HVM.store "name.txt" "Alice"
  (Done))
```

This will save `name.txt` with the contents `Alice`.

### HVM.load (key: String) (cont: String -> Term)

Loads a text file from the working directory. Example:

```javascript
Main =
  (HVM.load "name.txt" λname
  (HVM.print name
  (Done)))
```

This will print the contents of `name.txt`. 

Extending HVM
-------------

HVM's built-in effects may not be sufficient for your needs, but it is possible
to extend HVM with new effects via its Rust API. For example, in the snippet
below, we extend HVM with a custom "MyPrint" IO:

```rust
// File to foad definitions from
let file = "file.hvm";

// Term to evaluate
let term = "(MyPrint \"cats are life\" (Done))";

// Extends HVM with our custom MyPrint IO function
let funs = vec![
  ("MyPrint".toString(), hvm::runtime::Function::Compiled {
    arity: 2,
    visit: |ctx| false,
    apply: |ctx| {

      // Loads argument locations
      let arg0 = runtime::get_loc(ctx.term, 0);
      let arg1 = runtime::get_loc(ctx.term, 1);

      // Converts the argument #0 to a Rust string
      if let Some(text) = crate::language::readback::as_string(ctx.heap, ctx.prog, &[ctx.tid], arg0) {
        // Prints it
        println!("{}", text);
      }

      // Sets the returned result to be the argument #1
      hvm::runtime::link(ctx.heap, *ctx.host, arg1);

      // Collects the argument #0
      hvm::runtime::collect(ctx.heap, &ctx.prog.arit, ctx.tid, hvm::runtime::load_ptr(ctx.heap, arg0));

      // Frees the memory used by this function call
      hvm::runtime::free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), 2);

      // Tells HVM the returned value must be reduced
      return true;
    },
  })
];

// Alloc 2 GB for the heap
let size = 2 * runtime::CELLS_PER_GB;

// Use 2 threads
let tids = 2;

// Don't show step-by-step
let dbug = false;

// Evaluate the expression above with "MyPrint" available
hvm::runtime::eval(file, term, funs, size, tids, dbug);
```

*To learn how to design the `apply` function, first learn HVM's memory model
(documented on
[runtime/base/memory.rs](https://github.com/Kindelia/HVM/blob/master/src/runtime/base/memory.rs)),
and then consult some of the precompiled IO functions
[here](https://github.com/Kindelia/HVM/blob/master/src/runtime/base/precomp.rs).
You can also use this API to extend HVM with new compute primitives, but to make
this efficient, you'll need to use the `visit` function too. You can see some
examples by compiling a `.hvm` file to Rust, and then checking the `precomp.rs`
file on the generated project.*

TODO: this section is a draft, must finish it.

To be continued...
------------------

This guide is a work-in-progress and will be expanded soon.
