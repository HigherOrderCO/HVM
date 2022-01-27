High-Order Virtual Machine (HOVM)
=================================

Have you ever dreamed of a future where people wrote high-level recursive code
in language that felt as elegant as Haskell, and that code was compiled to an
executable as memory-efficiency of Rust and the massive parallelism of CUDA?
Wait no more, the future has arrived!

High-Order Virtual Machine (HOVM) is a pure functional compile target that is
**lazy**, **non-garbage-collected** and **massively parallel**. Not only that,
it is **beta-optimal**, which means it, in several cases, can be exponentially
faster than the every other functional runtime, including Haskell's GHC.

Usage
-----

Using HOVM couldn't be simpler. Just...

#### 1. Install it

First, install [Rust](https://www.rust-lang.org/). Then, type:

```bash
git clone git@github.com:Kindelia/HOVM
cd HOVM
cargo install --path .
```

#### 2. Create a HOVM file

HOVM files look like untyped Haskell, and are very easy to write. Below is an
algorithm that allocates a huge binary tree of ints, and then sums all its
elements in parallel:

```javascript
// Creates a tree with `2^n` copies of `x`
(Gen 0 x) = (Leaf x)
(Gen n x) = (Node (Gen (- n 1) x) (Gen (- n 1) x))

// Sums a tree in parallel
(Sum (Leaf x))   = x
(Sum (Node a b)) = (+ (Sum a) (Sum b))

// Performs 2^30 sums
(Main) = (Sum (Gen 30 1))
```

Notice there is no parallelism annotations.

#### 3. Try it with the interpreter

The `run` command allows you to test your algorithm using a fast interpreter:

```bash
hovm run main.hovm
```

#### 4. Compile it to fast, parallel C

Once you're ready, compile your file to blazingly fast, massively parallel C:

```bash
hovm c main.hovm                   # compiles hovm to C
clang -O2 main.c -o main -lpthread # compiles C to executable
./main                             # runs the executable
```

The program above runs in about **6.4 seconds** in a modern 8-core processor,
while the identical Haskell code takes about **19.2 seconds** in the same
machine with GHC.


Benchmarks
----------

Check the [benchmark](https://raw.githubusercontent.com/Kindelia/HOVM/master/WHY_HOVM_MATTERS.md) section!

How is that possible?
---------------------

Check [HOW.md](https://github.com/Kindelia/HOVM/blob/master/HOW.md) for more information.

How can I help?
---------------

Join us at the [Kindelia](https://discord.gg/QQ2jkxVj) community!
