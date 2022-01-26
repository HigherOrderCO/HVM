High-Order Virtual Machine (HOVM)
=================================

High-Order Virtual Machine (HOVM) is a functional runtime. It is **lazy** (like
Haskell), **non-garbage-collected** (like Rust), **massively parallel** (like
CUDA) and has a **cost model** (like the EVM). It aims to become the ultimate
compilation target for the inevitable parallel, functional future of computing.

Usage
-----

#### 1. Install

Install [Rust](https://www.rust-lang.org/), then enter:

```bash
git clone git@github.com:Kindelia/HOVM
cd HOVM
cargo install --path .
```

#### 2. Create a HOVM file

Despite being a low-level machine, HOVM provides a user-facing language that is
very easy to use. It is basically untyped Haskell with integers. Save the file
below as `main.hovm`:

```javascript
// Applies a function to all elements in a list
(Map fn (Nil))            = (Nil)
(Map fn (Cons head tail)) = (Cons (fn head) (Map fn tail))

// Increments all numbers on [1,2,3]
(Main) = (Map λx(+ x 1) (Cons 1 (Cons 2 (Cons 3 (Nil)))))
```

#### 3. Run it

* Interpreted:

    ```bash
    hovm run main.hovm
    ```

* Compiled:

    ```bash
    hovm compile main.hovm                               # compiles hovm to C
    clang -O2 main.hovm.out.c -o main.hovm.out -lpthread # compiles C to executable
    ./main.hovm.out                                      # runs executable
    ```

*That's it!*
