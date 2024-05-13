Higher-order Virtual Machine 2 (HVM2)
=====================================

**Higher-order Virtual Machine 2 (HVM2)** is a massively parallel [Interaction
Combinator](https://www.semanticscholar.org/paper/Interaction-Combinators-Lafont/6cfe09aa6e5da6ce98077b7a048cb1badd78cc76)
evaluator. By compiling programs from high-level languages (such as Python and
Haskell) to HVM, one can run these languages directly on massively parallel
hardware, like GPUs, with near-ideal speedup.

HVM2 is the successor to [HVM1](https://github.com/HigherOrderCO/HVM1), a 2022
prototype of this concept. Compared to its predecessor, HVM2 is simpler, faster
and, most importantly, more correct. [HOC](https://HigherOrderCO.com/) provides
long-term support to all features listed on its [REPORT.md](./REPORT). 

This repository exposes a low-level IR syntax for specifyig the HVM2 nets, and a
compiler from that syntax to C and CUDA. If you're looking for a high-level
language to interface with HVM2, check the
[Bend](https://github.com/HigherOrderCO/Bend) language instead. 

Usage
-----

Install HVM2:

```
cargo +nightly install hvm
```

There are multiple ways to run an HVM program.

1. **Rust interpreter** (single-core, eager, slow):

```
hvm run file.hvm
```

2. **C interpreter** (parallel, eager, fast):

```
hvm run-c file.hvm
```

3. **C compiler** (parallel, eager, faster):

```
hvm gen-c file.hvm >> file.c
gcc file.c -o file
./file
```

4. **CUDA interpreter** (massively parallel, eager, fast):

```
hvm run-cu file.hvm
```

5. **CUDA compiler** (massively parallel, eager, fastest):

```
hvm gen-c file.hvm >> file.c
nvcc file.c -o file
./file
```

All versions are equivalent. As a rule of thumb, to test, use the Rust
interpreter for testing and the C compiler for production. The C/CUDA
interpreters are also optimized for speed, and can be used in contexts where 
the compilation time is undesirable. If a NVIDIA GPU is available, the CUDA
versions can be used too, but they're still experimental.

## Example

HVMC is a low-level compile target for high-level languages. It provides a raw
syntax for wiring interaction nets. For example:

```javascript
@main = a
  & @sum ~ (28 (0 a))

@sum = (?(((a a) @sum__C0) b) b)

@sum__C0 = ({c a} ({$([*2] $([+1] d)) $([*2] $([+0] b))} f))
  &! @sum ~ (a (b $(:[+] $(e f))))
  &! @sum ~ (c (d e))
```

The file above implements a recursive sum. If that looks unreadable to you -
don't worry, it isn't meant to. [Bend](https://github.com/HigherOrderCO/Bend)
is the human-readable presentation, and can be used both by end users, and
languages aiming to target the HVM. Check it out!
