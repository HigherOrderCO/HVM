Higher-order Virtual Machine 2 (HVM2)
=====================================

**Higher-order Virtual Machine 2 (HVM2)** is a massively parallel [Interaction
Combinator](https://www.semanticscholar.org/paper/Interaction-Combinators-Lafont/6cfe09aa6e5da6ce98077b7a048cb1badd78cc76)
evaluator.

By compiling programs from high-level languages (such as Python and Haskell) to
HVM, one can run these languages directly on massively parallel hardware, like
GPUs, with near-ideal speedup.

HVM2 is the successor to [HVM1](https://github.com/HigherOrderCO/HVM1), a 2022
prototype of this concept. Compared to its predecessor, HVM2 is simpler, faster
and, most importantly, more correct. [HOC](https://HigherOrderCO.com/) provides
long-term support to all features listed on its [PAPER](./PAPER.pdf).

This repository provides a low-level IR language for specifyig the HVM2 nets,
and a compiler from that language to C and CUDA. It is not meant for direct
human usage. If you're looking for a high-level language to interface with HVM2,
check [Bend](https://github.com/HigherOrderCO/Bend) instead.

Usage
-----

Install HVM2:

```sh
cargo +nightly install hvm
```

There are multiple ways to run an HVM program:

```sh
hvm run    <file.hvm> # interpret via Rust
hvm run-c  <file.hvm> # interpret via C
hvm run-cu <file.hvm> # interpret via CUDA
hvm gen-c  <file.hvm> # compile to standalone C
hvm gen-cu <file.hvm> # compile to standalone CUDA
```

All modes produce the same output. The compiled modes require you to compile the
generated file (with `gcc file.c -o file`, for example), but are faster to run.
The CUDA versions have much higher peak performance, but are less stable. As a
rule of thumb, `gen-c` should be used in production.

Language
--------

HVM is a low-level compile target for high-level languages. It provides a raw
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
don't worry, it isn't meant to. [Bend](https://github.com/HigherOrderCO/Bend) is
the human-readable language, and should be used both by end users, and languages
aiming to target the HVM. If you're are looking to learn more about the core
syntax and tech, though, please do check the [PAPER](./PAPER.pdf).
