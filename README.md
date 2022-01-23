High-Order Virtual Machine (HOVM)
=================================

High-Order Virtual Machine (HOVM) is a massively parallel, beta-optimal,
functional runtime. It is:

* **Lazy**: like Haskell, expressions are only reduced when needed.

* **GC-Free**: like Rust, no garbage-collection is needed.

* **Parallel**: redexes can be reduced in parallel, thread-safely.

HOVM evolved from our old interaction-net based optimal evaluators, except using a new,
compact memory representation that makes it much more efficient in practice.

Usage
-----

TODO

Benchmarks
----------

TODO

How it works?
-------------

TODO
