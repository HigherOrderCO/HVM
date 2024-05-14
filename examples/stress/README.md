# stress

This is the basic stress-test used to test an implementation's maximum IPS. It
recursively creates a tree with a given depth, and then performs a recursive
computation with a given length:

```
def sum(n):
  if n == 0:
    return 0
  else:
    return n + sum(n - 1)

def fun(n):
  if n == 0:
    return sum(LENGTH)
  else:
    return fun(n - 1) + fun(n - 1)

fun(DEPTH)
```

This lets us test both the parallel and sequential performance of a runtime. For
example, by testing a tree of depth 14 and breadth 2^20, for example, we have
enough parallelism to use all the 32k threads of a RTX 4090, and enough
sequential work (1m calls) to keep each thread busy for a long time.
