def sum(n):
  if n == 0:
    return 0
  else:
    return n + sum(n - 1)

def fun(n):
  if n == 0:
    return sum(16)
  else:
    return fun(n - 1) + fun(n - 1)

print(fun(8))

# Demo Micro-Benchmark / Stress-Test
# 
# Complexity: 120,264,589,303 Interactions
#
# CPython: 640s on Apple M3 Max (1 thread) *
# HVM-CPU: 268s on Apple M3 Max (1 thread)
# Node.js: 128s on Apple M3 Max (1 thread) *
# HVM-CPU:  14s on Apple M3 Max (12 threads)
# HVM-GPU:   2s on NVIDIA RTX 4090 (32k threads)
#
# * estimated due to stack overflow














