#include <stdio.h>

int fib(int n, int z) {
  if (n == 0) {
    return z;
  }
  if (n == 1) {
    return z + 1;
  }
  return fib(n - 1, z) + fib(n - 2, z);
}

int main() {
  printf("%d ", fib(40, 0));
  printf("%d ", fib(40, 1));
  printf("%d ", fib(40, 2));
  printf("%d ", fib(40, 3));
  printf("%d ", fib(40, 4));
  printf("%d ", fib(40, 5));
  printf("%d ", fib(40, 6));
  printf("%d ", fib(40, 7));
}
