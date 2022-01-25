const int N = 30;
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
  printf("%d ", fib(N, 0));
  printf("%d ", fib(N, 1));
  printf("%d ", fib(N, 2));
  printf("%d ", fib(N, 3));
  printf("%d ", fib(N, 4));
  printf("%d ", fib(N, 5));
  printf("%d ", fib(N, 6));
  printf("%d ", fib(N, 7));
}
