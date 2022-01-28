function fib(n) {
  if (n == 0) {
    return 0;
  }
  if (n == 1) {
    return 1;
  }
  return fib(n - 1) + fib(n - 2);
}

function main() {
  const N = Number(process.argv[2]);
  console.log(fib(N));
}

main();
