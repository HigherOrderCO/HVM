function comp(n, f, x) {
  if (n == 0) {
    return f(x)
  } else {
    return comp(n-1, (x) => f(f(x)), x)
  }
}

function main() {
  const N = process.argv[2];
  console.log(comp(N, (x) => x, 0));
}

main();