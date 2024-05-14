function sum(n) {
  if (n === 0) {
    return 0;
  } else {
    return n + sum(n - 1);
  }
}

function fun(n) {
  if (n === 0) {
    return sum(4096);
  } else {
    return fun(n - 1) + fun(n - 1);
  }
}

console.log(fun(18));
