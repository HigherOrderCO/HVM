function sum(a, b) {
  if (a === b) {
    return a;
  } else {
    let mid = Math.floor((a + b) / 2);
    let fst = sum(a, mid + 0);
    let snd = sum(mid + 1, b);
    return fst + snd;
  }
}

console.log(sum(0, 10000000));
