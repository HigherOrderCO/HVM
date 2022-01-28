let ListNil = () => ({_: "Nil"});
let ListCons = (head) => (tail) => ({_: "Cons", head, tail});

function fold(list, c, n) {
  switch (list._) {
    case "Nil":
      return n;
    case "Cons":
      let a = c(list.head, fold(list.tail, c, n));
      return a;
  }
}

function range(n, list) {
  if (n == 0) {
    return list;
  } else {
    let m = n - 1;
    return range(m, ListCons(m)(list));
  }
}

function main() {
  const N = process.argv[2];
  let size =  N;
  let list = range(size, ListNil());
  console.log(fold(list, (x, y) => x + y, 0));
}

main();