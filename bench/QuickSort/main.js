let ListNil = () => ({_: "Nil"});
let ListCons = head => tail => ({_: "Cons", head, tail});

let TreeEmpty = () => ({_: "Empty"});
let TreeSingle = value => ({_: "Single", value});
let TreeConcat = left => right => ({_: "Concat", left, right});

function randoms(seed, size) {
  if (size == 0) {
    return ListNil();
  } else {
    return ListCons(seed)(randoms((seed * 1664525 + 1013904223) >>> 32, size-1))
  }
}

function sum(tree) {
  switch (tree._) {
    case "Empty":
      return 0;
    case "Single":
      return tree.value;
    case "Concat":
      return sum(tree.left) + sum(tree.right)
  }
}

function qsort(list) {
  switch (list._) {
    case "Nil":
      return TreeEmpty();
    case "Cons":
      switch (list.tail._) {
        case "Nil":
          return TreeSingle(list.head);
        case "Cons":
          return split(list.head, ListCons(list.head)(list.tail), ListNil(), ListNil());
      }
  }
}

function split(p, list, min, max) {
  switch (list._) {
    case "Nil":
      return TreeConcat(qsort(min))(qsort(max))
    case "Cons":
      return place(p, p < list.head, list.head, list.tail, min, max)
  }
}

function place(p, bool, head, tail, min, max) {
  if (bool) {
    return split(p, tail, min, ListCons(head)(max))
  } else {
    return split(p, tail, ListCons(head)(min), max)
  }
}

function main() {
  const N = process.argv[2];
  console.log(sum(qsort(randoms(1, 100000 * N))));
}

main();
