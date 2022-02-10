let TreeLeaf = (value) => ({_: "Leaf", value});
let TreeNode = (left, right) => ({_: "Node", left, right})

// Creates a tree with 2ˆn elements
function gen(n) {
  if (n == 0) {
    return TreeLeaf(1);
  } else {
    return TreeNode(gen(n-1), gen(n-1));
  }
}

// Adds all elements of a tree
function sum(tree) {
  switch(tree._) {
    case "Leaf":
      return tree.value;
    case "Node":
      return sum(tree.left) + sum(tree.right);
  }
}

// Performs 2^n additions
function main() {
  const N = process.argv[2];
  console.log(sum(gen(N)));
}

main();
