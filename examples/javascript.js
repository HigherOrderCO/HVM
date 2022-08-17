var hvm = require("./../pkg");

// Simple Usage
// ============

// Instantiates an HVM runtime given a source code
var runtime = hvm.Runtime.from_code(`
(U60.sum 0) = 0
(U60.sum n) = (+ n (U60.sum (- n 1)))
`);

// Evaluates an expression to normal form
console.log(runtime.eval("(U60.sum 1000000)"));

// Managing Memory
// ===============

// You can also handle HVM's memory directly

// Allocates an expression without reducing it
let loc = runtime.alloc_code("(U60.sum 10)");

// Reduces it to weak head normal form:
runtime.reduce(loc);

// If the result is a number, print its value:
let term = runtime.at(loc);
if (hvm.Runtime.get_tag(term) == hvm.Runtime.NUM()) {
  console.log("Result is Num(" + hvm.Runtime.get_val(term) + ")");
}
