HVM on JavaScript
=================

HVM is now available as a JavaScript library. To use it, first install with `npm`:

```
npm i hvm-js
```

Then, import and use it on your JavaScript application:

```javascript
var hvm = require("hvm");

// Instantiates an HVM runtime given a source code
var runtime = hvm.Runtime.from_code(`
(U60.sum 0) = 0
(U60.sum n) = (+ n (U60.sum (- n 1)))
`);

// Evaluates an expression to normal form
console.log(runtime.eval("(U60.sum 10000000)"));
```

You can also handle HVM's memory directly:

```javascript
// Allocates an expression without reducing it
let loc = runtime.alloc_code("(U60.sum 10)");

// Reduces it to weak head normal form:
runtime.reduce(loc);

// If the result is a number, print its value:
let term = runtime.at(loc);
if (hvm.Runtime.get_tag(term) == hvm.Runtime.NUM()) {
  console.log("Result is Num(" + hvm.Runtime.get_val(term) + ")");
}
```

This allows you to reduce a term lazily, layer by layer. This is useful, for
example, to implement IO actions and FFI with your app.

See [examples/javascript.js](examples/javascript.js) for a complete example.
