How!?
=====

![magic](https://c.tenor.com/md3foOULKGIAAAAC/magic.gif)

JK

How it works?
=============

HVM is, in essence, just a virtual machine that evaluates terms in its core
language. So, before we dig deeper, let's review that language.

HVM's Core Language Overview
============================

HVM's Core is a very simple language that resembles untyped Haskell. It features
lambdas (eliminated by applications), constructors (eliminated by user-defined
rewrite rules) and machine integers (eliminated by operators).

```
term ::=
  | λvar. term               # a lambda
  | (term term)              # an application
  | (ctr term term ...)      # a constructor
  | num                      # a machine int
  | (op2 term term)          # an operator
  | let var = term; term     # a local definition

rule ::=
  | term = term

file ::= list<rule>
```

A constructor begins with a name and is followed by up to 16 fields. Constructor
names must start uppercase. For example, below is a pair with 2 numbers:

```
(Pair 42 7)
```

HVM files consist of a series of `rules`, each one having a left-hand side and a
right-hand side. These rules enact a computational behavior where every
occurrence of the left-hand side must be replaced by the right-hand side. For
example, below is a rule that gets the first element of a pair:

```
(Fst (Pair x y)) = x
```

Once that rule is enacted, the `(Fst (Pair 42 7))` will be reduced to `42`. Note
that `(Fst ...)` is itself just constructor, even though it is used like a
function. This has important consequences, which will be elaborated later. From
this, the remaining syntax should be pretty easy to guess. For example, below we
define and use the List `Map` function:

```
(Map f Nil)         = Nil
(Map f (Cons x xs)) = (Cons (f x) (Map f xs))

(Main) =
  let list = (Cons 1 (Cons 2 Nil))
  let add1 = λx (+ x 1)
  (Map add1 list)
```

By running this file (with `hvm r main`), HVM outputs `(Cons 2 (Cons 3 Nil))`,
which incremented each number on the `[2,3]` list. Notes:

- Application is distinguised from constructors by case (`(f x)` vs `(F x)`).

- The parenthesis can be omitted from unitary constructors (`Nil` == `(Nil)`)

- You can abbreviate applications (`(((f a) b) c ...)` == `(f a b c ...)`).

The secret sauce
================

What makes HVM special, though, is **how** it evaluates its programs. HVM has
one simple trick that hackers don't want you to know, and this one trick is
responsbile for its major benefits: beta-optimality, parallelism, and no garbage
collection. But before we get too technical, we must first talk about
**clones**, and how their work obsession ruin everything, for everyone. This
will give you an intuition on the context, and why things are the way they are.
But if you just want to see the tech, feel free to skip this section.

### Clones are workaholics monsters that ruin everyting

By clones, I mean when a value is copied, shared, replicated, or whatever you
call it. For example, consider the JavaScript program below:

```
function foo(x, y) {
  return x + x;
}
```

To compute `foo(2, 3)`, the number `2` must be **cloned** before multiplying it
with itself. This seemingly innocent operation has made a lot of people very
confused and been widely regarded as the hardest problem of the 21th century.
The main issue with clones is how they interact with **order of evaluation**.
For example, consider the `foo(2 * 2, 3 * 3)` expression. In a **strict**
language, it goes like this:

```javascript
foo(2 * 2, 3 * 3) // the input
foo(4    , 9    ) // arguments are reduced
4 + 4             // foo is applied
8                 // the output
```

But this computation has a silly issue: the `3 * 3` value is not necessary to
produce the output, so the `3 * 3` multiplication was wasted work. This led to
the idea of **lazy** evaluation:

```javascript
foo(2 * 2, 3 * 3) // the input
(2 * 2) + (2 * 2) // foo is applied
4       + 4       // arguments are reduced
8                 // the output
```

Notice how the `3 * 3` expression was never computed, saving work. But, in turn,
the same `2 * 2` expression was computed twice, resulting in **wasted work**!
That's because `x` is used two times in the body of `foo`, which causes the
`2*2` expression to be **cloned**, and, thus, computed twice. In other words,
these evil clones ruined the virtue of laziness. Was /r/antiwork right all
along?

### Everyone's solution: ban the evil clones!

Imagine a language without clones. Such language would be computationally
perfect. Lazy evaluators wouldn't waste work, since expressions can't be cloned.
Garbage collection would be cheap, because every object would only have one
reference. Parallelism would be trivial, because there would be no simultaneous
accesses to the same object. Sadly, such language wouldn't be practical. Imagine
never being able to copy anything? Because of that, real languages must find a
way to let their users replicate values, without impacting the features they
desire, all while avoiding these expensive clones. And that's almsot always...
references.

For example, Haskell is lazy. To avoid "cloning computations", it implements
thunks, which are nothing but *mamoized references* to shared expressions. Rust
is GC-free, so every object has only one "owner". To avoid too much cloning, it
implements a complex *shared references* system, based on borrows. Finally,
parallel languages require mutexes and atomics to synchronize access to *shared
references*. In other words, references saved the world by letting us avoid
these evil clones, and that's great... right? Right? *awkward silence*

> clone wasn't the impostor

References. **They** ruin everything. They're the reason Rust is so hard to use.
They're the reason parallel programming is so complex. They're the reason Haskell
isn't optimal. Yes, that's right! Thunks break down when they have free
variables, demanding a whole de-sharing system that makes the glorious GHC be
beaten by a 1-month prototype in the same class of programs it should thrive. It
isn't GHC's fault  References are the culprit.

### HVM's solution: lofi and chill with clones

Clones aren't evil. They just need to relax.

Once you understand the context above, grasping how HVM can be optimally lazy,
non-GC'd and inherently parallel is easy. On its base, it has the same "linear"
core that both Haskell and Rust share in common (which, as we've just agreed, is
already all these things). The difference is that, instead of adding some kind
of clever reference system to circumvent the cost of cloning... **HVM introduces
a pervasive, lazy clone primitive**. Yes, that's it.

**HVM's runtime featuers a `.clone()` primitive that has zero cost, until the
cloned value needs to be read. Once that's the case, instead of copying it
whole, it does so layer by layer, on-demand.**

For the purpose of lazy evaluation, HVM's lazy clones works like Haskell's
thunks, except they do not break down on lambdas. For the context of garbage
collection, since the data is actually copied, there are no shared references,
so memory can be freed when values go out of scope.  For the same reason,
parallelism becomes trivial, and the runtime's `reduce()` procedure is almost
entirely thread safe, requiring minimal synchronization.

In other words, think of HVM as Rust, except replacing the burrow system by a
very cheap `.clone()` operation that can be used and abused with no mercy. This
is the secret souce! Easy, right? Well, no. There is still a big problem to be
solved: **how the hell do we incrementally clone a lambda?** The answer to this
problem is what made it all possible. Let's get technical!

HVM's Rewrite Rules
===================

HVM is, in essence, a graph-rewrite system, which means that all it does is
repeatedly rewrite terms in memory until there is no more work left to do. These
rewrites come in two forms: user-defined and primitive rules.

User-defined Rules
------------------

User-defined rules are generated from equations in a file. For example, the
following equation:

```
(Foo (Tic a) (Tac b)) = (+ a b)
```

Generates the following rewrite rule:

```
(Foo (Tic a) (Tac b))
--------------------- Foo-Rule-0
(+ a b)
```

It should be read as "the expression above reduces to the expression below".
So, for example, `Foo-rule-0` dictates that `(Foo (Tic 42) (Tac 7))` reduces to
`(+ 42 7)`. As for the primitive rules, they deal with lambdas, native numbers
and the duplication primitive. Let's start with numeric operations.

Operation
---------

```
(<op> x y)
------- Op2-U32
x +  y if <op> is +
x -  y if <op> is -
x *  y if <op> is *
x /  y if <op> is /
x %  y if <op> is %
x &  y if <op> is &
x |  y if <op> is |
x ^  y if <op> is ^
x >> y if <op> is >>
x << y if <op> is <<
x <  y if <op> is <
x <= y if <op> is <=
x == y if <op> is ==
x >= y if <op> is >=
x >  y if <op> is >
x != y if <op> is !=
```

This should be read as: *"the addition of `x` and `y` reduces to `x + y`"*. This
just says that we can perform numeric operations on HVM. For example, `(+ 2 3)`
is reduced to `5`, `(* 5 10)` is reduced to `50`, and so on. HVM numbers are
32-bit unsigned integers, but more numeric types will be added in a future.

Number Duplication
------------------

```
dup x y = N
-----------
x <- N
y <- N
```

This should be read as: *"the duplication of the N number as `x` and `y` reduces
to the substitution of `x` by a copy of `N`, and of `y` by another copy of `N`.
Before explaining what is going on here, let me also present the constructor
duplication rule below.

Constructor Duplication
-----------------------

```
dup x y = (Foo a b ...)
------------------------- Dup-Ctr
dup a0 a1 = a
dup b0 b1 = b
...
x <- (Foo a0 b0 ...)
y <- (Foo a1 b1 ...)
```

This should be read as: *"the duplication of the `(Foo a b ...)` constructor as
`x` and `y` reduces to the duplication of `a` as `a0` and `a1`, the duplication
of `b` as `b0` and `b1`, and the substitution of `x` by `(Foo a0 b0 ...)` and
the substitution of `y` by `(Foo a1 b1)`"*.

There is al ot of new information here, so, before moving on, let's dissect it
all one by one.

**1.** What the hell is `dup`? That is an **internal duplication node**. You
can't write it directly on the user-facing language; instead, it is inserted
by the pre-processor whenever you use a variable more than once. For example,
the equation below:

```
(Foo a) = (+ a a)
```

Is actually replaced by:

```
(Foo a) =
  dup x y = a
  (+ x y)
```

At compile time. Because of that transformation, **every runtime variable only
occurs once**. The effect of `dup` is that of cloning an expression, and moving
it to two locations. For example, the program below:

```
dup x y = 42
(Pair x y)
```

Is reduced to:

```
(Pair 42 42)
```
  
**2.** By "substition", we mean "replacing a variable by a value". For example,
the substitution of `x` by `7` in `[1, x, 8]` would be `[1, 7, 8]`. Since every
variable only occurs once in the runtime, substitution is a fast, constant time
operation that performs either 1 or 2 array writes.

**3.** `dup`s aren't stored inside the expressions. Instead, they "float" on
the global scope. That's why they're always written on top.

**4.** Remember that `dup` (like all other rules) is only triggered when it
is needed, due to lazy evaluation. That's what makes it ultra-cheap. In a way,
it is as if HVM's added a `.clone()` to every variable used more than once. And
that's fine.

**5.** Even though the user-facing language doesn't makes no distinction between
constructors and functions, the runtime does, for optimality purposes.  The
effect this has is that, if a constructor is used in a functional position of an
equation, then it is flagged as a function, and the duplication rule is NOT
triggered for it. For example, consider the programs below:

```javascript
(Map f Nil)         = Nil
(Map f (Cons x xs)) = (Cons (f x) (Map f xs))
```

```javascript
(Head (Map f xs)) = (f (Head xs))
(Tail (Map f xs)) = (Map f (Tail xs))
```

On the first one, `Map` is flagged as a function, and its introducers, `Nil` and
`Cons`, as constructors. On the second one, its eliminators, `Head` and `Tail`,
are flagged as functions, and `Map` as a constructor. As such, `Map` can NOT be
duplicated on the first program, but it CAN be duplicated on the second. Welcome
to the upside down world of codata!

#### Example

Now that you know all that, let's watch `dup` in action, by visualizing how the
`[1+1, 2+2, 3+3]` list is cloned. Lines separate reduction steps.

```javascript
dup x y = (Cons (+ 1 1) (Cons (+ 2 2) (Cons (+ 3 3) Nil)))
(Pair x y)
------------------------------------------- Dup-Ctr
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
dup a b = (+ 1 1)
(Pair
  (Cons a x)
  (Cons b y)
)
------------------------------------------- Op2-U32
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
dup a b = 2
(Pair
  (Cons a x)
  (Cons b y)
)
------------------------------------------- Dup-U32
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
(Pair
  (Cons 2 x)
  (Cons 2 y)
)
------------------------------------------- Dup-Ctr
dup x y = (Cons (+ 3 3) Nil)
dup a b = (+ 2 2)
(Pair
  (Cons 2 (Cons a x))
  (Cons 2 (Cons b y))
)
------------------------------------------- Op2-U32
dup x y = (Cons (+ 3 3) Nil)
dup a b = 4
(Pair
  (Cons 2 (Cons a x))
  (Cons 2 (Cons b y))
)
------------------------------------------- Dup-U32
dup x y = (Cons (+ 3 3) Nil)
(Pair
  (Cons 2 (Cons 4 x))
  (Cons 2 (Cons 4 y))
)
------------------------------------------- Dup-Ctr
dup x y = Nil
dup a b = (+ 3 3)
(Pair
  (Cons 2 (Cons 4 (Cons a x)))
  (Cons 2 (Cons 4 (Cons b y)))
)
------------------------------------------- Op2-U32
dup x y = Nil
dup a b = 6
(Pair
  (Cons 2 (Cons 4 (Cons a x)))
  (Cons 2 (Cons 4 (Cons b y)))
)
------------------------------------------- Dup-U32
dup x y = Nil
(Pair
  (Cons 2 (Cons 4 (Cons 6 x)))
  (Cons 2 (Cons 4 (Cons 6 y)))
)
------------------------------------------- Dup-Ctr
(Pair
  (Cons 2 (Cons 4 (Cons 6 Nil)))
  (Cons 2 (Cons 4 (Cons 6 Nil)))
)
```

In the end, we made two copies of the list. Note how the `(+ 1 1)` expression,
was NOT "cloned". It only happened once, even though we evaluated the program
lazily. And, of course, since the cloning itself is lazy, if we only needed
parts of the list, we wouldn't need to make two full copies. For example,
consider the following program instead:

```
dup x y = (Cons (+ 1 1) (Cons (+ 2 2) (Cons (+ 3 3) Nil)))
(Pair (Head x) (Head (Tail y)))
------------------------------------------- Dup-Ctr
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
dup a b = (+ 1 1)
(Pair (Head (Cons a x)) (Head (Tail (Cons b y))))
------------------------------------------- Head
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
dup a b = (+ 1 1)
(Pair a (Head (Tail (Cons b y))))
------------------------------------------- Op2-U32
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
dup a b = 2
(Pair a (Head (Tail (Cons b y))))
------------------------------------------- Dup-U32
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
(Pair 2 (Head (Tail (Cons 2 y))))
------------------------------------------- Tail
dup x y = (Cons (+ 2 2) (Cons (+ 3 3) Nil))
(Pair 2 (Head y))
------------------------------------------- Dup-Ctr
dup x y = (Cons (+ 3 3) Nil)
dup a b = (+ 2 2)
(Pair 2 (Head (Cons b y)))
------------------------------------------- Head
dup x y = (Cons (+ 3 3) Nil)
dup a b = (+ 2 2)
(Pair 2 b)
------------------------------------------- Op2-U32
dup x y = (Cons (+ 3 3) Nil)
dup a b = 4
(Pair 2 b)
------------------------------------------- Dup-U32
dup x y = (Cons (+ 3 3) Nil)
(Pair 2 4)
------------------------------------------- Collect
(Pair 2 4)
```

Notice how only the minimal amount of copying was performed. The first part of
the list (`(Cons (+ 1 1) ...`)) was copied twice, the second part (`(Cons (+ 2
2) ...)`) was copied once, and the rest (`(Cons (+ 3 3) Nil)`) was collected
without even touching it. Collection is orchestrated by variables that go out of
scope. For example, in the last lines, `x` and `y` both aren't referenced
anywhere. That triggers collection of the remaining list.

Uuf! That was a LOT of info. Hopefully, by now, you have an intuition on how the
lazy duplication primitive works. You must be tired. But don't worry. The worst
part comes now.

### Lambda Application

```
(λx(body) arg)
-------------- App-Lam
x <- arg
body
```

Okay, I lied. This part is simple. This is the famous beta-reduction rule! This
must be read as: "whenever there is a `(λx(body) arg)` pattern in the runtime,
substitute `x` by `arg` wherever it occurs, and return `body`. For example,
`(λx(Single x) 42)` is reduced to `(Single 42)`. Remember that variables only
occur once. Because of that, beta-reduction is a very fast operation. A modern
CPU can perform more than 200 million beta-reductions per second, in a single
core. As an example:

```
(λxλy(Pair x y) 2 3)
-------------------- App-Lam
(λy(Pair 2 y) 3)
-------------------- App-Lam
(Pair 2 3)
```

Simple, huh? Okay. Now, the worst part.

### Lambda Duplication

Incrementally cloning datatypes is a neat idea. But there is nothing special to
it. In fact, that **is** exactly how Haskell's thunks behave! But, now, take a
moment and ask yourself: **how the hell do we incrementally clone a lambda**?
Well, I won't waste your time. This is how:

```
dup a b = λx(body)
------------------ Dup-Lam
a <- λx0(b0)
b <- λx1(b1)
x <- {x0 x1}
dup b0 b1 = body

dup a b = <r s>
--------------- Dup-Sup
a <- r
b <- s
```

Sorry, **wat**? Well, I told you. This is where things get wild. Perhaps a good
place to start is by writing this in plain English. It reads as: "the
duplication of a lambda `λx(body)` as `a` and `b` reduces in the duplication of
its `body` as `b0` and `b1`, and the substitution of `a` by `λx0(b0)`, `b` by
`λx1(b1)` and `x` by the superposition `{x0 x1}`".

What this is saying is that, in order to duplicate a lambda, we must duplicate
its body. So, far, so good. But we must also duplicate the `λ` itself. And,
then, weird things happen with its variable, and there is a brand new construct,
the superposition, that I haven't explained yet. But, this is fine. Let's try to
do it with an example:

```
dup a b = λx λy (Pair x y)
(Pair a b)
```

This program just makes two copies of the `λx λy (Pair x y)` lambda. But, to get
there, we are not allowed to copy the entire lambda whole.  Instead, we must go
through a series of incremental lazy steps. Let's try it, and copy the outermost
lambda (`λx`):

```
dup a b = λy (Pair x y)
(Pair λx(a) λx(b))
```

Can you spot the issue? As soon as the lambda is copied, it is moved to another
location of the program, which means it gets detached from its own body. Because
of that, the `x` variable gets unbound on the first line, and the body of each
copied `λx` has no reference to `x`. That makes no sense at all! How to solve
this?

First, we must let go of material goods and accept a cruel reality of HVM:
**lambdas don't have scopes**. That is, a variable bound by a lambda can occur
outside of its body. So, for example, `(Pair x (λx(8) 7))` would reduce to
`(Pair 7 8)`. Please, take a moment to make sense out of this... even if looks
like it doesn't.

Once you accept that in your heart, you'll have the program above will make a
little more sense, because we can say the `λx` binder on the second line is
"connected" to the `x` variable on the first line, even if its outside. But
wait, there is still a problem: there are **two** lambdas bound to the same
variable. If the left lambda gets applied to an argument, it should NOT affect
the second one. But, the way it is written, that's what would happen. To work
around this issue, we need a new construct: the **superposition**. Written as
**{r s}**, a superposition stands for an expression that is part of two
partially copied lambdas. So, for example, `(Pair {1 2} 3)` can represent either
`(Pair 1 3)` or `(Pair 2 3)`, depending on its context.

This gives us the tools we need to incrementally copy these lambdas. Here is how
it goes:

```
dup a b = λx(λy(Pair x y))
(Pair a b)
------------------------------------------------ Dup-Lam
dup a b = λy(Pair {x0 x1} y)
(Pair λx0(a) λx1(b))
------------------------------------------------ Dup-Lam
dup a b = (Pair {x0 x1} {y0 y1})
(Pair λx0(λy0(a)) λx1(λy1(b)))
------------------------------------------------ Dup-Ctr
dup a b = {x0 x1}
dup c d = {y0 y1}
(Pair λx0(λy0(Pair a c)) λx1(λy1(Pair b d)))
------------------------------------------------ Dup-Sup
dup c d = {y0 y1}
(Pair λx0(λy0(Pair x0 c)) λx1(λy1(Pair x1 d)))
------------------------------------------------ Dup-Sup
(Pair λx0(λy0(Pair x0 y0)) λx1(λy1(Pair x1 y1)))
```

Wow, did it actually work? Yes, it did. Notice that, despite the fact that,
during the intermediate steps, "weird things" happened; specifically, variables
got out of their own lambda bodies, and parts of the program got temporarily
superposed; in the end, it all worked out, and the result was proper copies of
the original lambdas. This allows us to share computations inside lambdas,
something that GHC isn't capable of. For example, consider the following
reduction:

```
dup f g = ((λx λy (Pair (+ x x) y)) 2)
(Pair (g 10) (g 20))
----------------------------------- App-Lam
dup f g = λy (Pair (+ 2 2) y)
(Pair (f 10) (g 20))
----------------------------------- Dup-Lam
dup f g = (Pair (+ 2 2) {y0 y1})
(Pair (λy0(f) 10) (λy1(g) 20))
----------------------------------- App-Lam
dup f g = (Pair (+ 2 2) {10 y1})
(Pair f (λy1(g) 20))
----------------------------------- App-Lam
dup f g = (Pair (+ 2 2) {10 20})
(Pair f g)
----------------------------------- Dup-Ctr
dup a b = (+ 2 2)
dup c d = {10 20}
(Pair (Pair a c) (Pair b d))
----------------------------------- Op2-U32
dup a b = 4
dup c d = {10 20}
(Pair (Pair a c) (Pair b d))
----------------------------------- Dup-U32
dup c d = {10 20}
(Pair (Pair 4 c) (Pair 4 d))
----------------------------------- Dup-sup
(Pair (Pair 4 10) (Pair 4 20))
```


Notice that the `(+ 2 2)` addition only happened once, even though it was nested
inside two copied lambda binders! In Haskell, this situation would lead to the
un-sharing of the lambdas, and `(+ 2 2)` would happen twice. Notice also how in
some steps, lambdas were applied to arguments that appeared outside of their
bodies. This is all fine, and, in the end, the result is correct.

Uff. That was hard, wasn't it? The good news is the worst part is done. From now
on, nothing too surprising will happen.

Superposed Application
----------------------

Since duplications are pervasive, it may happen that a superposition will end up
in the function position of an application. For example, the situation below can
happen at runtime:

```
({λx(x) λy(x)} 10)
```

This represents two superposed lambdas, applied to an argument `10`. If we let
this expression as is, certain programs would be stuck, and we wouldn't be able
to evaluate them. We need a way out. Because of that, there is a superposed
application rule that deals with that situation:

```
({r s} a)
----------------------- App-Sup
dup x0 x1 = c
{(a x0) (b x1)}
```

That rule also applies to user-defined functions! The logic is the same, only
adapted depending on the arity.

Superposed Operation
--------------------

```
(+ {a0 a1} b)
------------- Op2-Sup-A
dup b0 b1 = b
{(+ a0 b0) (+ a1 b1)}


(+ a {b0 b1})
------------- Op2-Sup-B
dup a0 a1 = a
{(+ a0 b0) (+ a1 b1)}
```

TODO: to be continued...
