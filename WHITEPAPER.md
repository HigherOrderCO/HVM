# HVM2: Higher-order Virtual Machine 2

HVM2 is a massively parallel runtime for a superset of Interaction Combinators,
a graph-based model of computation. Unlike existing parallel solutions such as
CUDA, ArrayFire and OpenML, HVM2 is extremely general, being capable of handling
higher-order functions, algebraic datatypes, pattern-matching, recursion, loops,
continuations, mutable references and many other features available on modern
programming languages. As such, it is a suitable compile target for languages
seeking massive parallelism, yet don't fit the ultra low-level, array-based,
grid-like constraints of existing solutions.

Despite its generality, HVM2 is extremely minimal. Its memory consits of a
simple graph with just 8 node types. To compile modern languages like Python and
Haskell to HVM2, one has to translate its concepts to nets. For example,
datatypes like lists and maps can be trasnlated to CON nodes (via λ-encodings),
and loops and recursion can be done with DUP nodes. Even features like
continuations and mutable references have direct counterparts on HVM2, which
isn't bound to the functional paradigm. For a complete reference, please check
<...>, a user-friendly language that compiles to HVM2.

This documents is a complete specification of the HVM2 standard, 32-bit version,
which will be supported by HOC in the long term. If you're developing an
HVM2-compliant runtime, this is the minimal system you need to implement, and
doing so will let you run all languages that target HVM2. If you're a language
developer aiming for the HVM2, this is the most stable IR language you can
target - although, in that case, it is recommended to target Bend, as it takes
care of complex transformations, while still giving you direct HVM2 access, with
no loss of performance. This document will be later updated to include a 64-bit
version, which will be backwards compatible with the 32-bit one.

# Abstract Syntax Tree (AST)

HVM2's AST consists of an interaction system with 8 node types. Nodes form a
tree-like structure in memory. It is represented textually as:

    <Node> ::=
        | <name>                 -- (VAR)iable
        | "*"                    -- (ERA)ser
        | "@" <name>             -- (REF)erence
        | <number>               -- (NUM)ber
        | "(" <Node> <Node> ")"  -- (CON)structor
        | "{" <Node> <Node> "}"  -- (DUP)licator
        | "$(" <Node> <Node> ")" -- (OPE)rator
        | "?(" <Node> <Node> ")" -- (SWI)tch

The first 4 node types (VAR, ERA, REF, NUM) are nullary, and the last 4 types
(CON, DUP, OPE, SWI) are binary. Since Interaction Nets are graphs, every node
has an extra edge, also called the *main port*. That means nullary nodes have 1
port, and binary nodes have 3 ports. This port is omitted from the tree, but is
shown when two main ports connect, forming a *redex*, represented textually as:

    <Redex> ::= <tree> "~" <tree>

An HVM2 Net consists of a root tree, plus a list of "&"-separated redexes:
    
    <Net> ::= <Node> ["&" <Redex>]

An HVM2 Book consists of a list of top-level definitions (named nets):

    <Book> ::= "@" <name> "=" <Net>

Each `.hvm2` file contains a book, which is executed by HVM2.

## Interpretation

Semantically, `CON`, `DUP` and `ERA` nodes correspond to the symbols described
in the original Interaction Combinator paper, and behave almost identically. The
`VAR` node represents a wiring in the graph, connecting two points of the
program. They are linear but paired: at any given moment, there is always
exactly two copies of the same name in a program's memory.

`REF` nodes are an addition to the IC formulation, and they represent a static
closed net, which can be expanded in a single pass. While not essential for the
expressivity of the system, `REF` nodes are essential for performance reasons,
as they allow us to implement fast global functions, they give us a degree of
laziness in a strict setup (which contributes to making GPU implementations
viable), and they allow us to represnt tail recursion in constant space.

`NUM`, `OPE` and `SWI` nodes are also not essential expressivity-wise, but are,
too, important for performance reasons. Most modern processors are equiped with
native machine integer operations, including addition, multiplication and
others. Emulating these with explicit datatypes would be very inefficient, thus,
these numeric nodes are necessary for HVM2 to be efficient in practice.

## Example

The following definition:

    @succ = ({(a b) (b R)} (a R))

Represents the HVM2 encoding of the `λn λs λz (s n)` λ-term, and can be drawn as
the following Interaction Combinator net:

            :
           /_\
        ...: :...
        :       :
       /#\     /_\
     ..: :..   a R
    /_\   /_\  : :
    a b...b R..:.:
    :..........:

Notice how CON/DUP nodes in HVM2 correspond directly to constructor and
duplicator nodes on Interaction Combinators. Aux-to-main wires are represented
by the tree-like structure of the textual notation, while aux-to-aux wires are
represented by two variable nodes, which are always paired.

# Interactions

The AST above specifies HVM's data format. As a Virtual Machine, beyond data, it
also provides a mechanism to compute with that data. In traditional VMs, these
are called *instructions*. In term rewriting systems, there are usually
*reductions*. In HVM2, the mechanism for computation is called *interactions*.
There are 8 of them. All interactions are listed below, using a sequent calculus
notation: the line above computes to the line below.

## 1. LINK

### Text View:

    A ~ B
    ========== LINK
    link(A, B)

Links two ports `A` and `B`. If either `A` or `B` are var, a *global
substitution* is formed. If neither is a VAR, a *new redex* is created.

## 2. CALL

### Text View:

    @foo ~ B
    ================ CALL
    expand(@foo) ~ B

Expands a REF `@foo`, replacing it with its definition. The definition is
copied, with fresh VARs. 

## 3. VOID

### Text View:

    * ~ *
    ===== VOID
    void

### Graph View:

    () ---- ()
    ========== VOID
    void

Erases two nullary nodes (REF/ERA/NUM) connected to each other. This The result
is nothing: both nodes are consumed, fully cleared from memory. The VOID rule
completes a garbage collection process.

## 4. ERAS

### Text View:

    * ~ (B1 B2)
    =========== ERAS   
    * ~ B1
    * ~ B2

### Graph View:

    () ____ /| ---- B2
            \| ---- B1
    ==================
        () ---- B2
        () ---- B1

Erases a binary node `(B1 B2)` connected to an nullary node (REF/ERA/NUM),
propagating the nullary node towards both ports of the binary one. The ERAS rule
performs a granular, parallel garbage collection of nets that go out of scope.

When the nullary node is a NUM or a REF, the ERAS rule actually behaves as a
copy operation, cloning the NUM or REF, and connecting to both ports.

This rule has one exception: when a copy operation is applied to a REF which
contains DUP nodes, it instead is computed as a normal CALL operation. This
allows us to perform fast copy of "function pointers", while still preserving
Interaction Combinator semantics.

## 5. ANNI

### Text View:

    (A1 A2) ~ (B1 B2)
    ================= ANNI
    A1 ~ B1
    A2 ~ B2

### Graph View:

    A1 ---- |\ ____ /| ---- B2
    A2 ---- |/      \| ---- B1
    ========================== ANNI
    A1 ---------,,--------- B2
    A2 ---------''--------- B1

Annihilates two binary nodes of the same type (CON/DUP/OPE/SWI) connected to
each-other, replacing them with two redexes. The ANNI rule is the most essential
computation rule, and is used to implement beta-reduction and pattern-matching.

## 6. COMM

### Text View:

    (A1 A2) ~ {B1 B2}
    ================= COMM
    {x y} ~ A1
    {z w} ~ A2
    (x z) ~ B1
    (y w) ~ B2

### Graph View:

     A1 ---- |\____/# ---- B2
     A2 ---- |/    \# ---- B1
    ========================== COMM
    A1 ____ /#------|\ ____ B2
            \#--,,--|/
    A2 ____ /#--''--|\ ____ B1
            \#------|/

Commutes two binary nodes (CON/DUP/OPE/SWI) of different types, essentially
cloning them. The COMM rule can be used to clone data and to perform loops and
recursion, although these are preferably done via CALLs.

## 7. OPER

### Text View:

    #A ~ $(B1 B2)
    ===============
    if B1 is #B:
      #OP(A,B) ~ B2
    else:
      B1 ~ $(#A B2)

Performs a numeric operation between two NUM nodes `#A` and `#B` connected by an
OPE node. If `B1` is not a NUM, it is partially applied instead. There is only
one binary operation interaction. Dispatching to different native numeric
operations depends on the numbers themselves. More on that later.

Note that the else branch, where the operands are swapped, is not counted as an
interaction, as it would make the interaction count non-deterministic. (The
interaction count remains a linear cost model, however, as this "swap"
pseudo-interaction happens at most once per true OPER interaction).

## 8. SWIT

### Text View:

    #A ~ ?(B1 B2)
    =============
    if A == 0:
      B1 ~ (B2 *)
    else:
      B1 ~ (* (#A-1 B2))

Performs a switch on a NUM node `#A` connected to a SWI node, treating it like a
`Nat ::= Zero | (Succ pred)`. Here, `B1` is expected to be a tuple with both
cases: `zero` and `succ`, and `B2` is the return port. If `A` is 0, we return
the `zero` case, and erase the `succ` case. Otherwise, we return the `succ` case
applied to `A-1`, and erase the `zero` case.

# Interaction Table

Since there are 8 node types, there is a total of 64 possible pairwise node
interactions. The table below shows which interaction rule is triggered for each
possible pair of nodes that form a redex.

| A\B |  VAR |  REF |  ERA |  NUM |  CON |  DUP |  OPR |  SWI |
|-----|------|------|------|------|------|------|------|------|
| VAR | LINK | CALL | LINK | LINK | LINK | LINK | LINK | LINK |
| REF | CALL | VOID | VOID | VOID | CALL | ERAS | CALL | CALL |
| ERA | LINK | VOID | VOID | VOID | ERAS | ERAS | ERAS | ERAS |
| NUM | LINK | VOID | VOID | VOID | ERAS | ERAS | OPER | SWIT |
| CON | LINK | CALL | ERAS | ERAS | ANNI | COMM | COMM | COMM |
| DUP | LINK | ERAS | ERAS | ERAS | COMM | ANNI | COMM | COMM |
| OPR | LINK | CALL | ERAS | OPER | COMM | COMM | ANNI | COMM |
| SWI | LINK | CALL | ERAS | SWIT | COMM | COMM | COMM | ANNI |

Notice that, the linear nature of Interaction Combinators (in the sense there
are no shared pointers or references, and every node can only be accessed by one
part of the program), plus the fact that all interactions are local, allow HVM2
programs to be executed in parallel: after all, at any point of the execution,
all available redexes can be processed in isolation without interfering on
each-other.

Furthermore, interaction nets have a special property called *strong
confluence*, which ensures that, regardless of which order redexes are reduced,
the total amount of work (interactions) will remain constant. This, in turn,
allows us to safely go ahead and evaluate parallel, with no risk of accidental
complexity blowups. This property also gives us a perfect measure of the cost of
execution, which can be useful in contexts such as cloud computing and
peer-to-peer VM execution.

# Substitution Map / Atomic Linker

There is one exception to the proposed isolation of HVM2's interaction rules,
though: variables. That's because they link two different parts of the program,
and, thus, can cause interferences when two threads compute two redexes in
parallel. For example, consider the Net:

    & (a b) ~ (d c)
    & (c d) ~ (f e)

In the graphical point of view, it can be drawn as:

          Thread_0      Thread_1
    a----a|\____/|c----c|\____/|e----e
    b----b|/    \|d----d|/    \|f----f

Note that, in order to execute the first redex, `Thread_0` affects variables `c`
and `d`, which `Thread_1` is currently reading. This requires synchronization.
In HVM2, this is accomplished by a global, atomic substitution map.

The Substitution Map keeps track of partially substituted variables. That is,
when a variable is linked to a node, or to another variable, it forms a
substitution. When that same variable is linked again (remember that variables
always come in pairs), then the connection is made.

That substitution map can be represented efficiently in a computer with a flat
buffer, where the index is the variable name, and the value is the node that has
been substituted. This can be done atomically, via a simple lock-free linker:

```python
def link(subst: Map<Name, Node>, A: Node, B: Mode):
    # Attempts to link A and B.
    loop:
        # If A is not a VAR: swap A and B, and continue.
        if type A != VAR:
            swap(A, B)

        # If A is not a VAR: both are non-vars. Create a new redex.
        if type A != VAR:
            push_redex(A, B)

        # Here, A is a VAR. Create a `A: B` entry in the map.
        Port got = atomic_exchange(map[A], B)

        # If there was no `A` entry, stop.
        if got is None:
            break

        # Otherwise, delete `A` and link `got` to `B`.
        delete map[A]
        A = got
```

To see how this algorithm works, let's consider, again, the scenario above:

            Thread 0      Thread 1

      a----a|\____/|c----c|\____/|e----e
      b----b|/    \|d----d|/    \|f----f

Assume we start with a substitution `a: 42`, and let both threads compute a
redex in parallel. Since both redexes are an `ANNI` rule, their effect is to
link both ports; thus, at the end, focusing on the `a` line, we must have:

     42--------,              ,--------e
                '------------'

That is, `e` must be directly linked to `42`. Let's now evaluate the algorithm
in an arbitrary order, and see what happens, in a step-by-step manner. Remember
that the initial Net is:

    & (a b) ~ (d c)
    & (c d) ~ (f e)

And we're observing ports `a`, `d` and `e`. Two links must be performed:
`link(a,d)` and `link(d,e)`. There are many possible orders of execution:

## Example Order 1

    - a: 42
    ======= Thread_2: link(d,e)
    - a: 42
    - d: e
    ======= Thread_1: link(a,d)
    - a: d
    - d: e
    ======= Thread_1: got `a: 42`, thus, delete `a` and link(d,42)
    - d: 42
    ======= Thread_1: got `d: e`, thus, delete `d` and link(e,42)
    - e: 42

The final result is, indeed, linking `e` to `42`, as demanded.

## Example Order 2

    - a: 42
    ======= Thread_1: link(d,a)
    - a: 42
    - d: a
    ======= Thread_2: link(d,e)
    - a: 42
    - d: e
    ======= Thread_2: got `d: a`, thus, delete `d` and link(a,e)
    - a: e
    ======= Thread_2: got `a: 42`, thus, delete `a` and link(e,42)
    - e: 42

The final result is, again, linking `e` to `42`, as demanded.

## Example Order 3

    - a: 42
    ======= Thread_1: link(d,a)
    - a: 42
    - d: a
    ======= Thread_2: link(e,d)
    - a: 42
    - d: a
    - e: d
    
In this case, the result isn't directly linking `e` to `42`. But it does link
`e` to `d`, which links to `a`, which links to `42`. Thus, `e` is, indirectly,
linked to `42`. While it does temporarily use more memory in this case, it is,
semantically, the same result. And the indirect links will be cleared as soon as
`e` is linked to something else. It is easy enough to see that this holds for
all possible evaluation orders.

# Numbers

HVM2 has a built-in number system represented by the NUM node type. Numbers in
HVM2 have a tag (representable in 5 bits) and a 24-bit payload. Depending on the
tag, numbers can represent unsigned integers (U24), signed integers (I24), IEEE
754 binary32 floats (F24), or partially applied operators. These choices mean
any number can be represented in 29 bits, which can be unboxed in a 32-bit
pointer with a 3 bit tag for the node type. Larger numeric types will be added
to future revisions of HVM2's standard.

## Number Tags

There are three number tags that represent types:
```
tag  SYM syntax
---------------
U24  [u24]
I24  [i24]
F24  [f24]
```

And fifteen that represent operations:
```
tag      SYM syntax
-------------------
ADD      [+]
SUB      [-]
MUL      [*]
DIV      [/]
REM      [%]
EQ       [=]
NEQ      [!]
LT       [<]
GT       [>]
AND      [&]
OR       [|]
XOR      [^]
FLIP-SUB [:-]
FLIP-DIV [:/]
FLIP-REM [:%]
```

Finally, there is the `SYM` tag, which is treated specially and is used to cast
between tags.

### The SYM Operation

The SYM operation is special — its payload represents another numeric tag, and
when it is applied to another number, it combines the stored numeric tag with
the other payload, effectively casting the other number to a different numeric
tag. For example, `[+]` is a SYM number with payload ADD, and when it operates
on `1` (a U24 number with payload `0x000001`), it outputs `[+1]` (an ADD number
with payload `0x000001`).

### U24 - Unsigned 24-bit Integer 

U24 numbers represent unsigned integers from 0 to 16,777,215 (2^24 - 1).

The 24-bit payload directly encodes the integer value. For example:

    0000 0000 0000 0000 0000 0001 = 1
    0000 0000 0000 0000 0000 0010 = 2
    1111 1111 1111 1111 1111 1111 = 16,777,215


### I24 - Signed 24-bit Integer

I24 numbers represent signed integers from -8,388,608 to 8,388,607.

The 24-bit payload uses two's complement encoding. For example:

    0000 0000 0000 0000 0000 0000 = 0 
    0000 0000 0000 0000 0000 0001 = 1
    0111 1111 1111 1111 1111 1111 = 8,388,607  
    1000 0000 0000 0000 0000 0000 = -8,388,608
    1111 1111 1111 1111 1111 1111 = -1

### F24 - 24-bit IEEE 754 binary32 Float 

F24 numbers represent a subset of IEEE 754 binary32 floating point numbers.

The 24-bit payload is laid out as follows:

    SEEE EEEE EMMM MMMM MMMM MMMM

Where:
- S is the sign bit (1 = negative, 0 = positive) 
- E is the 7-bit exponent, with a bias of 63
- M is the 16-bit significand precision

The value is calculated as:
- If E = 0 and M = 0, the value is signed zero
- If E = 0 and M != 0, the value is a subnormal number:  
  (-1)^S * 2^-62 * (0.M in base 2)
- If 0 < E < 127, the value is a normal number:
  (-1)^S * 2^(E-63) * (1.M in base 2)
- If E = 127 and M = 0, the value is signed infinity
- If E = 127 and M != 0, the value is NaN (Not-a-Number)

F24 supports a range from about -3.4e38 to 3.4e38. The smallest positive normal
number is 2^-62 ≈ 2.2e-19. Subnormal F24 numbers go down to about 1.4e-45.

## Number Operations

When two NUM nodes are connected by an OPE node, a numeric operation is
performed. The operation to be performed depends on the tags of each number.

Some operations are invalid, and simply return zero:
- If both number tags represent types, the result is zero.
- If both number tags represent operations, the result is zero.
- If both number tags are SYM, the result is zero.

Otherwise:
- If one of the tags is SYM, the output has the tag represented by the SYM
  number and the payload of the other operand. For example:

  ```
  OP([+], 10) = [+10]
  OP(-1, [*]) = [*0xffffff]
  ```

- If one of the tags is an operation, and the other is a type, a native
  operation is performed, according to the following table:

  |   | U24 | I24 | F24   |
  |---|-----|-----|-------|
  |ADD| +   | +   | +     |
  |SUB| -   | -   | -     |
  |MUL| *   | *   | *     |    
  |DIV| /   | /   | /     |
  |REM| %   | %   | %     |
  |EQ | ==  | ==  | ==    |
  |NEQ| !=  | !=  | !=    |
  |LT | <   | <   | <     |
  |GT | >   | >   | >     |
  |AND| &   | &   | atan2 |
  |OR | \|  | \|  | log   |
  |XOR| ^   | ^   | pow   |

  The result type is the same as the input type, except for comparison
  operators (EQ, NEQ, LT, GT) which always return U24 0 or 1.

  The number tagged with the operation is the left operand of the native
  operation, and the number tagged with the type is the right operand.

  Note that this means that the number type used in an operation is always
  determined by the right operand; if the left operand is of a different type,
  its bits will be reinterpreted.

  Finally, flipped operations (such as `FLIP-SUB`) interpret their operands in
  the opposite order (e.g. `SUB` represents `a-b` whereas `FLIP-SUB` represents
  `b-a`). This allows representing e.g. both `1 - x` and `x - 1` with
  partially-applied operations (`[-1]` and `[:-1]` respectively).

  ```
  OP([-2], +1) = +1
  OP([:-2], 1) = -1
  ```

Note that `OP` is a symmetric function (since the order of the operands is
determined by their tags). This makes the "swap" pseudo-interaction in OPER
valid.
