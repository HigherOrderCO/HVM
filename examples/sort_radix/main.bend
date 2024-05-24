
# data Arr = Empty | (Single x) | (Concat x0 x1)
Empty  =         λempty λsingle λconcat empty
Single = λx      λempty λsingle λconcat (single x)
Concat = λx0 λx1 λempty λsingle λconcat (concat x0 x1)

# data Map = Free | Busy | (Node x0 x1)
Free = λfree λbusy λnode free
Busy = λfree λbusy λnode busy
Node = λx0 λx1 λfree λbusy λnode (node x0 x1)

# gen : u32 -> Arr
gen = λn switch n {
  0: λx (Single x)
  _: λx
    let x0 = (* x 2)
    let x1 = (+ x0 1)
    (Concat (gen n-1 x1) (gen n-1 x0))
}

# sum : Arr -> u32
sum = λa
  let a_empty  = 0
  let a_single = λx x
  let a_concat = λx0 λx1 (+ (sum x0) (sum x1))
  (a a_empty a_single a_concat)

# sort : Arr -> Arr
sort = λt (to_arr (to_map t) 0)

# to_arr : Map -> u32 -> Arr
to_arr = λa
  let a_free = λk Empty
  let a_busy = λk (Single k)
  let a_node = λx0 λx1 λk
    let x0 = (to_arr x0 (+ (* k 2) 0))
    let x1 = (to_arr x1 (+ (* k 2) 1))
    (Concat x0 x1)
  (a a_free a_busy a_node)

# to_map : Arr -> Map
to_map = λa
  let a_empty  = Free
  let a_single = λx (radix 24 x 1 Busy)
  let a_concat = λx0 λx1 (merge (to_map x0) (to_map x1))
  (a a_empty a_single a_concat)

# merge : Map -> Map -> Map
merge = λa
  let a_free = λb
    let b_free = Free
    let b_busy = Busy
    let b_node = λb0 λb1 (Node b0 b1)
    (b b_free b_busy b_node)
  let a_busy = λb
    let b_free = Busy
    let b_busy = Busy
    let b_node = λb0 λb1 0
    (b b_free b_busy b_node)
  let a_node = λa0 λa1 λb
    let b_free = λa0 λa1 (Node a0 a1)
    let b_busy = λa0 λa1 0
    let b_node = λb0 λb1 λa0 λa1 (Node (merge a0 b0) (merge a1 b1))
    (b b_free b_busy b_node a0 a1)
  (a a_free a_busy a_node)

# radix : u32 -> Map
radix = λi λn λk λr switch i {
  0: r
  _: (radix i-1 n (* k 2) (swap (& n k) r Free))
}

# swap : u32 -> Map -> Map -> Map
swap = λn switch n {
  0: λx0 λx1 (Node x0 x1)
  _: λx0 λx1 (Node x1 x0)
}

# main : u32
main = (sum (sort (gen 16 0)))
