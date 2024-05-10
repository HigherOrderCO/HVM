gen = λd switch d {
  0: λx x
  _: λx ((gen d-1 (+ (* x 2) 1)), (gen d-1 (* x 2)))
}

sum = λd λt switch d {
  0: t
  _: let (t.a,t.b) = t
    (+ (sum d-1 t.a) (sum d-1 t.b))
}

swap = λs λa λb switch s {
  0: (a,b)
  _: (b,a)
}

warp = λd λs λa λb switch d {
  0: (swap (^ s (> a b)) a b) 
  _:
    let (a.a,a.b) = a
    let (b.a,b.b) = b
    let (A.a,A.b) = (warp d-1 s a.a b.a)
    let (B.a,B.b) = (warp d-1 s a.b b.b)
    ((A.a,B.a),(A.b,B.b))
}

flow = λd λs λt switch d {
  0: t
  _: let (t.a,t.b) = t
     (down d s (warp d-1 s t.a t.b))
}

down = λd λs λt switch d {
  0: t
  _: let (t.a,t.b) = t
     ((flow d-1 s t.a),(flow d-1 s t.b))
}

sort = λd λs λt switch d {
  0: t
  _: let (t.a,t.b) = t
     (flow d s ((sort d-1 0 t.a),(sort d-1 1 t.b)))
}

main = (sum 16 (sort 16 0 (gen 16 0)))
