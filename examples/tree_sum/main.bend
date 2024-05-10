gen = λd switch d {
  0: λx x
  _: λx ((gen d-1 (+ (* x 2) 1)), (gen d-1 (* x 2)))
}

sum = λd λt switch d {
  0: 1
  _: let (t.a,t.b) = t
    (+ (sum d-1 t.a) (sum d-1 t.b))
}

main =
  let d = 28
  (sum d (gen d 0))
