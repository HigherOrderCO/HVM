type Tup8:
  New { a, b, c, d, e, f, g, h }

rot = λx match x {
  Tup8/New: (Tup8/New x.b x.c x.d x.e x.f x.g x.h x.a)
}

app = λn switch n {
  0: λf λx x
  _: λf λx (app n-1 f (f x))
}

main = (app 1234 rot (Tup8/New 1 2 3 4 5 6 7 8))
