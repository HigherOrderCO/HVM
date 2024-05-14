#flavor core

data Tup8 = (New a b c d e f g h)

rot = λx (x λa λb λc λd λe λf λg λh (Tup8/New b c d e f g h a))

app = λn switch n {
  0: λf λx x
  _: λf λx (app n-1 f (f x))
}

main = (app 20 rot (Tup8/New 1 2 3 4 5 6 7 8))
