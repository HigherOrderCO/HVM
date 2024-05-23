#flavor core

sum = λn λx switch n {
  0: x
  _: let fst = (sum n-1 (+ (* x 2) 0))
     let snd = (sum n-1 (+ (* x 2) 1))
     (+ fst snd)
}

main = (sum 20 0)
