loop = λn
  switch n {
    0: 0
    _: (loop n-1)
  }

fun = λn
  switch n {
    0: (loop 65536)
    _: (+ (fun n-1) (fun n-1))
  }

main = 
  (fun 18)
