xFib :: Int -> Int -> Int
xFib 0 z = z
xFib 1 z = z + 1
xFib n z = xFib (n - 1) z + xFib (n - 2) z

main :: IO ()
main = print
  (
    xFib 40 0,
    xFib 40 1,
    xFib 40 2,
    xFib 40 3,
    xFib 40 4,
    xFib 40 5,
    xFib 40 6,
    xFib 40 7
  )
