n = 10 :: Int
xFib :: Int -> Int -> Int
xFib 0 z = z
xFib 1 z = z + 1
xFib n z = xFib (n - 1) z + xFib (n - 2) z

main :: IO ()
main = print
  (
    xFib 30 0,
    xFib 30 1,
    
    xFib 30 2,
    xFib 30 3,
    xFib 30 4,
    xFib 30 5,
    xFib 30 6,
    xFib 30 7
  )
