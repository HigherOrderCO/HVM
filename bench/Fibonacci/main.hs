import System.Environment

-- Computes nth Fibonacci number
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Int
  print (fib n)
