import System.Environment

-- Computes f^(2^n)
comp :: Int -> (a -> a) -> a -> a
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x

-- Performs 2^n compositions
main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Int
  print $ comp n (\x -> x) (0 :: Int)
