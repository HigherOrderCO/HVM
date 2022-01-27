n = 32 :: Int

-- Computes f^(2^n)
comp :: Int -> (a -> a) -> a -> a
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x

-- Applies id 2^n times to 0
main :: IO ()
main = print$ comp n (\x->x) (0::Int)
