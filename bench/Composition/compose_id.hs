-- Applies a function N times to an argument
apply :: Int -> (a -> a) -> a -> a
apply 0 f x = x
apply n f x = apply (n `div` 2) (\k -> f (f k)) ((if n `mod` 2 == 0 then f else (\x -> x)) x)

-- Applies `id` 1k times to an argument
kid :: a -> a
kid x = apply 1000 (\x -> x) x

-- Applies `kid` 10m times to 42
main :: IO ()
main = print $ apply 10000000 kid 42
