xFunExp :: Int -> (a -> a) -> (a -> a)
xFunExp 0 fn = fn
xFunExp n fn = xFunExp (n - 1) (\x -> fn (fn x))

main :: IO ()
main = print $ (xFunExp 30 (\x -> x)) 42
