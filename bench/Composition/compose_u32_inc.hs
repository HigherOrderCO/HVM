pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print $ pow (26 :: Int) (\x -> x + 1) (0 :: Int)
