n         = 26 :: Int
pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print $ pow n (\x -> x + 1) (0 :: Int)
