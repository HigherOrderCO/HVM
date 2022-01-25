n          = 32 :: Int
comp 0 f x = f x
comp n f x = comp (n - 1) (\x -> f (f x)) x
main       = print$ comp n (\x->x) (0::Int)
