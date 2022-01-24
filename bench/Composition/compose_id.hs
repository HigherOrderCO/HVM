pow 0 f x = f x
pow n f x = pow (n - 1) (\x -> f (f x)) x
main      = print$ pow (32::Int) (\x->x) (0::Int)
