{-# LANGUAGE RankNTypes #-}
n          = 26 :: Int
newtype BS = BS { get :: forall a. a -> (BS -> a) -> (BS -> a) -> a }
comp 0 f x = f x
comp n f x = comp (n - 1) (\k -> f (f k)) x
e          = BS (\e -> \o -> \i -> e)
o pred     = BS (\e -> \o -> \i -> o pred)
i pred     = BS (\e -> \o -> \i -> i pred)
inc bs     = BS (\e -> \o -> \i -> get bs e i (\pred -> o (inc pred)))
zero 0     = e
zero n     = o (zero (n - 1))
toInt bs   = get bs 0 (\n -> toInt n * 2) (\n -> toInt n * 2 + 1)
main       = print $ toInt (comp n (\x -> inc x) (zero 64))
