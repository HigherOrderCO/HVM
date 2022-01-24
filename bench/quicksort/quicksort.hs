import Data.Word

data List a = Nil | Cons a (List a)

xSum :: List Word32 -> Word32
xSum Nil         = 0
xSum (Cons x xs) = x + xSum xs

xFilter :: (a -> Word32) -> List a -> List a
xFilter _  Nil         = Nil
xFilter fn (Cons x xs) = xConsIf (fn x) x (xFilter fn xs)

xConsIf :: Word32 -> a -> List a -> List a
xConsIf 0 x xs = xs
xConsIf n x xs = Cons x xs

xConcat :: List a -> List a -> List a
xConcat Nil b = b
xConcat (Cons x xs) b = Cons x (xConcat xs b)

xQuicksort :: List Word32 -> List Word32
xQuicksort Nil         = Nil
xQuicksort (Cons x xs) =
  let min = xFilter (\n -> if n < x then 1 else 0) xs
      max = xFilter (\n -> if n > x then 1 else 0) xs
  in xConcat (xQuicksort min) (Cons x (xQuicksort max))

xRandoms :: Word32 -> Word32 -> List Word32
xRandoms seed 0    = Nil
xRandoms seed size = Cons seed (xRandoms (seed * 1664525 + 1013904223) (size - 1))

main :: IO ()
main = do
  print $ (
      (xSum (xQuicksort (xRandoms 0 400000))),
      (xSum (xQuicksort (xRandoms 1 400000))),
      (xSum (xQuicksort (xRandoms 2 400000))),
      (xSum (xQuicksort (xRandoms 3 400000))),
      (xSum (xQuicksort (xRandoms 4 400000))),
      (xSum (xQuicksort (xRandoms 5 400000))),
      (xSum (xQuicksort (xRandoms 6 400000))),
      (xSum (xQuicksort (xRandoms 7 400000)))
    )
