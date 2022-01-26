import Data.Word

data List a = Nil | Cons a (List a)

xsum :: List Word32 -> Word32
xsum Nil         = 0
xsum (Cons x xs) = x + xsum xs

xfilter :: (a -> Word32) -> List a -> List a
xfilter _  Nil         = Nil
xfilter fn (Cons x xs) = consIf (fn x) x (xfilter fn xs)

consIf :: Word32 -> a -> List a -> List a
consIf 0 x xs = xs
consIf n x xs = Cons x xs

xconcat :: List a -> List a -> List a
xconcat Nil b = b
xconcat (Cons x xs) b = Cons x (xconcat xs b)

quicksort :: List Word32 -> List Word32
quicksort Nil         = Nil
quicksort (Cons x xs) =
  let min = xfilter (\n -> if n < x then 1 else 0) xs
      max = xfilter (\n -> if n > x then 1 else 0) xs
  in xconcat (quicksort min) (Cons x (quicksort max))

xrandoms :: Word32 -> Word32 -> List Word32
xrandoms seed 0    = Nil
xrandoms seed size = Cons seed (xrandoms (seed * 1664525 + 1013904223) (size - 1))

main :: IO ()
main = print $ (xsum (quicksort (xrandoms 0 400000)))
