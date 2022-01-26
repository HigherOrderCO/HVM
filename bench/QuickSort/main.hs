import Data.Word

data List a = Nil | Cons a (List a) deriving Show
data Tree a = Empty | Single a | Concat (Tree a) (Tree a) deriving Show

-- Generates a random list
randoms :: Word32 -> Word32 -> List Word32
randoms seed 0    = Nil
randoms seed size = Cons seed (randoms (seed * 1664525 + 1013904223) (size - 1))

-- Sums all elements in a concatenation tree
sun :: Tree Word32 -> Word32
sun Empty        = 0
sun (Single a)   = a
sun (Concat a b) = sun a + sun b

-- Parallel QuickSort
quicksort :: List Word32 -> Tree Word32
quicksort Nil                    = Empty
quicksort (Cons x Nil)           = Single x
quicksort l@(Cons p (Cons x xs)) = split p l Nil Nil where
  split p Nil         min max    = Concat (quicksort min) (quicksort max)
  split p (Cons x xs) min max    = place p (p < x) x xs min max
  place p False x xs  min max    = split p xs (Cons x min) max
  place p True  x xs  min max    = split p xs min (Cons x max)

main :: IO ()
main = do
  print $ sun $ quicksort $ randoms 1 10000000
