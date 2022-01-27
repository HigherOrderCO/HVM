import Data.Word
import System.Environment

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
qsort :: List Word32 -> Tree Word32
qsort Nil          = Empty
qsort (Cons x Nil) = Single x
qsort (Cons p xs)  = split p xs Nil Nil where
  split p Nil         min max = Concat (qsort min) (qsort max)
  split p (Cons x xs) min max = place p (p < x) x xs min max
  place p False x xs  min max = split p xs (Cons x min) max
  place p True  x xs  min max = split p xs min (Cons x max)

-- Sorts and sums n random numbers
main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Word32
  print $ sun $ qsort $ randoms 1 n
