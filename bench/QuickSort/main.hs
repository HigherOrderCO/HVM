import Data.Bits
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

-- Initial pivot
pivot = 2147483648

-- Parallel QuickSort
qsort p s Nil          = Empty
qsort p s (Cons x Nil) = Single x
qsort p s (Cons x xs)  =
  split p s (Cons x xs) Nil Nil

-- Splits list in two partitions
split p s Nil min max =
  let s'   = shiftR s 1
      min' = qsort (p - s') s' min
      max' = qsort (p + s') s' max
  in  Concat min' max'
split p s (Cons x xs) min max =
  place p s (p < x) x xs min max

-- Moves element to its partition
place p s False x xs min max =
  split p s xs (Cons x min) max
place p s True  x xs min max =
  split p s xs min (Cons x max)

-- Sorts and sums n random numbers
main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Word32
  let list = randoms 1 (100000 * n)
  print $ sun $ qsort pivot pivot $ list 




