{-# LANGUAGE RankNTypes #-}
module Main where

import Data.Word
import System.Environment

data List = Nil | Cons Word64 List deriving Show
data Tree = Leaf | Node Tree Word64 Tree deriving Show
type Pair = forall a . (List -> List -> a) -> a

-- Parallel QuickSort
sort' :: List -> Tree
sort' Nil         = Leaf
sort' (Cons x xs) =
  part x xs $ \min max ->
    let lft = sort' min
        rgt = sort' max
    in Node lft x rgt
  where

    -- Partitions a list in two halves, less-than-p and greater-than-p
    part :: Word64 -> List -> Pair
    part p Nil         = \t -> t Nil Nil
    part p (Cons x xs) = push (if x > p then 1 else 0) x (part p xs)

    -- Pushes a value to the first or second list of a pair
    push :: Word64 -> Word64 -> Pair -> Pair
    push 0 x pair = pair $ \min max p -> p (Cons x min) max
    push 1 x pair = pair $ \min max p -> p min (Cons x max)
  
-- Generates a random list
rnd' :: Word64 -> Word64 -> List
rnd' 0 s = Nil
rnd' n s = Cons s (rnd' (n - 1) ((s * 1664525 + 1013904223) `mod` 4294967296))

-- Sums a list
sum' :: Tree -> Word64
sum' Leaf         = 0
sum' (Node l n r) = n + sum' l + sum' r

main :: IO ()
main = do
  n <- read . head <$> getArgs :: IO Word64
  print $ (sum' (sort' (rnd' (2 ^ n) 1)))
  

-- - Sorts and sums n random numbers
-- (Main n) = (Sum (Sort (Rnd (<< 1 20) 1)))
