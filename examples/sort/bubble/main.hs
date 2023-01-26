module Main where

import Data.Word
import System.Environment

data List = Nil | Cons Word64 List

-- Sorts a list
sort' :: List -> List
sort' Nil         = Nil
sort' (Cons x xs) = insert x (sort' xs)

insert :: Word64 -> List -> List
insert v Nil         = Cons v Nil
insert v (Cons x xs) = swapGT (if v > x then 1 else 0) v x xs

swapGT :: Word64 -> Word64 -> Word64 -> List -> List
swapGT 0 v x xs = Cons v (Cons x xs)
swapGT 1 v x xs = Cons x $! insert v xs

-- Generates a random list
rnd' :: Word64 -> Word64 -> List
rnd' 0 s = Nil
rnd' n s = Cons s (rnd' (n - 1) ((s * 1664525 + 1013904223) `mod` 4294967296))

-- Sums a list
sum' :: List -> Word64
sum' Nil         = 0
sum' (Cons x xs) = x + sum' xs

main :: IO ()
main = do
  n <- read . head <$> getArgs :: IO Word64
  print $ sum' (sort' (rnd' (n * 1000) 1))
