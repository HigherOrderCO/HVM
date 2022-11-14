import Data.Word
import System.Environment

data Tree = Node Tree Tree | Leaf Word64 deriving Show

-- Atomic Swap
swap' :: Word64 -> Word64 -> Tree -> Tree -> Tree
swap' 0 0 a b = Node a b
swap' 0 1 a b = Node b a
swap' 1 0 a b = Node b a
swap' 1 1 a b = Node a b

-- Swaps distant values in parallel; corresponds to a Red Box
warp' :: Word64 -> Tree -> Tree -> Tree
warp' s (Leaf a)   (Leaf b)   = swap' (if a > b then 1 else 0) s (Leaf a) (Leaf b)
warp' s (Node a b) (Node c d) = join' (warp' s a c) (warp' s b d)

-- Rebuilds the warped tree in the original order
join' :: Tree -> Tree -> Tree
join' (Node a b) (Node c d) = Node (Node a c) (Node b d)

-- Recursively warps each sub-tree; corresponds to a Blue/Green Box
flow' :: Word64 -> Tree -> Tree
flow' s (Leaf a)   = Leaf a
flow' s (Node a b) = down' s (warp' s a b)

-- Propagates Flow downwards
down' :: Word64 -> Tree -> Tree
down' s (Leaf a)   = Leaf a
down' s (Node a b) = Node (flow' s a) (flow' s b)

-- Bitonic Sort
sort' :: Word64 -> Tree -> Tree
sort' s (Leaf a)   = Leaf a
sort' s (Node a b) = flow' s (Node (sort' 0 a) (sort' 1 b))

-- Generates a tree of depth `n`
gen' :: Word64 -> Word64 -> Tree
gen' 0 x = Leaf x
gen' n x = Node (gen' (n - 1) (x * 2)) (gen' (n - 1) (x * 2 + 1))

-- Reverses a tree
rev' :: Tree -> Tree
rev' (Leaf a)   = Leaf a
rev' (Node a b) = Node (rev' b) (rev' a)

-- Sums a tree
sum' :: Tree -> Word64
sum' (Leaf a)   = a
sum' (Node a b) = sum' a + sum' b

main :: IO ()
main = do
  n <- read . head <$> getArgs :: IO Word64
  print $ sum' (sort' 0 (rev' (gen' n 0)))
