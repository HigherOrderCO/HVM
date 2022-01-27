import Data.Word

data Tree = Node Tree Tree | Leaf Word32

-- Generates a binary tree
gen :: Word32 -> Word32 -> Tree
gen 0 x = Leaf x
gen n x = Node (gen (n - 1) x) (gen (n - 1) x)

-- Sums its elements
sun :: Tree -> Word32
sun (Leaf x)   = 1
sun (Node a b) = sun a + sun b

-- Performs 2^30 sums
main = print $ sun (gen 30 1)
