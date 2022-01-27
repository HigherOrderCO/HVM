import Data.Word

data Tree = Node Tree Tree | Leaf Word32

-- Creates a tree with 2^n elements
gen :: Word32 -> Tree
gen 0 = Leaf 1
gen n = Node (gen(n - 1)) (gen(n - 1))

-- Adds all elements of a tree
sun :: Tree -> Word32
sun (Leaf x)   = 1
sun (Node a b) = sun a + sun b

-- Performs 2^30 additions
main = print $ sun (gen 30)
