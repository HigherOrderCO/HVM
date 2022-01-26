n = 28 :: Int

data Tree = Bin Tree Tree | Tip

-- Generates a binary tree
gen :: Int -> Tree
gen 0 = Tip
gen n = Bin (gen (n - 1)) (gen (n - 1))

-- Sums its elements
sun :: Tree -> Int
sun Tip       = 1
sun (Bin a b) = sun a + sun b

main = print $ sun (gen 28)
