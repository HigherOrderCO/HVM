data Tree = Bin Tree Tree | Tip

gen :: Int -> Tree
gen 0 = Tip
gen n = Bin (gen (n - 1)) (gen (n - 1))

sun :: Tree -> Int
sun Tip       = 1
sun (Bin a b) = sun a + sun b

main = print $ sun (gen 30)
