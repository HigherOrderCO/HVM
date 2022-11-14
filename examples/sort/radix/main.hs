import Data.Word
import System.Environment

data NTree = Null | Leaf Word64 | Node NTree NTree deriving Show
data STree = Free | Used        | Both STree STree deriving Show

sort :: NTree -> NTree
sort t = stree_to_ntree (stree_from_ntree t)

stree_from_ntree :: NTree -> STree
stree_from_ntree Null       = Free
stree_from_ntree (Leaf a)   = stree_from_num a
stree_from_ntree (Node a b) = stree_merge (stree_from_ntree a) (stree_from_ntree b)

stree_from_num :: Word64 -> STree
stree_from_num n = stree_from_num_go 24 n Used where
  stree_from_num_go 0 n r = r
  stree_from_num_go s n r = stree_from_num_go (s - 1) (n `div` 2) (stree_from_num_mk (n `mod` 2) r)
  stree_from_num_mk 0 r   = Both r Free
  stree_from_num_mk 1 r   = Both Free r

stree_merge :: STree -> STree -> STree
stree_merge Free       Free       = Free
stree_merge Free       Used       = Used
stree_merge Used       Free       = Used
stree_merge Used       Used       = Used
stree_merge Free       (Both c d) = (Both c d)
stree_merge (Both a b) Free       = (Both a b)
stree_merge (Both a b) (Both c d) = (Both (stree_merge a c) (stree_merge b d))

stree_to_ntree :: STree -> NTree
stree_to_ntree t = stree_to_ntree_go 0 t where
  stree_to_ntree_go :: Word64 -> STree -> NTree
  stree_to_ntree_go x Free       = Null
  stree_to_ntree_go x Used       = Leaf x
  stree_to_ntree_go x (Both a b) =
    let x' = (x * 2)
        a' = stree_to_ntree_go x'       a
        b' = stree_to_ntree_go (x' + 1) b
    in Node a' b'

reverse' :: NTree -> NTree
reverse' Null       = Null
reverse' (Leaf a)   = Leaf a
reverse' (Node a b) = Node (reverse' b) (reverse' a)

sum' :: NTree -> Word64
sum' Null       = 0
sum' (Leaf x)   = x
sum' (Node a b) = sum' a + sum' b

gen :: Word64 -> NTree
gen n = gen_go n 0 where
  gen_go :: Word64 -> Word64 -> NTree
  gen_go 0 x = Leaf x
  gen_go n x =
    let x' = x * 2
        n' = n - 1
    in Node (gen_go n' x') (gen_go n' (x' + 1))

main :: IO ()
main = do
  n <- read . head <$> getArgs :: IO Word64
  print $ sum' (sort (reverse' (gen n)))
