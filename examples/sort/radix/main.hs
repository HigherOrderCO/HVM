module Main where

import Data.Word
import Data.Bits
import System.Environment

data NTree = Null | Leaf Word64 | Node NTree NTree deriving Show
data STree = Free | Used        | Both STree STree deriving Show

sort :: NTree -> NTree
sort t = stree_back (stree_make t)

stree_merge :: STree -> STree -> STree
stree_merge Free       Free       = Free
stree_merge Free       Used       = Used
stree_merge Used       Free       = Used
stree_merge Used       Used       = Used
stree_merge Free       (Both c d) = (Both c d)
stree_merge (Both a b) Free       = (Both a b)
stree_merge (Both a b) (Both c d) = (Both (stree_merge a c) (stree_merge b d))

stree_make :: NTree -> STree
stree_make Null       = Free
stree_make (Leaf a)   = stree_word a
stree_make (Node a b) = stree_merge (stree_make a) (stree_make b)

stree_back :: STree -> NTree
stree_back t = stree_back_go 0 t where
  stree_back_go :: Word64 -> STree -> NTree
  stree_back_go x Free       = Null
  stree_back_go x Used       = Leaf x
  stree_back_go x (Both a b) =
    let x' = x * 2
        y' = x' + 1
        a' = stree_back_go x' a
        b' = stree_back_go y' b
    in Node a' b'

stree_word :: Word64 -> STree
stree_word n =
  let r0 = Used
      r1 = u60_swap (n .&. 1) r0 Free
      r2 = u60_swap (n .&. 2) r1 Free
      r3 = u60_swap (n .&. 4) r2 Free
      r4 = u60_swap (n .&. 8) r3 Free
      r5 = u60_swap (n .&. 16) r4 Free
      r6 = u60_swap (n .&. 32) r5 Free
      r7 = u60_swap (n .&. 64) r6 Free
      r8 = u60_swap (n .&. 128) r7 Free
      r9 = u60_swap (n .&. 256) r8 Free
      rA = u60_swap (n .&. 512) r9 Free
      rB = u60_swap (n .&. 1024) rA Free
      rC = u60_swap (n .&. 2048) rB Free
      rD = u60_swap (n .&. 4096) rC Free
      rE = u60_swap (n .&. 8192) rD Free
      rF = u60_swap (n .&. 16384) rE Free
      rG = u60_swap (n .&. 32768) rF Free
      rH = u60_swap (n .&. 65536) rG Free
      rI = u60_swap (n .&. 131072) rH Free
      rJ = u60_swap (n .&. 262144) rI Free
      rK = u60_swap (n .&. 524288) rJ Free
      rL = u60_swap (n .&. 1048576) rK Free
      rM = u60_swap (n .&. 2097152) rL Free
      rN = u60_swap (n .&. 4194304) rM Free
      rO = u60_swap (n .&. 8388608) rN Free
  in rO

u60_swap :: Word64 -> STree -> STree -> STree
u60_swap 0 a b = Both a b
u60_swap n a b = Both b a

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
        y' = x' + 1
        n' = n - 1
    in Node (gen_go n' x') (gen_go n' y')

main :: IO ()
main = do
  n <- read . head <$> getArgs :: IO Word64
  print $ sum' (sort (reverse' (gen n)))
