module Main where

import Data.Word
import Data.Bits
import System.Environment

data Arr = Null | Leaf !Word64 | Node Arr Arr deriving Show
data Map = Free | Used         | Both !Map !Map deriving Show

sort :: Arr -> Arr
sort t = toArr 0 (toMap t)

toMap :: Arr -> Map
toMap Null       = Free
toMap (Leaf a)   = radix a
toMap (Node a b) = merge (toMap a) (toMap b)

toArr :: Word64 -> Map -> Arr
toArr x Free       = Null
toArr x Used       = Leaf x
toArr x (Both a b) =
  let a' = toArr (x * 2 + 0) a
      b' = toArr (x * 2 + 1) b
  in Node a' b'

merge :: Map -> Map -> Map
merge Free       Free       = Free
merge Free       Used       = Used
merge Used       Free       = Used
merge Used       Used       = Used
merge Free       (Both c d) = (Both c d)
merge (Both a b) Free       = (Both a b)
merge (Both a b) (Both c d) = (Both (merge a c) (merge b d))

radix :: Word64 -> Map
radix n =
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

u60_swap :: Word64 -> Map -> Map -> Map
u60_swap 0 a b = Both a b
u60_swap n a b = Both b a

reverse' :: Arr -> Arr
reverse' Null       = Null
reverse' (Leaf a)   = Leaf a
reverse' (Node a b) = Node (reverse' b) (reverse' a)

sum' :: Arr -> Word64
sum' Null       = 0
sum' (Leaf x)   = x
sum' (Node a b) = sum' a + sum' b

gen :: Word64 -> Arr
gen n = gen_go n 0 where
  gen_go :: Word64 -> Word64 -> Arr
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
