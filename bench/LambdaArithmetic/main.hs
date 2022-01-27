{-# LANGUAGE RankNTypes #-}

import Data.Word
import System.Environment

-- The Scott-Encoded Bits type
newtype Bits = Bits { get :: forall a. a -> (Bits -> a) -> (Bits -> a) -> a }
end  = Bits (\e -> \o -> \i -> e)
b0 p = Bits (\e -> \o -> \i -> o p)
b1 p = Bits (\e -> \o -> \i -> i p)

-- Applies `f` `xs` times to `x`
app :: Bits -> (a -> a) -> a -> a
app xs f x =
  let e = \f -> \x -> x
      o = \p -> \f -> \x -> app p (\k -> f (f k)) x
      i = \p -> \f -> \x -> app p (\k -> f (f k)) (f x)
  in get xs e o i f x

-- Increments a Bits by 1
inc :: Bits -> Bits
inc xs = Bits $ \ex -> \ox -> \ix ->
  let e = ex
      o = ix
      i = \p -> ox (inc p)
  in get xs e o i

-- Adds two Bits
add :: Bits -> Bits -> Bits
add xs ys = app xs (\x -> inc x) ys

-- Muls two Bits
mul :: Bits -> Bits -> Bits
mul xs ys = 
  let e = end
      o = \p -> b0 (mul p ys)
      i = \p -> add ys (b1 (mul p ys))
  in get xs e o i

-- Converts a Bits to an U32
toU32 :: Bits -> Word32
toU32 ys =
  let e = 0
      o = \p -> toU32 p * 2 + 0
      i = \p -> toU32 p * 2 + 1
  in get ys e o i

-- Converts an U32 to a Bits
fromU32 :: Word32 -> Word32 -> Bits
fromU32 0 i = end
fromU32 s i = fromU32Put (s - 1) (i `mod` 2) (i `div` 2) where
  fromU32Put s 0 i = b0 (fromU32 s i)
  fromU32Put s 1 i = b1 (fromU32 s i)

-- Squares (n * 100k)
main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Word32
  let a = fromU32 32 (100000 * n)
  let b = fromU32 32 (100000 * n)
  print $ toU32 (mul a b)
