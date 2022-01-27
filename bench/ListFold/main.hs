import Data.Word
import System.Environment

data List a = Nil | Cons a (List a)

-- Folds over a list
fold :: List a -> (a -> r -> r) -> r -> r
fold Nil         c n = n
fold (Cons x xs) c n =
  c x (fold xs c n)

-- A list from 0 to n
range :: Word32 -> List Word32 -> List Word32
range 0 xs = xs
range n xs =
  let m = n - 1
  in range m (Cons m xs)

-- Sums a big list with fold
main :: IO ()
main = do
  n <- read.head <$> getArgs :: IO Word32
  let size = 1000000 * n
  let list = range size Nil
  print $ fold list (+) 0
