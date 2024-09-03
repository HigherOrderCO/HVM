#{
  Tries to remove a non existing file.
#}

test-io = 1

rm path =
  (call "RM" path)

main =
  use path = "./non_existing.txt"
  with IO {
    ask s = (rm path)
    (wrap s)
  }
