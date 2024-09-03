#{
  Uses the rm function to remove an empty directory.
#}

test-io = 1

mkdir path =
  (call "MKDIR" path)

rm path =
  (call "RM" path)

main =
  with IO {
    ask * = (mkdir "temp")
    ask s = (rm "temp")
    (wrap s)
  }
