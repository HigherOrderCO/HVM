#{
  Creates the batata directory and then removes it.
#}

test-io = 1

mkdir path =
  (call "MKDIR" path)

rm_all path =
  (call "RM_ALL" path)

main =
  let path = "./batata"
  with IO {
    ask * = (mkdir path)
    ask s = (rm_all path)
    (wrap s)
  }
