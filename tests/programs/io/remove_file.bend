#{
  Creates a temporary file and then removes it.
#}

test-io = 1

rm path =
  (call "RM" path)

main =
  let path = "./temp.txt"
  with IO {
    ask * = (IO/FS/write_file path (String/encode_utf8 "Contents"))
    ask s = (rm path)
    (wrap s)
  }
