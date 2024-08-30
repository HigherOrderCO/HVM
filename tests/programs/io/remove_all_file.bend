#{
  Calls the rm_all function with a file path as argument.
#}

test-io = 1

rm_all path =
  (call "RM_ALL" path)

main =
  let temp = "./temp.txt"
  with IO {
    ask * = (IO/FS/write_file temp (String/encode_utf8 "Contents"))
    ask s = (rm_all temp)
    (wrap s)
  }
