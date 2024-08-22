#{
  Creates a temporary file and then deletes it.
#}

test-io = 1

IO/FS/delete_file path =
  (call "DELETE_FILE" path)

main =
  let path = "./temp.txt"
  with IO {
    ask * = (IO/FS/write_file path (String/encode_utf8 "Contents"))
    ask s = (IO/FS/delete_file path)
    (wrap s)
  }
