#{
  Tries to delete a non existing file.
#}

test-io = 1

IO/FS/delete_file path =
  (call "DELETE_FILE" path)

main =
  use path = "./non_existing.txt"
  with IO {
    ask s = (IO/FS/delete_file path)
    (wrap s)
  }
