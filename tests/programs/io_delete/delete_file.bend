test-io = 1

IO/FS/delete_file path =
  (call "DELETE_FILE" path)

main =
  use path = "./tests/programs/io_remove/files/remove_file.txt"
  with IO {
    ask s = (IO/FS/delete_file path)
    (wrap s)
  }
