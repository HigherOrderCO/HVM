#{
  Uses the delete_file function to delete an empty directory.
#}

test-io = 1

IO/FS/mkdir path =
  (call "MKDIR" path)

IO/FS/delete_file path =
  (call "DELETE_FILE" path)

main =
  with IO {
    ask * = (IO/FS/mkdir "temp")
    ask s = (IO/FS/delete_file "temp")
    (wrap s)
  }
