#{
  Creates the batata directory and then deletes it.
#}

test-io = 1

IO/FS/mkdir path =
  (call "MKDIR" path)

IO/FS/delete_directory path recursive =
  (call "DELETE_DIRECTORY" (path, recursive))

False = 0

main =
  let path = "./batata"
  with IO {
    ask * = (IO/FS/mkdir path)
    ask s = (IO/FS/delete_directory path False)
    (wrap s)
  }
