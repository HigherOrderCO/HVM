#{
  Calls the delete_directory function with a file path as argument.
#}

test-io = 1

IO/FS/delete_directory path recursive =
  (call "DELETE_DIRECTORY" (path, recursive))

False = 0

main =
  with IO {
    ask s = (IO/FS/delete_directory "./delete_dir_file.bend" False)
    (wrap s)
  }
