#{
  Creates the following tree structure and then deletes A and its children.
  A
  |-- a.txt
  |-- AA
  |   `-- aa.txt
  `-- AB
      `-- ab.txt
#}

test-io = 1

IO/FS/mkdir path =
  (call "MKDIR" path)

IO/FS/delete_directory path recursive =
  (call "DELETE_DIRECTORY" (path, recursive))

True = 1

main =
  with IO {
    ask * = (IO/FS/mkdir "A")
    ask * = (IO/FS/mkdir "A/AA")
    ask * = (IO/FS/mkdir "A/AB")
    ask * = (IO/FS/write_file "A/a.txt" (String/encode_utf8 "a"))
    ask * = (IO/FS/write_file "A/AA/aa.txt" (String/encode_utf8 "aa"))
    ask * = (IO/FS/write_file "A/AB/ab.txt" (String/encode_utf8 "ab"))

    ask s = (IO/FS/delete_directory "A" True)
    (wrap s)
  }
