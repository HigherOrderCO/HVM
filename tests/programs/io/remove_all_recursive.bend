#{
  Creates the following tree structure and then removes A and its children.
  A
  |-- a.txt
  |-- AA
  |   `-- aa.txt
  `-- AB
      `-- ab.txt
#}

test-io = 1

mkdir path =
  (call "MKDIR" path)

rm_all path =
  (call "RM_ALL" path)

test =
  with IO {
    ask * = (mkdir "A")
    ask * = (mkdir "A/AA")
    ask * = (mkdir "A/AB")
    ask * = (IO/FS/write_file "A/a.txt" (String/encode_utf8 "a"))
    ask * = (IO/FS/write_file "A/AA/aa.txt" (String/encode_utf8 "aa"))
    ask * = (IO/FS/write_file "A/AB/ab.txt" (String/encode_utf8 "ab"))
    ask s = (rm_all "./A")
    (wrap s)
  }

main =
  with IO {
    ask res = (test)
    (wrap res)
  }
