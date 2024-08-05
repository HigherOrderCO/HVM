def unwrap(res):
  match res:
    case Result/Ok:
      return res.val
    case Result/Err:
      return res.val

def main():
  with IO:
    f <- call("OPEN", ("./README.md", "r"))
    f = unwrap(f)
    bytes <- call("READ", (f, 30))
    bytes = unwrap(bytes)
    * <- call("WRITE", (1, bytes))
    * <- call("WRITE", (1, "\n"))
    res <- call("CLOSE", f)

    return wrap(res)
