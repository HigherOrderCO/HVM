test-io = 1

def unwrap(res):
  match res:
    case Result/Ok:
      return res.val
    case Result/Err:
      return res.val

def open():
  return call("OPEN", ("./LICENSE", "r"))

def read(f):
  return call("READ", (f, 47))

def print(bytes):
  with IO:
    * <- call("WRITE", (1, bytes))
    * <- call("WRITE", (1, "\n"))

    return wrap(*)

def close(f):
  return call("CLOSE", f)

def main():
  with IO:
    f <- open()
    f = unwrap(f)
    bytes <- read(f)
    bytes = unwrap(bytes)
    * <- print(bytes)
    res <- close(f)

    return wrap(res)
