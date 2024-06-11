test-io = 1

type IO_T:
  Done { magic, expr }
  Call { magic, func, argm, cont }

def IO_T/MAGIC:
  return (0xD0CA11, 0xFF1FF1)

def IO_T/bind(a, b):
  match a:
    case IO_T/Done:
      return (undefer(b))(a.expr)
    case IO_T/Call:
      return IO_T/Call(IO_T/MAGIC, a.func, a.argm, lambda x: IO_T/bind(a.cont(x), b))

def call_io(func, argm):
  return IO_T/Call(IO_T/MAGIC, func, argm, lambda x: IO_T/Done(IO_T/MAGIC, x))

def read_input():
  with IO_T:
    * <- call_io("WRITE", (1, "What is your name?\n"))
    return call_io("READ", (0, 10))

def write_to_file():
  with IO_T:
    fp <- call_io("OPEN", ("testing.txt", "w"))
    input <- read_input()
    * <- call_io("WRITE", (fp, input))
    * <- call_io("WRITE", (fp, "\n"))

    return call_io("CLOSE", fp)

def read_from_file():
  with IO_T:
    fp <- call_io("OPEN", ("testing.txt", "r"))
    bytes <- call_io("READ", (fp, 5))
    * <- call_io("WRITE", (1, bytes))
    * <- call_io("SEEK", (fp, 2, 0))
    bytes <- call_io("READ", (fp, 5))
    * <- call_io("WRITE", (1, bytes))

    return call_io("CLOSE", fp)

def main():
  with IO_T:
    * <- write_to_file()
    * <- read_from_file()
    * <- call_io("WRITE", (1, "\n"))

    return 42
