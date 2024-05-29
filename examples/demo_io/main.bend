# TODO: update tests to handle IO
test-skip = 1

type IO_T:
  Done { magic, expr }
  Call { magic, func, argm, cont }

def IO_T/MAGIC:
  return (0xD0CA11, 0xFF1FF1)

def IO_T/wrap(x):
  return IO_T/Done(IO_T/MAGIC, x)

def IO_T/bind(a, b):
  match a:
    case IO_T/Done:
      return b(a.expr)
    case IO_T/Call:
      return IO_T/Call(IO_T/MAGIC, a.func, a.argm, lambda x: IO_T/bind(a.cont(x), b))

def call_io(func, argm):
  return IO_T/Call(IO_T/MAGIC, func, argm, lambda x: IO_T/Done(IO_T/MAGIC, x))

def main:
  do IO_T:
    * <- call_io("PUT_TEXT", "what is your name?\n")
    n <- call_io("GET_LINE", "")
    c <- call_io("PUT_LINE", "pick a number 0-9?\n")
    * <- call_io("PUT_TEXT", "hello, '")
    * <- call_io("PUT_TEXT", n)
    * <- call_io("PUT_TEXT", "', you picked '")
    * <- call_io("PUT_TEXT", c)
    * <- call_io("PUT_TEXT", "'! Nice choice!")
    return 42
