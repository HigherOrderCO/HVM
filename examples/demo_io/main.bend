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
    x <- call_io("PUT_TEXT", "hello\n")
    y <- call_io("PUT_TEXT", "world\n")
    return 42
