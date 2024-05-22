type FFI:
  Done { magic, expr }
  Call { magic, func, argm, cont }

def FFI/MAGIC:
  return (0xD0CA11, 0xFF1FF1)

def FFI/wrap(x):
  return FFI/Done(FFI/MAGIC, x)

def FFI/bind(a, b):
  match a:
    case FFI/Done:
      return b(a.expr)
    case FFI/Call:
      return FFI/Call(FFI/MAGIC, a.func, a.argm, lambda x: FFI/bind(a.cont(x), b))

def call(func, argm):
  return FFI/Call(FFI/MAGIC, func, argm, lambda x: FFI/Done(FFI/MAGIC, x))

def main:
  do FFI:
    x <- call("foo", 42)
    y <- call("bar", 64)
    z <- call("tic", 12)
    w <- call("tac", 34)
    return 80
