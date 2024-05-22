def loop(n):
  switch n:
    case 0:
      return 0
    case _:
      return loop(n-1)

def fun(n):
  switch n:
    case 0:
      return loop(0x10000)
    case _:
      return fun(n-1) + fun(n-1)

def main:
  return fun(12)
