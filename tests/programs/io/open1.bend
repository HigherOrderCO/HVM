test-io = 1

def main():
  with IO:
    f <- call("OPEN", ("./LICENSE", "r"))

    return wrap(f)
