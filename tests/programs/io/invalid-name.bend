test-io = 1

def main():
  with IO:
    f <- call("INVALID-NAME", ("./README.md", "r"))

    return wrap(f)
