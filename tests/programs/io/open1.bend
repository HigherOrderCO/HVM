def main():
  with IO:
    f <- call("OPEN", ("./README.md", "r"))

    return wrap(f)
