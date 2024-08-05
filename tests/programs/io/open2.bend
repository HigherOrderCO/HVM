def main():
  with IO:
    f <- call("OPEN", ("fake-file", "r"))

    return wrap(f)
