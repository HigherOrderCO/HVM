test-io = 1

def main():
  with IO:
    # calling open with an unexpected type of arg
    f <- call("OPEN", 123)

    return wrap(f)
