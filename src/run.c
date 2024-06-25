#include <dlfcn.h>
#include "hvm.c"

// Readback: λ-Encoded Ctr
typedef struct Ctr {
  u32  tag;
  u32  args_len;
  Port args_buf[16];
} Ctr;

// Readback: Tuples
typedef struct Tup {
  u32  elem_len;
  Port elem_buf[8];
} Tup;

// Readback: λ-Encoded Str (UTF-32), null-terminated
// FIXME: this is actually ASCII :|
typedef struct Str {
  u32  len;
  char *buf;
} Str;

// Readback: λ-Encoded list of bytes
typedef struct Bytes {
  u32  len;
  char *buf;
} Bytes;

// IO Magic Number
#define IO_MAGIC_0 0xD0CA11
#define IO_MAGIC_1 0xFF1FF1

// IO Tags
#define IO_DONE 0
#define IO_CALL 1

// List Type
#define LIST_NIL  0
#define LIST_CONS 1

// Readback
// --------

// Reads back a λ-Encoded constructor from device to host.
// Encoding: λt ((((t TAG) arg0) arg1) ...)
Ctr readback_ctr(Net* net, Book* book, Port port) {
  Ctr ctr;
  ctr.tag = -1;
  ctr.args_len = 0;

  // Loads root lambda
  Port lam_port = expand(net, book, port);
  if (get_tag(lam_port) != CON) return ctr;
  Pair lam_node = node_load(net, get_val(lam_port));

  // Loads first application
  Port app_port = expand(net, book, get_fst(lam_node));
  if (get_tag(app_port) != CON) return ctr;
  Pair app_node = node_load(net, get_val(app_port));

  // Loads first argument (as the tag)
  Port arg_port = expand(net, book, get_fst(app_node));
  if (get_tag(arg_port) != NUM) return ctr;
  ctr.tag = get_u24(get_val(arg_port));

  // Loads remaining arguments
  while (TRUE) {
    app_port = expand(net, book, get_snd(app_node));
    if (get_tag(app_port) != CON) break;
    app_node = node_load(net, get_val(app_port));
    arg_port = expand(net, book, get_fst(app_node));
    ctr.args_buf[ctr.args_len++] = arg_port;
  }

  return ctr;
}

// Reads back a tuple of at most `size` elements. Tuples are
// (right-nested con nodes) (CON 1 (CON 2 (CON 3 (...))))
// The provided `port` should be `expanded` before calling.
extern Tup readback_tup(Net* net, Book* book, Port port, u32 size) {
  Tup tup;
  tup.elem_len = 0;

  // Loads remaining arguments
  while (get_tag(port) == CON && (tup.elem_len + 1 < size)) {
    Pair node = node_load(net, get_val(port));
    tup.elem_buf[tup.elem_len++] = expand(net, book, get_fst(node));

    port = expand(net, book, get_snd(node));
  }

  tup.elem_buf[tup.elem_len++] = port;

  return tup;
}

// Converts a Port into a list of bytes.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
Bytes readback_bytes(Net* net, Book* book, Port port) {
  Bytes bytes;
  u32 capacity = 256;
  bytes.buf = (char*) malloc(sizeof(char) * capacity);
  bytes.len = 0;

  // Readback loop
  while (TRUE) {
    // Normalizes the net
    normalize(net, book);

    // Reads the λ-Encoded Ctr
    Ctr ctr = readback_ctr(net, book, peek(net, port));

    // Reads string layer
    switch (ctr.tag) {
      case LIST_NIL: {
        break;
      }
      case LIST_CONS: {
        if (ctr.args_len != 2) break;
        if (get_tag(ctr.args_buf[0]) != NUM) break;

        if (bytes.len == capacity - 1) {
          capacity *= 2;
          bytes.buf = realloc(bytes.buf, capacity);
        }

        bytes.buf[bytes.len++] = get_u24(get_val(ctr.args_buf[0]));
        boot_redex(net, new_pair(ctr.args_buf[1], ROOT));
        port = ROOT;
        continue;
      }
    }
    break;
  }

  return bytes;
}

// Converts a Port into a UTF-32 (truncated to 24 bits) null-terminated string.
// Since unicode scalars can fit in 21 bits, HVM's u24
// integers can contain any unicode scalar value.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
Str readback_str(Net* net, Book* book, Port port) {
  // readback_bytes is guaranteed to return a buffer with a capacity of at least one more
  // than the number of bytes read, so we can null-terminate it.
  Bytes bytes = readback_bytes(net, book, port);

  Str str;
  str.len = bytes.len;
  str.buf = bytes.buf;
  str.buf[str.len] = 0;

  return str;
}

/// Returns a λ-Encoded Ctr for a NIL: λt (t NIL)
/// A previous call to `get_resources(tm, 0, 2, 1)` is required.
Port inject_nil(Net* net) {
  u32 v1 = tm[0]->vloc[0];

  u32 n1 = tm[0]->nloc[0];
  u32 n2 = tm[0]->nloc[1];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(new_port(NUM, new_u24(LIST_NIL)), var));
  node_create(net, n2, new_pair(new_port(CON, n1), var));

  return new_port(CON, n2);
}

/// Returns a λ-Encoded Ctr for a CONS: λt (((t CONS) head) tail)
/// A previous call to `get_resources(tm, 0, 4, 1)` is required.
Port inject_cons(Net* net, Port head, Port tail) {
  u32 v1 = tm[0]->vloc[0];

  u32 n1 = tm[0]->nloc[0];
  u32 n2 = tm[0]->nloc[1];
  u32 n3 = tm[0]->nloc[2];
  u32 n4 = tm[0]->nloc[3];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(tail, var));
  node_create(net, n2, new_pair(head, new_port(CON, n1)));
  node_create(net, n3, new_pair(new_port(NUM, new_u24(LIST_CONS)), new_port(CON, n2)));
  node_create(net, n4, new_pair(new_port(CON, n3), var));

  return new_port(CON, n4);
}

// Converts a list of bytes to a Port.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
Port inject_bytes(Net* net, Bytes *bytes) {
  // Allocate all resources up front:
  // - NIL needs  2 nodes & 1 var
  // - CONS needs 4 nodes & 1 var
  u32 len = bytes->len;
  if (!get_resources(net, tm[0], 0, 2, 1)) {
    fprintf(stderr, "inject_bytes: failed to get resources\n");
    return new_port(ERA, 0);
  }
  Port port = inject_nil(net);

  // TODO: batch-allocate these (within the limits of TM)
  for (u32 i = 0; i < len; i++) {
    if (!get_resources(net, tm[0], 0, 4, 1)) {
      fprintf(stderr, "inject_bytes: failed to get resources\n");
      return new_port(ERA, 0);
    }
    Port byte = new_port(NUM, new_u24(bytes->buf[len - i - 1]));
    port = inject_cons(net, byte, port);
  }

  return port;
}

// Primitive IO Fns
// -----------------

// Open file pointers. Indices into this array
// are used as "file descriptors".
// Indices 0 1 and 2 are reserved.
// - 0 -> stdin
// - 1 -> stdout
// - 2 -> stderr
static FILE* FILE_POINTERS[256];

// Open dylibs handles. Indices into this array
// are used as opaque loadedd object "handles".
static void* DYLIBS[256];

// Converts a NUM port (file descriptor) to file pointer.
FILE* readback_file(Port port) {
  if (get_tag(port) != NUM) {
    fprintf(stderr, "non-num where file descriptor was expected: %i\n", get_tag(port));
    return NULL;
  }

  u32 idx = get_u24(get_val(port));

  if (idx == 0) return stdin;
  if (idx == 1) return stdout;
  if (idx == 2) return stderr;

  FILE* fp = FILE_POINTERS[idx];
  if (fp == NULL) {
    fprintf(stderr, "invalid file descriptor\n");
    return NULL;
  }

  return fp;
}

// Converts a NUM port (dylib handle) to an opaque dylib object.
void* readback_dylib(Port port) {
  if (get_tag(port) != NUM) {
    fprintf(stderr, "non-num where dylib handle was expected: %i\n", get_tag(port));
    return NULL;
  }

  u32 idx = get_u24(get_val(port));

  void* dl = DYLIBS[idx];
  if (dl == NULL) {
    fprintf(stderr, "invalid dylib handle\n");
    return NULL;
  }

  return dl;
}

// Reads from a file a specified number of bytes.
// `argm` is a tuple of (file_descriptor, num_bytes).
Port io_read(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 2);
  if (tup.elem_len != 2) {
    fprintf(stderr, "io_read: expected 2-tuple\n");
    return new_port(ERA, 0);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  u32 num_bytes = get_u24(get_val(tup.elem_buf[1]));

  if (fp == NULL) {
    fprintf(stderr, "io_read: invalid file descriptor\n");
    return new_port(ERA, 0);
  }

  /// Read a string.
  Bytes bytes;
  bytes.buf = (char*) malloc(sizeof(char) * num_bytes);
  bytes.len = fread(bytes.buf, sizeof(char), num_bytes, fp);

  if ((bytes.len != num_bytes) && ferror(fp)) {
    fprintf(stderr, "io_read: failed to read\n");
    free(bytes.buf);
    return new_port(ERA, 0);
  }

  // Convert it to a port.
  Port ret = inject_bytes(net, &bytes);
  free(bytes.buf);
  return ret;
}

// Opens a file with the provided mode.
// `argm` is a tuple (CON node) of the
// file name and mode as strings.
Port io_open(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 2);
  if (tup.elem_len != 2) {
    fprintf(stderr, "io_open: expected 2-tuple\n");
    return new_port(ERA, 0);
  }

  Str name = readback_str(net, book, tup.elem_buf[0]);
  Str mode = readback_str(net, book, tup.elem_buf[1]);

  for (u32 fd = 3; fd < sizeof(FILE_POINTERS); fd++) {
    if (FILE_POINTERS[fd] == NULL) {
      FILE_POINTERS[fd] = fopen(name.buf, mode.buf);

      free(name.buf);
      free(mode.buf);

      return new_port(NUM, new_u24(fd));
    }
  }

  fprintf(stderr, "io_open: too many open files\n");

  free(name.buf);
  free(mode.buf);

  return new_port(ERA, 0);
}

// Closes a file, reclaiming the file descriptor.
Port io_close(Net* net, Book* book, Port argm) {
  FILE* fp = readback_file(argm);
  if (fp == NULL) {
    fprintf(stderr, "io_close: invalid file descriptor\n");
    return new_port(ERA, 0);
  }

  int err = fclose(fp) != 0;
  if (err != 0) {
    fprintf(stderr, "io_close: failed to close: %i\n", err);
    return new_port(ERA, 0);
  }

  FILE_POINTERS[get_u24(get_val(argm))] = NULL;
  return new_port(ERA, 0);
}

// Writes a list of bytes to a file.
// `argm` is a tuple (CON node) of the
// file descriptor and list of bytes to write.
Port io_write(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 2);
  if (tup.elem_len != 2) {
    fprintf(stderr, "io_write: expected 2-tuple\n");
    return new_port(ERA, 0);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  Bytes bytes = readback_bytes(net, book, tup.elem_buf[1]);

  if (fp == NULL) {
    fprintf(stderr, "io_write: invalid file descriptor\n");
    free(bytes.buf);
    return new_port(ERA, 0);
  }

  if (fwrite(bytes.buf, sizeof(char), bytes.len, fp) != bytes.len) {
    fprintf(stderr, "io_write: failed to write\n");
  }

  free(bytes.buf);
  return new_port(ERA, 0);
}

// Flushes an output stream.
Port io_flush(Net* net, Book* book, Port argm) {
  FILE* fp = readback_file(argm);
  if (fp == NULL) {
    fprintf(stderr, "io_flush: invalid file descriptor\n");
    return new_port(ERA, 0);
  }

  int err = fflush(fp) != 0;
  if (err != 0) {
    fprintf(stderr, "io_flush: failed to flush: %i\n", err);
    return new_port(ERA, 0);
  }

  return new_port(ERA, 0);
}

// Seeks to a position in a file.
// `argm` is a 3-tuple (CON fd (CON offset whence)), where
// - fd is a file descriptor
// - offset is a signed byte offset
// - whence is what that offset is relative to:
//    - 0 (SEEK_SET): beginning of file
//    - 1 (SEEK_CUR): current position of the file pointer
//    - 2 (SEEK_END): end of the file
Port io_seek(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 3);
  if (tup.elem_len != 3) {
    fprintf(stderr, "io_seek: expected 3-tuple\n");
    return new_port(ERA, 0);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  i32 offset = get_i24(get_val(tup.elem_buf[1]));
  u32 whence = get_i24(get_val(tup.elem_buf[2]));

  if (fp == NULL) {
    fprintf(stderr, "io_write: invalid file descriptor\n");
    return new_port(ERA, 0);
  }

  int cwhence;
  switch (whence) {
    case 0: cwhence = SEEK_SET; break;
    case 1: cwhence = SEEK_CUR; break;
    case 2: cwhence = SEEK_END; break;
    default:
      fprintf(stderr, "io_seek: invalid whence\n");
      return new_port(ERA, 0);
  }

  if (fseek(fp, offset, cwhence) != 0) {
    fprintf(stderr, "io_seek: failed to seek\n");
  }

  return new_port(ERA, 0);
}

// Returns the current time as a tuple of the high
// and low 24 bits of a 48-bit nanosecond timestamp.
Port io_get_time(Net* net, Book* book, Port argm) {
  // Get the current time in nanoseconds
  u64 time_ns = time64();
  // Encode the time as a 64-bit unsigned integer
  u32 time_hi = (u32)(time_ns >> 24) & 0xFFFFFFF;
  u32 time_lo = (u32)(time_ns & 0xFFFFFFF);
  // Allocate a node to store the time
  u32 lps = 0;
  u32 loc = node_alloc_1(net, tm[0], &lps);
  node_create(net, loc, new_pair(new_port(NUM, new_u24(time_hi)), new_port(NUM, new_u24(time_lo))));
  // Return the encoded time
  return new_port(CON, loc);
}

// Sleeps.
// `argm` is a tuple (CON node) of the high and low
// 24 bits for a 48-bit duration in nanoseconds.
Port io_sleep(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 2);
  if (tup.elem_len != 2) {
    fprintf(stderr, "io_sleep: expected 2-tuple\n");
    return new_port(ERA, 0);
  }

  // Get the sleep duration node
  Pair dur_node = node_load(net, get_val(argm));
  // Get the high and low 24-bit parts of the duration
  u32 dur_hi = get_u24(get_val(tup.elem_buf[0]));
  u32 dur_lo = get_u24(get_val(tup.elem_buf[1]));
  // Combine into a 48-bit duration in nanoseconds
  u64 dur_ns = (((u64)dur_hi) << 24) | dur_lo;
  // Sleep for the specified duration
  struct timespec ts;
  ts.tv_sec = dur_ns / 1000000000;
  ts.tv_nsec = dur_ns % 1000000000;
  nanosleep(&ts, NULL);
  // Return an eraser
  return new_port(ERA, 0);
}

// Opens a dylib at the provided path.
// `argm` is a tuple of `filename` and `lazy`.
// `filename` is a λ-encoded string.
// `lazy` is a `bool` indicating if functions should be lazily loaded.
Port io_dl_open(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 2);
  Str str = readback_str(net, book, tup.elem_buf[0]);
  u32 lazy = get_u24(get_val(tup.elem_buf[1]));

  int flags = lazy ? RTLD_LAZY : RTLD_NOW;

  for (u32 dl = 0; dl < sizeof(DYLIBS); dl++) {
    if (DYLIBS[dl] == NULL) {
      DYLIBS[dl] = dlopen(str.text_buf, flags);
      if (DYLIBS[dl] == NULL) {
        fprintf(stderr, "failed to open dylib '%s': %s\n", str.text_buf, dlerror());

        return new_port(ERA, 0);
      }

      return new_port(NUM, new_u24(dl));
    }
  }

  fprintf(stderr, "io_dl_open: too many open dylibs\n");
  return new_port(ERA, 0);
}

// Calls a function from a loaded dylib.
// `argm` is a 3-tuple of `dylib_handle`, `symbol`, `args`.
// `dylib_handle` is the numeric node returned from a `DL_OPEN` call.
// `symbol` is a λ-encoded string of the symbol name.
// `args` is the argument to be provided to the dylib symbol.
Port io_dl_call(Net* net, Book* book, Port argm) {
  Tup tup = readback_tup(net, book, argm, 3);
  if (tup.elem_len != 3) {
    fprintf(stderr, "io_dl_call: expected 3-tuple\n");
    return new_port(ERA, 0);
  }

  void* dl = readback_dylib(tup.elem_buf[0]);
  Str symbol = readback_str(net, book, tup.elem_buf[1]);

  dlerror();
  Port (*func)(Net*, Book*, Port) = dlsym(dl, symbol.text_buf);
  char* error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "io_dl_call: failed to get symbol '%s': %s\n", symbol.text_buf, error);
  }

  return func(net, book, tup.elem_buf[2]);
}

// Closes a loaded dylib, reclaiming the handle.
Port io_dl_close(Net* net, Book* book, Port argm) {
  void* dl = readback_dylib(argm);
  if (dl == NULL) {
    fprintf(stderr, "io_dl_close: invalid handle\n");
    return new_port(ERA, 0);
  }

  int err = dlclose(dl) != 0;
  if (err != 0) {
    fprintf(stderr, "io_dl_close: failed to close: %i\n", err);
    return new_port(ERA, 0);
  }

  DYLIBS[get_u24(get_val(argm))] = NULL;
  return new_port(ERA, 0);
}

// Book Loader
// -----------

void book_init(Book* book) {
  book->ffns_buf[book->ffns_len++] = (FFn){"READ", io_read};
  book->ffns_buf[book->ffns_len++] = (FFn){"OPEN", io_open};
  book->ffns_buf[book->ffns_len++] = (FFn){"CLOSE", io_close};
  book->ffns_buf[book->ffns_len++] = (FFn){"FLUSH", io_flush};
  book->ffns_buf[book->ffns_len++] = (FFn){"WRITE", io_write};
  book->ffns_buf[book->ffns_len++] = (FFn){"SEEK", io_seek};
  book->ffns_buf[book->ffns_len++] = (FFn){"GET_TIME", io_get_time};
  book->ffns_buf[book->ffns_len++] = (FFn){"SLEEP", io_sleep};
  book->ffns_buf[book->ffns_len++] = (FFn){"DL_OPEN", io_dl_open};
  book->ffns_buf[book->ffns_len++] = (FFn){"DL_CALL", io_dl_call};
  book->ffns_buf[book->ffns_len++] = (FFn){"DL_CLOSE", io_dl_open};
}

// Monadic IO Evaluator
// ---------------------

// Runs an IO computation.
void do_run_io(Net* net, Book* book, Port port) {
  book_init(book);

  setlinebuf(stdout);
  setlinebuf(stderr);

  // IO loop
  while (TRUE) {
    // Normalizes the net
    normalize(net, book);

    // Reads the λ-Encoded Ctr
    Ctr ctr = readback_ctr(net, book, peek(net, port));

    // Checks if IO Magic Number is a CON
    if (get_tag(ctr.args_buf[0]) != CON) {
      break;
    }

    // Checks the IO Magic Number
    Pair io_magic = node_load(net, get_val(ctr.args_buf[0]));
    //printf("%08x %08x\n", get_u24(get_val(get_fst(io_magic))), get_u24(get_val(get_snd(io_magic))));
    if (get_val(get_fst(io_magic)) != new_u24(IO_MAGIC_0) || get_val(get_snd(io_magic)) != new_u24(IO_MAGIC_1)) {
      break;
    }

    switch (ctr.tag) {
      case IO_CALL: {
        Str  func = readback_str(net, book, ctr.args_buf[1]);
        FFn* ffn  = NULL;
        // FIXME: optimize this linear search
        for (u32 fid = 0; fid < book->ffns_len; ++fid) {
          if (strcmp(func.buf, book->ffns_buf[fid].name) == 0) {
            ffn = &book->ffns_buf[fid];
            break;
          }
        }
        if (ffn == NULL) {
          fprintf(stderr, "Unknown IO func '%s'\n", func.buf);

          free(func.buf);

          break;
        }

        free(func.buf);

        Port argm = ctr.args_buf[2];
        Port cont = ctr.args_buf[3];
        Port ret = ffn->func(net, book, argm);

        u32 lps = 0;
        u32 loc = node_alloc_1(net, tm[0], &lps);
        node_create(net, loc, new_pair(ret, ROOT));
        boot_redex(net, new_pair(new_port(CON, loc), cont));
        port = ROOT;
        continue;
      }
      case IO_DONE: {
        break;
      }
    }
    break;
  }
}
