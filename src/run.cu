#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ftw.h>
#include "hvm.cu"

// Readback: λ-Encoded Ctr
struct Ctr {
  u32  tag;
  u32  args_len;
  Port args_buf[16];
};

// Readback: Tuples
struct Tup {
  u32  elem_len;
  Port elem_buf[8];
};

// Readback: λ-Encoded Str (UTF-32)
// FIXME: this is actually ASCII :|
struct Str {
  u32  len;
  char* buf;
};

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

// Result Tags
#define RESULT_OK  0
#define RESULT_ERR 1

// IOError = {
//   Type,           -- a type error
//   Name,           -- invalid io func name
//   Inner {val: T}, -- an error while calling an io func
// }
#define IO_ERR_TYPE 0
#define IO_ERR_NAME 1
#define IO_ERR_INNER 2

typedef struct IOError {
  u32 tag;
  Port val;
} IOError;

// List Type
#define LIST_NIL  0
#define LIST_CONS 1

// Readback
// --------

// Reads back a λ-Encoded constructor from device to host.
// Encoding: λt ((((t TAG) arg0) arg1) ...)
Ctr gnet_readback_ctr(GNet* gnet, Port port) {
  Ctr ctr;
  ctr.tag = -1;
  ctr.args_len = 0;

  // Loads root lambda
  Port lam_port = gnet_expand(gnet, port);
  if (get_tag(lam_port) != CON) return ctr;
  Pair lam_node = gnet_node_load(gnet, get_val(lam_port));

  // Loads first application
  Port app_port = gnet_expand(gnet, get_fst(lam_node));
  if (get_tag(app_port) != CON) return ctr;
  Pair app_node = gnet_node_load(gnet, get_val(app_port));

  // Loads first argument (as the tag)
  Port arg_port = gnet_expand(gnet, get_fst(app_node));
  if (get_tag(arg_port) != NUM) return ctr;
  ctr.tag = get_u24(get_val(arg_port));

  // Loads remaining arguments
  while (true) {
    app_port = gnet_expand(gnet, get_snd(app_node));
    if (get_tag(app_port) != CON) break;
    app_node = gnet_node_load(gnet, get_val(app_port));
    arg_port = gnet_expand(gnet, get_fst(app_node));
    ctr.args_buf[ctr.args_len++] = arg_port;
  }

  return ctr;
}

// Reads back a tuple of at most `size` elements. Tuples are
// (right-nested con nodes) (CON 1 (CON 2 (CON 3 (...))))
// The provided `port` should be `expanded` before calling.
extern "C" Tup gnet_readback_tup(GNet* gnet, Port port, u32 size) {
  Tup tup;
  tup.elem_len = 0;

  // Loads remaining arguments
  while (get_tag(port) == CON && (tup.elem_len + 1 < size)) {
    Pair node = gnet_node_load(gnet, get_val(port));
    tup.elem_buf[tup.elem_len++] = gnet_expand(gnet, get_fst(node));

    port = gnet_expand(gnet, get_snd(node));
  }

  tup.elem_buf[tup.elem_len++] = port;

  return tup;
}


// Converts a Port into a list of bytes.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
extern "C" Bytes gnet_readback_bytes(GNet* gnet, Port port) {
  // Result
  Bytes bytes;
  u32 capacity = 256;
  bytes.buf = (char*) malloc(sizeof(char) * capacity);
  bytes.len = 0;

  // Readback loop
  while (true) {
    // Normalizes the net
    gnet_normalize(gnet);

    // Reads the λ-Encoded Ctr
    Ctr ctr = gnet_readback_ctr(gnet, gnet_peek(gnet, port));

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
          bytes.buf = (char*) realloc(bytes.buf, capacity);
        }

        bytes.buf[bytes.len++] = get_u24(get_val(ctr.args_buf[0]));
        gnet_boot_redex(gnet, new_pair(ctr.args_buf[1], ROOT));
        port = ROOT;
        continue;
      }
    }
    break;
  }

  return bytes;
}

// Reads back a UTF-32 (truncated to 24 bits) string.
// Since unicode scalars can fit in 21 bits, HVM's u24
// integers can contain any unicode scalar value.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
extern "C" Str gnet_readback_str(GNet* gnet, Port port) {
  // gnet_readback_bytes is guaranteed to return a buffer with a capacity of at least one more
  // than the number of bytes read, so we can null-terminate it.
  Bytes bytes = gnet_readback_bytes(gnet, port);

  Str str;
  str.len = bytes.len;
  str.buf = bytes.buf;
  str.buf[str.len] = 0;

  return str;
}


/// Returns a λ-Encoded Ctr for a NIL: λt (t NIL)
/// Should only be called within `inject_bytes`, as a previous call
/// to `get_resources` is expected.
__device__ Port inject_nil(Net* net, TM* tm) {
  u32 v1 = tm->vloc[0];

  u32 n1 = tm->nloc[0];
  u32 n2 = tm->nloc[1];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(new_port(NUM, new_u24(LIST_NIL)), var));
  node_create(net, n2, new_pair(new_port(CON, n1), var));

  return new_port(CON, n2);
}

/// Returns a λ-Encoded Ctr for a CONS: λt (((t CONS) head) tail)
/// Should only be called within `inject_bytes`, as a previous call
/// to `get_resources` is expected.
/// The `char_idx` parameter is used to offset the vloc and nloc
/// allocations, otherwise they would conflict with each other on
/// subsequent calls.
__device__ Port inject_cons(Net* net, TM* tm, Port head, Port tail) {
  u32 v1 = tm->vloc[0];

  u32 n1 = tm->nloc[0];
  u32 n2 = tm->nloc[1];
  u32 n3 = tm->nloc[2];
  u32 n4 = tm->nloc[3];

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
__device__ Port inject_bytes(Net* net, TM* tm, Bytes *bytes) {
  // Allocate all resources up front:
  // - NIL needs  2 nodes & 1 var
  // - CONS needs 4 nodes & 1 var
  u32 len = bytes->len;

  if (!get_resources(net, tm, 0, 2, 1)) {
    return new_port(ERA, 0);
  }
  Port port = inject_nil(net, tm);

  for (u32 i = 0; i < len; i++) {
    if (!get_resources(net, tm, 0, 4, 1)) {
      return new_port(ERA, 0);
    }

    Port byte = new_port(NUM, new_u24(bytes->buf[len - i - 1]));
    port = inject_cons(net, tm, byte, port);
  }

  return port;
}

__global__ void make_bytes_port(GNet* gnet, Bytes bytes, Port* ret) {
  if (GID() == 0) {
    TM tm = tmem_new();
    Net net = vnet_new(gnet, NULL, gnet->turn);
    *ret = inject_bytes(&net, &tm, &bytes);
  }
}

// Converts a list of bytes to a Port.
// Encoding:
// - λt (t NIL)
// - λt (((t CONS) head) tail)
extern "C" Port gnet_inject_bytes(GNet* gnet, Bytes *bytes) {
  Port* d_ret;
  cudaMalloc(&d_ret, sizeof(Port));

  Bytes bytes_cu;
  bytes_cu.len = bytes->len;

  cudaMalloc(&bytes_cu.buf, sizeof(char) * bytes_cu.len);
  cudaMemcpy(bytes_cu.buf, bytes->buf, sizeof(char) * bytes_cu.len, cudaMemcpyHostToDevice);

  make_bytes_port<<<1,1>>>(gnet, bytes_cu, d_ret);

  Port ret;
  cudaMemcpy(&ret, d_ret, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);
  cudaFree(bytes_cu.buf);

  return ret;
}

/// Returns a λ-Encoded Ctr for a RESULT_OK: λt ((t RESULT_OK) val)
__device__ Port inject_ok(Net* net, TM* tm, Port val) {
  if (!get_resources(net, tm, 0, 3, 1)) {
    printf("inject_ok: failed to get resources\n");
    return new_port(ERA, 0);
  }

  u32 v1 = tm->vloc[0];

  u32 n1 = tm->nloc[0];
  u32 n2 = tm->nloc[1];
  u32 n3 = tm->nloc[2];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(val, var));
  node_create(net, n2, new_pair(new_port(NUM, new_u24(RESULT_OK)), new_port(CON, n1)));
  node_create(net, n3, new_pair(new_port(CON, n2), var));

  return new_port(CON, n3);
}

__global__ void make_ok_port(GNet* gnet, Port val, Port* ret) {
  if (GID() == 0) {
    TM tm = tmem_new();
    Net net = vnet_new(gnet, NULL, gnet->turn);
    *ret = inject_ok(&net, &tm, val);
  }
}

extern "C" Port gnet_inject_ok(GNet* gnet, Port val) {
  Port* d_ret;
  cudaMalloc(&d_ret, sizeof(Port));

  make_ok_port<<<1,1>>>(gnet, val, d_ret);

  Port ret;
  cudaMemcpy(&ret, d_ret, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);

  return ret;
}

/// Returns a λ-Encoded Ctr for a RESULT_ERR: λt ((t RESULT_ERR) err)
__device__ Port inject_err(Net* net, TM* tm, Port err) {
  if (!get_resources(net, tm, 0, 3, 1)) {
    printf("inject_err: failed to get resources\n");
    return new_port(ERA, 0);
  }

  u32 v1 = tm->vloc[0];

  u32 n1 = tm->nloc[0];
  u32 n2 = tm->nloc[1];
  u32 n3 = tm->nloc[2];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(err, var));
  node_create(net, n2, new_pair(new_port(NUM, new_u24(RESULT_ERR)), new_port(CON, n1)));
  node_create(net, n3, new_pair(new_port(CON, n2), var));

  return new_port(CON, n3);
}


__global__ void make_err_port(GNet* gnet, Port val, Port* ret) {
  if (GID() == 0) {
    TM tm = tmem_new();
    Net net = vnet_new(gnet, NULL, gnet->turn);
    *ret = inject_err(&net, &tm, val);
  }
}

extern "C" Port gnet_inject_err(GNet* gnet, Port val) {
  Port* d_ret;
  cudaMalloc(&d_ret, sizeof(Port));

  make_err_port<<<1,1>>>(gnet, val, d_ret);

  Port ret;
  cudaMemcpy(&ret, d_ret, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);

  return ret;
}

/// Returns a λ-Encoded Ctr for a Result/Err(IOError(..))
__device__ Port inject_io_err(Net* net, TM* tm, IOError err) {
  if (err.tag <= IO_ERR_NAME) {
    if (!get_resources(net, tm, 0, 2, 1)) {
      return new_port(ERA, 0);
    }

    u32 v1 = tm->vloc[0];

    u32 n1 = tm->nloc[0];
    u32 n2 = tm->nloc[1];

    vars_create(net, v1, NONE);
    Port var = new_port(VAR, v1);

    node_create(net, n1, new_pair(new_port(NUM, new_u24(err.tag)), var));
    node_create(net, n2, new_pair(new_port(CON, n1), var));

    return inject_err(net, tm, new_port(CON, n2));
  }

  if (!get_resources(net, tm, 0, 3, 1)) {
    return new_port(ERA, 0);
  }

  u32 v1 = tm->vloc[0];

  u32 n1 = tm->nloc[0];
  u32 n2 = tm->nloc[1];
  u32 n3 = tm->nloc[2];

  vars_create(net, v1, NONE);
  Port var = new_port(VAR, v1);

  node_create(net, n1, new_pair(err.val, var));
  node_create(net, n2, new_pair(new_port(NUM, new_u24(IO_ERR_INNER)), new_port(CON, n1)));
  node_create(net, n3, new_pair(new_port(CON, n2), var));

  return inject_err(net, tm, new_port(CON, n3));
}

__global__ void make_io_err_port(GNet* gnet, IOError err, Port* ret) {
  if (GID() == 0) {
    TM tm = tmem_new();
    Net net = vnet_new(gnet, NULL, gnet->turn);
    *ret = inject_io_err(&net, &tm, err);
  }
}

extern "C" Port gnet_inject_io_err(GNet* gnet, IOError err) {
  Port* d_ret;
  cudaMalloc(&d_ret, sizeof(Port));

  make_io_err_port<<<1,1>>>(gnet, err, d_ret);

  Port ret;
  cudaMemcpy(&ret, d_ret, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);

  return ret;
}

/// Returns a λ-Encoded Ctr for a Result/Err(IOError/Type)
extern "C" Port gnet_inject_io_err_type(GNet* gnet) {
  IOError io_error = {
    .tag = IO_ERR_TYPE,
  };

  return gnet_inject_io_err(gnet, io_error);
}

/// Returns a λ-Encoded Ctr for a Result/Err(IOError/Name)
extern "C" Port gnet_inject_io_err_name(GNet* gnet) {
  IOError io_error = {
    .tag = IO_ERR_NAME,
  };

  return gnet_inject_io_err(gnet, io_error);
}

/// Returns a λ-Encoded Ctr for a Result/Err(IOError/Inner(val))
extern "C" Port gnet_inject_io_err_inner(GNet* gnet, Port val) {
  IOError io_error = {
    .tag = IO_ERR_INNER,
    .val = val,
  };

  return gnet_inject_io_err(gnet, io_error);
}

/// Returns a λ-Encoded Ctr for an Result<T, IOError<String>>
/// `err` must be `NUL`-terminated.
extern "C" Port gnet_inject_io_err_str(GNet* gnet, char* err) {
  Port* d_bytes_port;
  cudaMalloc(&d_bytes_port, sizeof(Port));

  Bytes bytes_cu;
  bytes_cu.len = strlen(err);

  cudaMalloc(&bytes_cu.buf, sizeof(char) * bytes_cu.len);
  cudaMemcpy(bytes_cu.buf, err, sizeof(char) * bytes_cu.len, cudaMemcpyHostToDevice);

  make_bytes_port<<<1,1>>>(gnet, bytes_cu, d_bytes_port);

  Port bytes_port;
  cudaMemcpy(&bytes_port, d_bytes_port, sizeof(Port), cudaMemcpyDeviceToHost);
  cudaFree(d_bytes_port);

  cudaFree(bytes_cu.buf);

  return gnet_inject_io_err_inner(gnet, bytes_port);
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
    fprintf(stderr, "non-num where file descriptor was expected: %s\n", show_port(port).x);
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
// Returns: Result<Bytes, IOError<i24>>
Port io_read(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 2);
  if (tup.elem_len != 2) {
    fprintf(stderr, "io_read: expected 2-tuple\n");
    return gnet_inject_io_err_type(gnet);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  u32 num_bytes = get_u24(get_val(tup.elem_buf[1]));

  if (fp == NULL) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EBADF)));
  }

  /// Read a string.
  Bytes bytes;
  bytes.buf = (char*) malloc(sizeof(char) * num_bytes);
  bytes.len = fread(bytes.buf, sizeof(char), num_bytes, fp);

  if ((bytes.len != num_bytes) && ferror(fp)) {
    fprintf(stderr, "io_read: failed to read\n");
    free(bytes.buf);
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(ferror(fp))));
  }

  // Convert it to a port.
  Port ret = gnet_inject_bytes(gnet, &bytes);
  free(bytes.buf);

  return gnet_inject_ok(gnet, ret);
}

// Opens a file with the provided mode.
// `argm` is a tuple (CON node) of the
// file name and mode as strings.
// Returns: Result<File, IOError<i24>>
Port io_open(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 2);
  if (tup.elem_len != 2) {
    return gnet_inject_io_err_type(gnet);
  }

  Str name = gnet_readback_str(gnet, tup.elem_buf[0]);
  Str mode = gnet_readback_str(gnet, tup.elem_buf[1]);

  for (u32 fd = 3; fd < sizeof(FILE_POINTERS); fd++) {
    if (FILE_POINTERS[fd] == NULL) {
      FILE_POINTERS[fd] = fopen(name.buf, mode.buf);

      free(name.buf);
      free(mode.buf);

      if (FILE_POINTERS[fd] == NULL) {
        return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(errno)));
      }

      return gnet_inject_ok(gnet, new_port(NUM, new_u24(fd)));
    }
  }

  free(name.buf);
  free(mode.buf);

  // too many open files
  return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EMFILE)));
}

// Closes a file, reclaiming the file descriptor.
// Returns: Result<*, IOError<i24>>
Port io_close(GNet* gnet, Port argm) {
  FILE* fp = readback_file(argm);
  if (fp == NULL) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EBADF)));
  }

  if (fclose(fp) != 0) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(ferror(fp))));
  }

  FILE_POINTERS[get_u24(get_val(argm))] = NULL;

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Writes a list of bytes to a file.
// `argm` is a tuple (CON node) of the
// file descriptor and list of bytes to write.
// Returns: Result<*, IOError<i24>>
Port io_write(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 2);
  if (tup.elem_len != 2) {
    return gnet_inject_io_err_type(gnet);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  Bytes bytes = gnet_readback_bytes(gnet, tup.elem_buf[1]);

  if (fp == NULL) {
    free(bytes.buf);

    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EBADF)));
  }

  if (fwrite(bytes.buf, sizeof(char), bytes.len, fp) != bytes.len) {
    free(bytes.buf);

    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(ferror(fp))));
  }

  free(bytes.buf);

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Flushes an output stream.
// Returns: Result<*, IOError<i24>>
Port io_flush(GNet* gnet, Port argm) {
  FILE* fp = readback_file(argm);
  if (fp == NULL) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EBADF)));
  }

  if (fflush(fp) != 0) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(ferror(fp))));
  }

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Seeks to a position in a file.
// `argm` is a 3-tuple (CON fd (CON offset whence)), where
// - fd is a file descriptor
// - offset is a signed byte offset
// - whence is what that offset is relative to:
//    - 0 (SEEK_SET): beginning of file
//    - 1 (SEEK_CUR): current position of the file pointer
//    - 2 (SEEK_END): end of the file
// Returns: Result<*, IOError<i24>>
Port io_seek(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 3);
  if (tup.elem_len != 3) {
    fprintf(stderr, "io_seek: expected 3-tuple\n");
    return gnet_inject_io_err_type(gnet);
  }

  FILE* fp = readback_file(tup.elem_buf[0]);
  i32 offset = get_i24(get_val(tup.elem_buf[1]));
  u32 whence = get_i24(get_val(tup.elem_buf[2]));

  if (fp == NULL) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(EBADF)));
  }

  int cwhence;
  switch (whence) {
    case 0: cwhence = SEEK_SET; break;
    case 1: cwhence = SEEK_CUR; break;
    case 2: cwhence = SEEK_END; break;
    default:
      return gnet_inject_io_err_type(gnet);
  }

  if (fseek(fp, offset, cwhence) != 0) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(ferror(fp))));
  }

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Returns the current time as a tuple of the high
// and low 24 bits of a 48-bit nanosecond timestamp.
// Returns: Result<(u24, u24), IOError<*>>
Port io_get_time(GNet* gnet, Port argm) {
  // Get the current time in nanoseconds
  u64 time_ns = time64();
  // Encode the time as a 64-bit unsigned integer
  u32 time_hi = (u32)(time_ns >> 24) & 0xFFFFFFF;
  u32 time_lo = (u32)(time_ns & 0xFFFFFFF);
  // Return the encoded time
  return gnet_make_node(gnet, CON, new_port(NUM, new_u24(time_hi)), new_port(NUM, new_u24(time_lo)));
}

// Sleeps.
// `argm` is a tuple (CON node) of the high and low
// 24 bits for a 48-bit duration in nanoseconds.
// Returns: Result<*, IOError<*>>
Port io_sleep(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 2);
  if (tup.elem_len != 2) {
    return gnet_inject_io_err_type(gnet);
  }

  // Get the sleep duration node
  Pair dur_node = gnet_node_load(gnet, get_val(argm));
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

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Opens a dylib at the provided path.
// `argm` is a tuple of `filename` and `lazy`.
// `filename` is a λ-encoded string.
// `lazy` is a `bool` indicating if functions should be lazily loaded.
// Returns: Result<Dylib, IOError<String>>
Port io_dl_open(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 2);
  Str str = gnet_readback_str(gnet, tup.elem_buf[0]);
  u32 lazy = get_u24(get_val(tup.elem_buf[1]));

  int flags = lazy ? RTLD_LAZY : RTLD_NOW;

  for (u32 dl = 0; dl < sizeof(DYLIBS); dl++) {
    if (DYLIBS[dl] == NULL) {
      DYLIBS[dl] = dlopen(str.buf, flags);

      free(str.buf);

      if (DYLIBS[dl] == NULL) {
        return gnet_inject_io_err_str(gnet, dlerror());
      }

      return gnet_inject_ok(gnet, new_port(NUM, new_u24(dl)));
    }
  }

  return gnet_inject_io_err_str(gnet, "too many open dylibs");
}

// Calls a function from a loaded dylib.
// `argm` is a 3-tuple of `dylib_handle`, `symbol`, `args`.
// `dylib_handle` is the numeric node returned from a `DL_OPEN` call.
// `symbol` is a λ-encoded string of the symbol name.
// `args` is the argument to be provided to the dylib symbol.
//
// This function returns a Result with an Ok variant containing an
// arbitrary type.
//
// Returns Result<T, IOError<String>>
Port io_dl_call(GNet* gnet, Port argm) {
  Tup tup = gnet_readback_tup(gnet, argm, 3);
  if (tup.elem_len != 3) {
    fprintf(stderr, "io_dl_call: expected 3-tuple\n");

    return gnet_inject_io_err_type(gnet);
  }

  void* dl = readback_dylib(tup.elem_buf[0]);
  Str symbol = gnet_readback_str(gnet, tup.elem_buf[1]);

  dlerror();
  Port (*func)(GNet*, Port) = (Port (*)(GNet*, Port)) dlsym(dl, symbol.buf);
  char* error = dlerror();
  if (error != NULL) {
    return gnet_inject_io_err_str(gnet, error);
  }

  return gnet_inject_ok(gnet, func(gnet, tup.elem_buf[2]));
}

// Closes a loaded dylib, reclaiming the handle.
//
// Returns:  Result<*, IOError<String>>
Port io_dl_close(GNet* gnet, Book* book, Port argm) {
  void* dl = readback_dylib(argm);
  if (dl == NULL) {
    fprintf(stderr, "io_dl_close: invalid handle\n");

    return gnet_inject_io_err_type(gnet);
  }

  int err = dlclose(dl) != 0;
  if (err != 0) {
    return gnet_inject_io_err_str(gnet, dlerror());
  }

  DYLIBS[get_u24(get_val(argm))] = NULL;

  return gnet_inject_ok(gnet, new_port(ERA, 0));
}

// Removes a single file or an empty directory at the specified path.
// Returns Ok(None) if successful, or Err(reason) if an error occurs.
// This function attempts to remove both files and empty directories without
// first checking the type of the path.
// Returns: Result<*, IOError<i24>>
Port io_remove(GNet* gnet, Port argm) {
  Str s = gnet_readback_str(gnet, argm);

  int result = remove(s.buf);
  free(s.buf);

  if (result == 0) {
    return gnet_inject_ok(gnet, new_port(ERA, 0));
  } else {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(errno)));
  }
}

int remove_all_aux(const char* path, const struct stat* stat, int flags, struct FTW* ftw) {
  return remove(path);
}

int remove_all(const char* path) {
  struct stat st;
  if (stat(path, &st) != 0) {
    return remove(path);
  }
  if (S_ISDIR(st.st_mode)) {
    return nftw(path, remove_all_aux, 32, FTW_DEPTH | FTW_PHYS);
  } else {
    return remove(path);
  }
}

// Removes any file or directory recursively at the specified path.
// it will delete the directory and all its contents.
// Returns Ok(None) if successful, or Err(reason) if an error occurs.
// Note: For non-recursive deletion of an empty directory,
// this function behaves the same as delete_file(path).
// Returns: Result<*, IOError<i24>>
Port io_remove_all(GNet* gnet, Port argm) {
  Str path = gnet_readback_str(gnet, argm);
  
  int res = remove_all(path.buf);
  free(path.buf);

  if (0 == res) {
    return gnet_inject_ok(gnet, new_port(ERA, 0));
  } else {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(errno)));
  }
}

// Creates a new directory with the given path.
// Returns Ok(None) if sucessfull, or Err(reason) if an error occurs.
// Returns: Result<*, IOError<i24>>
Port io_mkdir(GNet* gnet, Port argm) {
  Str name = gnet_readback_str(gnet, argm);

  const mode_t mode = 0777;
  int status = mkdir(name.buf, mode);
  free(name.buf);

  if (status) {
    return gnet_inject_io_err_inner(gnet, new_port(NUM, new_i24(errno)));
  } else {
    return gnet_inject_ok(gnet, new_port(ERA, 0));
  }
}

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
  book->ffns_buf[book->ffns_len++] = (FFn){"RM", io_remove};
  book->ffns_buf[book->ffns_len++] = (FFn){"RM_ALL", io_remove_all};
  book->ffns_buf[book->ffns_len++] = (FFn){"MKDIR", io_mkdir};

  cudaMemcpyToSymbol(BOOK, book, sizeof(Book));
}

// Monadic IO Evaluator
// ---------------------

// Runs an IO computation.
void do_run_io(GNet* gnet, Book* book, Port port) {
  book_init(book);

  setlinebuf(stdout);
  setlinebuf(stderr);

  // IO loop
  while (true) {
    // Normalizes the net
    gnet_normalize(gnet);

    // Reads the λ-Encoded Ctr
    Ctr ctr = gnet_readback_ctr(gnet, gnet_peek(gnet, port));

    // Checks if IO Magic Number is a CON
    if (ctr.args_len < 1 || get_tag(ctr.args_buf[0]) != CON) {
      break;
    }

    // Checks the IO Magic Number
    Pair io_magic = gnet_node_load(gnet, get_val(ctr.args_buf[0]));
    //printf("%08x %08x\n", get_u24(get_val(get_fst(io_magic))), get_u24(get_val(get_snd(io_magic))));
    if (get_val(get_fst(io_magic)) != new_u24(IO_MAGIC_0) || get_val(get_snd(io_magic)) != new_u24(IO_MAGIC_1)) {
      break;
    }

    switch (ctr.tag) {
      case IO_CALL: {
        if (ctr.args_len != 4) {
          fprintf(stderr, "invalid IO_CALL: args_len = %u\n", ctr.args_len);
          break;
        }

        Str  func = gnet_readback_str(gnet, ctr.args_buf[1]);
        FFn* ffn  = NULL;
        // FIXME: optimize this linear search
        for (u32 fid = 0; fid < book->ffns_len; ++fid) {
          if (strcmp(func.buf, book->ffns_buf[fid].name) == 0) {
            ffn = &book->ffns_buf[fid];
            break;
          }
        }

        free(func.buf);

        Port argm = ctr.args_buf[2];
        Port cont = ctr.args_buf[3];

        Port ret;
        if (ffn == NULL) {
          ret = gnet_inject_io_err_name(gnet);
        } else {
          ret = ffn->func(gnet, argm);
        }

        Port p = gnet_make_node(gnet, CON, ret, ROOT);
        gnet_boot_redex(gnet, new_pair(p, cont));
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
