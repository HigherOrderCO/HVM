#ifndef structs_book_cuh_INCLUDED
#define structs_book_cuh_INCLUDED

#include "../structs.cuh"

// Loads a compiled book buffer `buf` into `book`.
bool book_load(Book* book, u32* buf) {
  // Reads defs_len
  book->defs_len = *buf++;

  // Parses each def
  for (u32 i = 0; i < book->defs_len; ++i) {
    // Reads fid
    u32 fid = *buf++;

    // Gets def
    Def* def = &book->defs_buf[fid];

    // Reads name
    memcpy(def->name, buf, 256);
    buf += 64;

    // Reads safe flag
    def->safe = *buf++;

    // Reads lengths
    def->rbag_len = *buf++;
    def->node_len = *buf++;
    def->vars_len = *buf++;

    if (def->rbag_len > L_NODE_LEN/TPB) {
      fprintf(stderr, "def '%s' has too many redexes: %u\n", def->name, def->rbag_len);
      return false;
    }

    if (def->node_len > L_NODE_LEN/TPB) {
      fprintf(stderr, "def '%s' has too many nodes: %u\n", def->name, def->node_len);
      return false;
    }

    // Reads root
    def->root = *buf++;

    // Reads rbag_buf
    memcpy(def->rbag_buf, buf, 8*def->rbag_len);
    buf += def->rbag_len * 2;

    // Reads node_buf
    memcpy(def->node_buf, buf, 8*def->node_len);
    buf += def->node_len * 2;
  }

  return true;
}

#endif // structs_book_cuh_INCLUDED
