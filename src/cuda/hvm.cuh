#ifndef hvm_cuh_INCLUDED
#define hvm_cuh_INCLUDED

#include "evaluator.cuh"
#include "structs/book.cuh"
#include "show.cuh"

#include <stdio.h>

//COMPILED_BOOK_BUF//
///COMPILED_INTERACT_CALL///

// Normalize the root net in book_buffer and print the result.
extern "C" void hvm_cu(u32* book_buffer) {
  // Loads the Book
  Book* book = (Book*)malloc(sizeof(Book));
  if (book_buffer) {
    if (!book_load(book, (u32*)book_buffer)) {
      fprintf(stderr, "failed to load book\n");

      return;
    }
    cudaMemcpyToSymbol(BOOK, book, sizeof(Book));
  }

  // Configures Shared Memory Size
  cudaFuncSetAttribute(evaluator, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(LNet));

  // Creates a new GNet
  GNet* gnet = gnet_create();

  // Start the timer
  clock_t start = clock();

  // Boots root redex, to expand @main
  gnet_boot_redex(gnet, new_pair(new_port(REF, 0), ROOT));

  #ifdef IO
  void do_run_io(GNet* gnet, Book* book, Port port);
  do_run_io(gnet, book, ROOT);
  #else
  gnet_normalize(gnet);
  #endif

  cudaDeviceSynchronize();

  // Stops the timer
  clock_t end = clock();
  double duration = ((double)(end - start)) / CLOCKS_PER_SEC;

  // Prints the result
  print_result<<<1,1>>>(gnet);

  // Reports errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernels. Error code: %s.\n", cudaGetErrorString(err));
    if (err == cudaErrorInvalidConfiguration) {
      fprintf(stderr, "Note: for now, HVM-CUDA requires a GPU with at least 128 KB of L1 cache per SM.\n");
    }
    exit(EXIT_FAILURE);
  }

  // Prints entire memdump
  //{
    //// Allocate host memory for the net
    //GNet *h_gnet = (GNet*)malloc(sizeof(GNet));

    //// Copy the net from device to host
    //cudaMemcpy(h_gnet, gnet, sizeof(GNet), cudaMemcpyDeviceToHost);

    //// Create a Net view of the host GNet
    //Net net;
    //net.g_node_buf = h_gnet->node_buf;
    //net.g_vars_buf = h_gnet->vars_buf;

    //// Print the net
    //print_net(&net, L_NODE_LEN, G_NODE_LEN);

    //// Free host memory
    //free(h_gnet);
  //}

  // Gets interaction count
  //cudaMemcpy(&itrs, &gnet->itrs, sizeof(u64), cudaMemcpyDeviceToHost);

  // Prints interactions, time and MIPS
  printf("- ITRS: %llu\n", gnet_get_itrs(gnet));
  printf("- LEAK: %llu\n", gnet_get_leak(gnet));
  printf("- TIME: %.2fs\n", duration);
  printf("- MIPS: %.2f\n", (double)gnet_get_itrs(gnet) / duration / 1000000.0);
}

#endif // hvm_cuh_INCLUDED
