// Simple IO platform, that prints out normal forms + statistics
// (does not actually implement any IO)

// This file is designed to be concatenated to the generated runtime
// (so look at runtime.c to see what is in scope)

#include <stdio.h>
#include <sys/time.h>

struct timeval* start_time_p;

void io_setup() {
    start_time_p = malloc (sizeof (struct timeval));
    gettimeofday(start_time_p, NULL);
}

bool io_step() {

    struct timeval stop, start;
    gettimeofday(&stop, NULL);
    start = *start_time_p;

    // Prints result normal form
    const u64 code_mcap = 256 * 256 * 256; // max code size = 16 MB
    char* code_data = (char*)malloc(code_mcap * sizeof(char));
    assert(code_data);
    readback(
        code_data,
        code_mcap,
        mem,
        mem->node[0],
        id_to_name_data,
        id_to_name_size);
    printf("%s\n", code_data);
    fflush(stdout);

    // Prints result statistics
    u64 delta_time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    double rwt_per_sec = (double)ffi_cost / (double)delta_time;
    fprintf(stderr, "\n");
    fprintf(stderr, "Rewrites: %"PRIu64" (%.2f MR/s).\n", ffi_cost, rwt_per_sec);
    fprintf(stderr, "Mem.Size: %"PRIu64" words.\n", ffi_size);

    // Cleanup
    free(code_data);
    free(start_time_p);
    return false;
}
