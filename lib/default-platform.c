// Default IO platform, providing console IO

#include "hvm-api.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <inttypes.h>

void* io_setup() {
    printf("Reducing.\n");
    struct timeval * startp = malloc(sizeof(struct timeval));
    gettimeofday(startp, NULL);
    return (void*)startp;
}

// TODO: proper IO
// For now, just pritns stats & asks to halt
bool io_step(void* startp, Ptr* node) {
    struct timeval stop, start;
    gettimeofday(&stop, NULL);
    start = *((struct timeval*) startp);

    // Prints result statistics
    u64 delta_time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    double rwt_per_sec = (double)ffi_cost / (double)delta_time;
    fprintf(stderr, "\n");
    fprintf(stderr, "Rewrites: %"PRIu64" (%.2f MR/s).\n", ffi_cost, rwt_per_sec);
    fprintf(stderr, "Mem.Size: %"PRIu64" words.\n", ffi_size);

    // Cleanup
    free(startp);
    return false;
}
