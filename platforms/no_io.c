// Simple IO platform, that prints out normal forms + statistics
// (does not actually implement any IO)

// This file is designed to be concatenated to the generated runtime
// (so look at runtime.c to see what is in scope)

#include <stdio.h>
#include <sys/time.h>


int main(int argc, char* argv[]){

    Worker mem;
    build_main_term_with_args(&mem,_MAIN_,argc,argv);

    struct timeval stop, start;

    gettimeofday(&start, NULL);
    whnf(&mem,0);
    gettimeofday(&stop, NULL);

    // Prints result normal form
    const u64 code_mcap = 256 * 256 * 256; // max code size = 16 MB
    char* code_data = (char*)malloc(code_mcap * sizeof(char));
    assert(code_data);
    readback(
        code_data,
        code_mcap,
        &mem,
        mem.node[0],
        id_to_name_data,
        id_to_name_size);
    printf("%s\n", code_data);
    fflush(stdout);

    // Prints result statistics
    u64 delta_time = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    double rwt_per_sec = (double)mem.cost / (double)delta_time;
    fprintf(stderr, "\n");
    fprintf(stderr, "Rewrites: %"PRIu64" (%.2f MR/s).\n", mem.cost, rwt_per_sec);
    fprintf(stderr, "Mem.Size: %"PRIu64" words.\n", mem.size);

    // Cleanup
    free(code_data);
    free(mem.node);
    return 0;
}
