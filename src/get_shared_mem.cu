#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t sharedMemPerBlock = prop.sharedMemPerBlock;
    int maxSharedMemPerBlockOptin;
    cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    size_t maxSharedMem = (sharedMemPerBlock > (size_t)maxSharedMemPerBlockOptin) ? sharedMemPerBlock : (size_t)maxSharedMemPerBlockOptin;

    // Subtract 3KB (3072 bytes) from the max shared memory as is allocated somewhere else
    maxSharedMem -= 3072;

    // Calculate the hex value
    unsigned int hexValue = (unsigned int)(maxSharedMem / 12);

    printf("0x%X", hexValue);

    return 0;
}
