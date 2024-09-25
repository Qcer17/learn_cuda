#include <stdio.h>

__global__ void hello(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("idx: %d ", idx);
}

int main(){
    hello<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}