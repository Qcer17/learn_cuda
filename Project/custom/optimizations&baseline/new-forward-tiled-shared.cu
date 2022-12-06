#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    extern __shared__ float shared_mem[]; // shared memory
    const int sm_width = TILE_WIDTH + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define sm_3d(i2, i1, i0) shared_mem[(i2) * (sm_width * sm_width) + (i1) * (sm_width) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int num_tile = ceil((float)(Width - K) / TILE_WIDTH);
    int h_out = ty + TILE_WIDTH * (bz / num_tile);
    int w_out = tx + TILE_WIDTH * (bz % num_tile);
    int h_base = TILE_WIDTH * (bz / num_tile);
    int w_base = TILE_WIDTH * (bz % num_tile);
    int m_out = by;  
    int b_out = bx;

    // float sum = 0.0;
    for (int m = 0; m < Channel; m++) {
        for (int i = ty; i < sm_width; i += TILE_WIDTH) {
            for (int j = tx; j < sm_width; j += TILE_WIDTH) {
                int h_in = i + h_base;
                int w_in = j + w_base;
                if (h_in >= 0 && h_in < Height && w_in >= 0 && w_in < Width) {
                    sm_3d(m, i, j) = in_4d(b_out, m, h_in, w_in);
                } 
                else {
                    sm_3d(m, i, j) = 0.0;
                }
            }
        }
    }
    __syncthreads();

    
    if (h_out < Height_out && w_out < Width_out) {
        float sum = 0.0;
        for (int m = 0; m < Channel; m++) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    sum += sm_3d(m, ty + i, tx + j) * mask_4d(m_out, m, i, j);
                }
            }
        }
        out_4d(b_out, m_out, h_out, w_out) = sum;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef sm_3d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
    cudaMalloc((void**)device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
    cudaMalloc((void**)device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);
    // cudaMemcpy(*device_output_ptr, host_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int sm_width = TILE_WIDTH + K - 1;
    dim3 dimGrid(Batch, Map_out, ceil((float)(Width - K + 1) / TILE_WIDTH) * ceil((float)(Height - K + 1) / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock, sizeof(float) * sm_width * sm_width * Channel>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
