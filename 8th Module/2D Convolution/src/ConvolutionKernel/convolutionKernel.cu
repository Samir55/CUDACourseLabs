#include <device_launch_parameters.h>
#include "convolutionKernel.h"

// RGB RGB RGB RGB R:0 G:1 B:2
#define in(i, j, k) d_in[((i * width + j) * NUM_CHANNELS + k)]
#define out(i, j, k) d_out[((i * width + j) * NUM_CHANNELS + k)]

__device__ float clamp(float x, float mn, float mx) {
    return max(mn, min(x, mx));
}

__global__ void applyFilerKernel(float *d_in, float *d_out, float *__restrict__ d_kernel, int height, int width) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    float __shared__ ds_in[I_TILE_WIDTH][I_TILE_WIDTH][NUM_CHANNELS]; // You should here pad

    // Loop over the three channels
    for (int k = 0; k < NUM_CHANNELS; k++) {
        // Copy to the shared memory
        if (i < height && j < width) {
            ds_in[threadIdx.y][threadIdx.x][k] = in(i, j - 2, k);
        }
        __syncthreads();

        // Run the convolution if the thread is in range
        float val = 0.0;

        if (threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH) {
            for (int y = 0; y < KERNEL_SIZE; y++) {
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    val += (ds_in[y + threadIdx.y][x + threadIdx.x][k] * d_kernel[y * KERNEL_SIZE + x]);
                }
            }

            out(i, j, k) = clamp(val, 0, 255);
            __syncthreads();
        }
    }
}

void ConvolutionKernel::run(float *d_in, float *d_out, float *d_kernel, int height, int width) {
    applyFilerKernel << < this->gridSize, this->blockSize >> > (d_in, d_out, d_kernel, width, height);
}
