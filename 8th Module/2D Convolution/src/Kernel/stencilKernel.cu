#include <device_launch_parameters.h>
#include "stencilKernel.h"

# define O_TILE_WIDTH 4
# define MASK_WIDTH 3
# define I_TILE_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)

# define out(i, j, k) output[i * width * depth + j + k * width]
# define in(i, j, k) input[i * width * depth + j + k * width]

__device__ float clamp(float x, float mn, float mx) {
    return max(mn, min(x, mx));
}

// Needs Testing
__global__ void stencil(float *output, float *input, int width, int height, int depth) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float ds_input[I_TILE_WIDTH][I_TILE_WIDTH];

    float front = in(i, j, 0);
    float back = in(i, j, 2);

    for (int k = 1; k < depth - 1; k++) {
        float val = 0.0;

        // Copying the 2d tile values at the current depth
        ds_input[threadIdx.y][threadIdx.x] = in(i, j, k);
        __syncthreads();

        // If it's within the boundaries
        if (i > 0 && j > 0 && i < height && j < width) {
            val = ds_input[threadIdx.y][threadIdx.x] +
                  ds_input[threadIdx.y + 1][threadIdx.x] +
                  ds_input[threadIdx.y - 1][threadIdx.x] +
                  ds_input[threadIdx.y][threadIdx.x + 1] +
                  ds_input[threadIdx.y][threadIdx.x - 1] +
                  front + back;
        }

        // Saving output
        if (i < O_TILE_WIDTH && j < O_TILE_WIDTH) {
            out(i, j, k) = clamp(val, 0, 255.0);
        }

        back = ds_input[threadIdx.y][threadIdx.x];
        front = in(i, j, k + 2);
        __syncthreads();
    }
}

void Kernel::stencilKernel(float *output, float *input, int width, int height, int depth) {
    stencil << < this->gridSize, this->blockSize >> > (output, input, width, height, depth);

}

void Kernel::run() {
}
