#ifndef TEMPLATEPROJECT_STENCILKERNEL_H
#define TEMPLATEPROJECT_STENCILKERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;

#define NUM_CHANNELS 3
#define KERNEL_SIZE 5
#define O_TILE_WIDTH 12
#define I_TILE_WIDTH (O_TILE_WIDTH + KERNEL_SIZE - 1)

class ConvolutionKernel {
    dim3 gridSize;
    dim3 blockSize;
public:

    void setGridSize(int x, int y, int z) {
        this->gridSize = dim3(x, y, z);
    }

    void setBlockSize(int x, int y, int z) {
        this->blockSize = dim3(x, y, z);
    }

    void run(float *d_in, float *d_out, float *d_kernel, int height, int width);
};

#endif //TEMPLATEPROJECT_STENCILKERNEL_H
