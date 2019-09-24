#ifndef TEMPLATEPROJECT_KERNEL_H
#define TEMPLATEPROJECT_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>

using namespace std;

#define BLOCK_SIZE 1024

class ScanKernel {
    dim3 gridSize;
    dim3 blockSize;
public:

    void setGridSize(int x, int y, int z) {
        this->gridSize = dim3(x, y, z);
    }

    void setBlockSize(int x, int y, int z) {
        this->blockSize = dim3(x, y, z);
    }

    // TODO to be changed
    void run(int *d_in, long *d_out, long *aux, int n);
};

#endif //TEMPLATEPROJECT_KERNEL_H
