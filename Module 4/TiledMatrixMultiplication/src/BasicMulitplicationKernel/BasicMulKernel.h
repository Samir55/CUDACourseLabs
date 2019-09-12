#ifndef TEMPLATEPROJECT_KERNEL_H
#define TEMPLATEPROJECT_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;


class BasicMulKernel {
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
    void run();
};

#endif //TEMPLATEPROJECT_KERNEL_H
