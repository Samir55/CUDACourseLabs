#ifndef TEMPLATEPROJECT_KERNEL_H
#define TEMPLATEPROJECT_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;


class TiledMulKernel {
    dim3 gridSize;
    dim3 blockSize;
public:

    void setGridSize(int x, int y, int z) {
        this->gridSize = dim3(x, y, z);
    }

    void setBlockSize(int x, int y, int z) {
        this->blockSize = dim3(x, y, z);
    }

    void run(int *d_matrix_a, int *d_matrix_b, int *d_matrix_res, int n, int m, int l);
};

#endif //TEMPLATEPROJECT_KERNEL_H
