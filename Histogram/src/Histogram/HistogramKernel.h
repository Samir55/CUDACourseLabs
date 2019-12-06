#ifndef MATRIXMULTIPLICATION_HISTOGRAMKERNEL_H
#define MATRIXMULTIPLICATION_HISTOGRAMKERNEL_H
#ifndef TEMPLATEPROJECT_KERNEL_H
#define TEMPLATEPROJECT_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define BLOCK_WIDTH 256
#define ELEMENTS_PER_THREAD 256

#define TEXT_NUM_BINS 128 // ASCII characters count
#define NUM_BINS 4096

using namespace std;


class HistogramKernel {
    dim3 gridSize;
    dim3 blockSize;
public:

    void setGridSize(int x, int y, int z) {
        this->gridSize = dim3(x, y, z);
    }

    void setBlockSize(int x, int y, int z) {
        this->blockSize = dim3(x, y, z);
    }

    void runHistogram(int *vec, int n, unsigned int *out);

    void runTextHistogram(char *vec, int n, unsigned int *out);
};

#endif //TEMPLATEPROJECT_KERNEL_H
#endif //MATRIXMULTIPLICATION_HISTOGRAMKERNEL_H
