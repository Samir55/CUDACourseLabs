//
// Created by aabdelreheem@EJAD.LOCAL on ١٨‏/٩‏/٢٠١٩.
//

#ifndef MATRIXMULTIPLICATION_HISTOGRAMKERNEL_H
#define MATRIXMULTIPLICATION_HISTOGRAMKERNEL_H
#ifndef TEMPLATEPROJECT_KERNEL_H
#define TEMPLATEPROJECT_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define NUM_BINS 4096
#define TEXT_NUM_BINS 128 // ASCII characters count
#define MAX_ELEMENTS_PER_THREAD 50
#define BLOCK_SIZE 256
#define BLOCK_WIDTH 256

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
