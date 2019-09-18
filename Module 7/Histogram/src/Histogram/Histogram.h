//
// Created by aabdelreheem@EJAD.LOCAL on ١٨‏/٩‏/٢٠١٩.
//

#ifndef MATRIXMULTIPLICATION_HISTOGRAM_H
#define MATRIXMULTIPLICATION_HISTOGRAM_H

#include "HistogramKernel.h"
#include <cmath>

using namespace std;

class Histogram {

public:
    Histogram() = default;

    unsigned int *hist(int *vec = nullptr, int n = 0, unsigned int *out = nullptr) {
        HistogramKernel kernel;

        kernel.setGridSize(ceil(n * 1.0  / BLOCK_SIZE), 1, 1);
        kernel.setBlockSize(BLOCK_WIDTH, 1, 1);

        // Allocating memory and copying data
        int *d_in;
        unsigned int *d_out;
        unsigned int *h_out;

        cudaMalloc((void **) &d_in, sizeof(int) * n);
        cudaMalloc((void **) &d_out, sizeof(unsigned int) * n);
        cudaMemcpy((void **) (&d_in), vec, sizeof(int) * n, cudaMemcpyHostToDevice);

        kernel.runHistogram(d_in, n, d_out);

        cudaMemcpy(h_out, d_out, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);
        return h_out;
    }

    unsigned int *textHist(const char *vec = nullptr, int n = 0, unsigned int *out = nullptr) {
        HistogramKernel kernel;

        kernel.setGridSize(ceil(n * 1.0 / BLOCK_SIZE), 1, 1);
        kernel.setBlockSize(BLOCK_WIDTH, 1, 1);

        // Allocating memory and copying data
        char *d_in;
        unsigned int *d_out;
        unsigned int *h_out = nullptr;
        cudaMalloc((void **) &d_in, sizeof(char) * n);
        cudaMalloc((void **) &d_out, sizeof(unsigned int) * n);

        cudaMemcpy((void **)&d_in, vec, sizeof(char) * n, cudaMemcpyHostToDevice);

        kernel.runTextHistogram(d_in, n, d_out);

        cudaMemcpy(h_out, d_out, sizeof(unsigned int) * TEXT_NUM_BINS, cudaMemcpyDeviceToHost);
        return h_out;
    }
};

#endif //MATRIXMULTIPLICATION_HISTOGRAM_H
