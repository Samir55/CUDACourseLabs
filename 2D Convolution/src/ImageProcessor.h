#ifndef TEMPLATEPROJECT_IMAGEPROCESSOR_H
#define TEMPLATEPROJECT_IMAGEPROCESSOR_H

#include <cmath>
#include "ConvolutionKernel/convolutionKernel.h"

float *filter(float *h_img, int width, int height, int channels, float *h_kernel, int kernel_size) {
    if (kernel_size != KERNEL_SIZE || channels != NUM_CHANNELS)
        return nullptr;

    // Prepare a kernel
    ConvolutionKernel convKernel;
    convKernel.setBlockSize(I_TILE_WIDTH, I_TILE_WIDTH, 1);
    convKernel.setGridSize(ceil(height * 1.0 / I_TILE_WIDTH), ceil(width * 1.0 / I_TILE_WIDTH), 1);

    // Transfer data
    float *d_img = nullptr;
    float *d_out = nullptr;
    float *d_kernel = nullptr;

    cudaMalloc((void **) d_img, sizeof(float) * width * height * channels);
    cudaMalloc((void **) d_out, sizeof(float) * width * height * channels);
    cudaMalloc((void **) d_kernel, sizeof(float) * kernel_size * kernel_size);

    cudaMemcpy(d_img, h_img, sizeof(float) * width * height * channels, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * kernel_size * kernel_size);

    convKernel.run(d_img, d_out, d_kernel, height, width);

    float *h_out = new float[width * height * channels];
    cudaMemcpy(h_img, d_out, sizeof(float) * width * height * channels, cudaMemcpyDeviceToHost);

    return h_out;
}


#endif //TEMPLATEPROJECT_IMAGEPROCESSOR_H
