//
// Created by user1 on ٩‏/٩‏/٢٠١٩.
//

#ifndef PORJ_IMAGE_BLUR_KERNEL_H
#define PORJ_IMAGE_BLUR_KERNEL_H

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;


class BlurImageKernel {
    dim3 gridSize;
    dim3 blockSize;
public:

    void setGridSize(int x, int y, int z) {
        this->gridSize = dim3(x, y, z);
    }

    void setBlockSize(int x, int y, int z) {
        this->blockSize = dim3(x, y, z);
    }

    void run(const float* d_img_input, float* d_img_output, int kernel_size, int img_height, int img_width, int img_channels);
};

#endif //PORJ_IMAGE_BLUR_KERNEL_H
