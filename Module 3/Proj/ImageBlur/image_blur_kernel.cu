#include "image_blur_kernel.h"

#define cudaCheck(stmt)                                                        \
    do {                                                                    \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                            \
            cudaLog(ERROR, "Failed to run stmt ", #stmt);                    \
            cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));    \
            return -1;                                                        \
            }                                                                \
        } while (0)                                                            \

#define BLUR_KERNEL_SIZE 5


__global__ void img_blur_kernel(const float *d_img_input, float *d_img_output, int img_height, int img_width) {
    // Get the thread index to get the mapping to the corresponding image pixel
    int r = threadIdx.x + blockIdx.x * blockDim.x; // height (row)
    int c = threadIdx.y + blockIdx.y * blockDim.y; // width (column);

    if (r >= img_height || c >= img_width) // Avoid of image borders
        return;

    int i_thread = r * img_width + c;
    int num_pixels = 0;
    int pixels_sum = 0;
    for (int i = -BLUR_KERNEL_SIZE; i < BLUR_KERNEL_SIZE; i++) {
        for (int j = -BLUR_KERNEL_SIZE; j < BLUR_KERNEL_SIZE; j++) {
            int pr = r + i;
            int pc = c + j;

            if (pr < 0 || pr >= img_height || pc < 0 || pc >= img_width)
                continue;

            num_pixels++;
            pixels_sum += d_img_input[pr * img_width + pc];
        }
    }

    // Put the new pixel value
    d_img_output[i_thread] = float(pixels_sum * 1.0 / num_pixels);
}

void Kernel::run(const float *d_img_input, float *d_img_output, int img_height, int img_width) {
    img_blur_kernel <<<this->gridSize, this->blockSize>>> (d_img_input, d_img_output, img_height, img_width);
}
