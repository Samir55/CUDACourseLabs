#include "image_blur_kernel.h"

__global__ void
img_blur_kernel(const float *d_img_input, float *d_img_output, int kernel_size, int img_height, int img_width,
                int img_channels) {
    // Get the thread index to get the mapping to the corresponding image pixel
    int r = threadIdx.x + blockIdx.x * blockDim.x; // height (row)
    int c = threadIdx.y + blockIdx.y * blockDim.y; // width (column);

    if (r >= img_height || c >= img_width || kernel_size % 2 == 0 ||
        kernel_size <= 0) // Avoid out of image borders and even kernel size
    {
        return;
    }

    int kernel_half_width = (kernel_size - 1) / 2;
    int i_thread = (r * img_width + c) * img_channels;

    for (int k = 0; k < img_channels; k++) {
        int num_pixels = 1;
        float pixels_sum = d_img_input[i_thread + k];

        for (int i = -kernel_half_width; i < kernel_half_width; i++) {
            for (int j = -kernel_half_width; j < kernel_half_width; j++) {
                int pr = r + i;
                int pc = c + j;

                if (pr < 0 || pr >= img_height || pc < 0 || pc >= img_width || (i == r && j == c))
                    continue;

                num_pixels++;
                pixels_sum += d_img_input[(pr * img_width + pc) * img_channels + k];
            }
        }

        // Put the new pixel value
        d_img_output[i_thread + k] = float(pixels_sum * 1.0 / num_pixels);
    }
}

void
BlurImageKernel::run(const float *d_img_input, float *d_img_output, int kernel_size, int img_height, int img_width,
                     int channels) {
    img_blur_kernel << < this->gridSize, this->blockSize >> >
                                         (d_img_input, d_img_output, kernel_size, img_height, img_width, channels);
}
