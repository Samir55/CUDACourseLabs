#include "color_to_greyscale_kernel.h"

__global__ void
color_to_greyscale(const float *d_img_input, float *d_img_output, int img_height, int img_width,
                   int num_channels) {
    // Get the thread index to get the mapping to the corresponding image pixel
    int r = threadIdx.x + blockIdx.x * blockDim.x; // height (row)
    int c = threadIdx.y + blockIdx.y * blockDim.y; // width (column);

    if (r >= img_height || c >= img_width) // Avoid out of image borders and even kernel size
    {
        return;
    }

    int pixel_vals[3] = {};
    int i_thread = r * img_width + c;

    for (int k = 0; k < num_channels; k++) {
        pixel_vals[k] = d_img_input[i_thread * num_channels + k];
    }

    d_img_output[i_thread] = 0.21 * pixel_vals[0] + 0.71 * pixel_vals[1] + 0.07 * pixel_vals[2];
}

void ColorToGreyscaleKernel::run(const float *d_img_input, float *d_img_output, int img_height, int img_width,
                                 int num_channels) {
    color_to_greyscale << < this->gridSize, this->blockSize >> >
                                            (d_img_input, d_img_output, img_height, img_width, num_channels);
}
