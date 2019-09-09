#include <color_to_greyscale_kernel.h>
#include "ImageBlurKernel/image_blur_kernel.h"
#include "Utils/utils.h"

int main() {
    // Read the image
    string img_path = "/home/user1/GPUAcceleratedComputing/Module 3/Proj/input/mellon_cat.jpg";
//    string img_path = "/home/user1/GPUAcceleratedComputing/Module 3/Proj/input/cameraman.png";
    Mat org_img = imread(img_path);

    int choice = -1;
    cout << "Apply which of the following\n"
            "1. Blur image.\n"
            "2. Convert to greyscale.\n"
            "3. Exit.\n";
    cin >> choice;

    if (choice == 3) {
        return 0;
    }

    int img_width = org_img.rows;
    int img_height = org_img.cols;
    int img_channels = org_img.channels();

    // Reading img in a one dimensional array
    float *d_img_input = nullptr;
    float *d_img_output = nullptr;

    // Allocating the input and the output images at the gpu memory using cudaMalloc
    auto img_bytes_count = (sizeof(float) * img_height * img_width * img_channels);
    cudaMalloc((void **) &d_img_input, img_bytes_count);

    // Transfer the input image from the cpu (host) memory to the gpu memory
    float *h_img = ImageHandler::toArray(org_img);
    float *h_out = nullptr;
    Mat out;
    cudaMemcpy(d_img_input, h_img, img_bytes_count, cudaMemcpyHostToDevice);

    // Run the kernel, Choose the block dimensions and the grid size
    // Try with different block sizes,
    // you get better selection by knowing the number of threads in a block and the maximum number of threads in the microprocessor
    int block_height = 16;
    int block_width = 16;
    int grid_height = ceil(img_height * 1.0 / block_height);
    int grid_width = ceil(img_width * 1.0 / block_width);

    if (choice == 1) {
        cudaMalloc((void **) &d_img_output, img_bytes_count);
        int kernel_size = 5;

        BlurImageKernel blur_kernel;
        blur_kernel.setGridSize(grid_height, grid_width, 1);
        blur_kernel.setBlockSize(block_height, block_width, 1);
        blur_kernel.run(d_img_input, d_img_output, kernel_size, img_height, img_width, img_channels);

        // Copy the output to the host memory
        h_out = new float[img_bytes_count];
        cudaMemcpy(h_out, d_img_output, img_bytes_count, cudaMemcpyDeviceToHost);
        out = ImageHandler::toMat(h_out, img_width, img_height, img_channels);
    } else if (choice == 2) {
        cudaMalloc((void **) &d_img_output, img_bytes_count / 3);

        ColorToGreyscaleKernel kernel;
        kernel.setGridSize(grid_height, grid_width, 1);
        kernel.setBlockSize(block_height, block_width, 1);
        kernel.run(d_img_input, d_img_output, img_height, img_width, img_channels);

        // Copy the output to the host memory
        h_out = new float[img_bytes_count / 3];
        cudaMemcpy(h_out, d_img_output, img_bytes_count / 3, cudaMemcpyDeviceToHost);
        out = ImageHandler::toMat(h_out, img_width, img_height, 1);
    }

    imshow("Input image", org_img);
    imshow("Output Image", out);
    waitKey(0);

    // Free the memory
    cudaFree(d_img_output);
    cudaFree(d_img_input);
    delete h_img;
    delete h_out;

    return 0;
}