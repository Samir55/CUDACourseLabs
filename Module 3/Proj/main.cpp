#include <iostream>
#include <cmath>
#include <opencv/cv.hpp>
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "ImageBlur/image_blur_kernel.h"

using namespace cv;
using namespace std;

#define cudaCheck(stmt)                                                        \
    do {                                                                    \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                            \
            cudaLog(ERROR, "Failed to run stmt ", #stmt);                    \
            cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));    \
            return -1;                                                        \
            }                                                                \
        } while (0)                                                            \

class ImageHandler {
public:
    static float *toArray(const Mat &m) {
        int r = m.rows;
        int c = m.cols;
        auto *arr = new float[r * c];

        int ic = 0;
        auto data = m.data;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                arr[ic] = float(data[ic++]);
            }
        }

        return arr;
    }

    static Mat toMat(const float *arr, int r, int c) {
        Mat m(r, c, 0);

        int ic = 0;
        auto &data = m.data;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                cout << arr[ic] << endl;
                data[ic] = int(arr[ic++]);
            }
        }
        return m;
    }
};

void allocateMemory(int img_height, int img_width, float *&d_img_input, float *&d_img_output) {
    // allocating the needed gpu memory for the input and the output
    cudaMalloc((void **) &d_img_input, (long long) (sizeof(float) * img_height * img_width));
    cudaMalloc((void **) &d_img_output, (long long) (sizeof(float) * img_height * img_width));
}


int main() {
    // Read the image
    string img_path = "/home/user1/GPUAcceleratedComputing/Module 3/Proj/ImageBlur/cameraman.png";
    Mat org_img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);

    imshow("Input image", org_img);

    int img_width = org_img.rows;
    int img_height = org_img.cols;

    // Reading img in a one dimensional array
    float *h_img = ImageHandler::toArray(org_img);
    float *d_img_input = nullptr;
    float *d_img_output = nullptr;

    // Allocating the input and the output images at the gpu memory using cudaMalloc
    allocateMemory(img_height, img_width, d_img_input, d_img_output);

    // Transfer the input image from the cpu (host) memory to the gpu memory
    auto img_bytes_count = (long long) (sizeof(float) * img_height * img_width);
    cudaMemcpy(d_img_input, h_img , img_bytes_count, cudaMemcpyHostToDevice);

    // Run the kernel, Choose the block dimensions and the grid size
    // Try with different block sizes,
    // you get better selection by knowing the number of threads in a block and the maximum number of threads in the microprocessor
    int block_height = 16;
    int block_width = 16;
    int grid_height = ceil(img_height * 1.0 / block_height);
    int grid_width = ceil(img_width * 1.0 / block_width);

    Kernel blur_kernel;
    blur_kernel.setGridSize(grid_height, grid_width, 1);
    blur_kernel.setBlockSize(block_height, block_width, 1);
    blur_kernel.run(d_img_input, d_img_output, img_height, img_width);

    // Copy the output to the host memory
    cudaMemcpy(h_img, d_img_output, img_bytes_count, cudaMemcpyDeviceToHost);

    // Create and save the image
    Mat out = ImageHandler::toMat(h_img, img_height, img_width);

    imshow("Output Image", out);
    waitKey(0);
    // Free the memory
    cudaFree(d_img_output);
    cudaFree(d_img_input);
    return 0;
}