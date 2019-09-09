#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <cmath>

using namespace std;

#define cudaCheck(stmt)													 	\
	do {																	\
		cudaError_t err = stmt;												\
		if (err != cudaSuccess) {											\
			cudaLog(ERROR, "Failed to run stmt ", #stmt);					\
			cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));	\
			return -1;														\
			}																\
		} while (0)															\

#define BLUR_KERNEL_SIZE 3


void allocateMemory(int img_width, int img_height, int* &d_img_input, int* &d_img_output) {
	// allocating the needed gpu memory for the input and the output
	cudaMalloc((void **) &d_img_input, (long long) (sizeof(int) * img_height * img_width));
	cudaMalloc((void **) &d_img_output, (long long) (sizeof(int) * img_height * img_width));
}

__global__ void img_blur_kernel(int* d_img_input, int* d_img_output, int img_height, int img_width){
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

			if (pr < 0 || pr >= img_height || pc <0 || pc >= img_width)
				continue;

			num_pixels++;
			pixels_sum += d_img_input[pr * img_width + pc];
		}
	}

	// Put the new pixel value
	d_img_output[i_thread] = int(pixels_sum * 1.0 / num_pixels);

	return;
}

int main() {
	int img_width = 1024;
	int img_height = 1024;

	// Reading img in a one dimensional array
	int* h_img = new int[img_height * img_width];
	int* d_img_input = nullptr;
	int* d_img_output = nullptr;

	// Allocating the input and the output images at the gpu memory using cudaMalloc
	allocateMemory(img_height, img_width, d_img_input, d_img_output);

	// Transfer the input image from the cpu (host) memory to the gpu memory
	long long img_bytes_count = (long long) (sizeof(int) * img_height * img_width);
	cudaMemcpy(h_img, d_img_input, img_bytes_count , cudaMemcpyHostToDevice);

	// Run the kernel, Choose the block dimensions and the grid size
	// Try with different block sizes, 
	// you get better selection by knowing the number of threads in a block and the maximum number of threads in the microprocessor
	int block_height = 16; 
	int block_width = 16;
	int grid_height = ceil(img_height * 1.0 / block_height);
	int grid_width = ceil(img_width * 1.0 / block_width);

	dim3 gridSize(grid_height, grid_width, 1);
	dim3 blockSize(block_height, block_width, 1);

	img_blur_kernel<<<gridSize, blockSize>>>(d_img_input, d_img_output, img_height, img_width);

	// Copy the output to the host memory
	cudaMemcpy(h_img, d_img_output, img_bytes_count, cudaMemcpyDeviceToHost);

	// Create and save the image

	return 0;
} 