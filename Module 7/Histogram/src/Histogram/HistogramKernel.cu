#include <device_launch_parameters.h>
#include "HistogramKernel.h"

#define cudaCheck(stmt)                                                        \
    do {                                                                    \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                            \
            cudaLog(ERROR, "Failed to run stmt ", #stmt);                    \
            cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));    \
            return -1;                                                        \
            }                                                                \
        } while (0)                                                            \


__global__ void histogramKernel(int *vec, int n, unsigned int *d_out) {
    // Taking care of the following: Privatization, using shared memory and using atomic operations, num bins mapping
    // Splitting into sections and determining the number of the sections for each thread
    // and howe much the thread will take

    // Block size is 256 thread, so the elements_per_threads will be 50 in each thread
    // We can make use of the caolesced memories without nothing changed
    __shared__ unsigned int s_bins[NUM_BINS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculating hist in privatization shared memory
    while (index < n) {
        int count = atomicAdd(&s_bins[vec[index]], 1);
        index += gridDim.x * blockDim.x;
        __syncthreads();

    }

    __syncthreads();

    // Add to the global memory
    if (threadIdx.x == 0)
        for (int j = 0; j < NUM_BINS; j++)
            atomicAdd(&d_out[j], s_bins[j]);

}

__global__ void truncate_output(unsigned int *d_out) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < NUM_BINS)
        d_out[index] = 127 & d_out[index];
}


__global__ void textHistogramKernel(char *vec, int n, unsigned int *out) {
    // Taking care of the following: Privatization, using shared memory and using atomic operations, num bins mapping
    // Splitting into sections and determining the number of the sections for each thread
    // and howe much the thread will take

    // Block size is 256 thread, so the index_jump will be 50 in each thread
    // We can make use of the caolesced memories without nothing changed
    __shared__ unsigned int s_bins[TEXT_NUM_BINS];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Initializing the bin counters. This Step can be removed
    if (i < TEXT_NUM_BINS)
        s_bins[i] = 0;
    __syncthreads();

    // Calculating hist in privatization shared memory

    while (i < n) {
        // Atomic add
        atomicAdd(&s_bins[int(vec[i])], 1);
        i += gridDim.x * blockDim.x;
        __syncthreads();
    }

    __syncthreads();

    // Add to the global memory, as the block has 256 threads (more than the number of bins)
    int i_thread = threadIdx.x;
    if (i_thread < TEXT_NUM_BINS) {
        atomicAdd(&out[i_thread], s_bins[i_thread]);
    }

    __syncthreads();

}

void HistogramKernel::runHistogram(int *vec, int n, unsigned int *out) {
    histogramKernel << < this->gridSize, this->blockSize >> > (vec, n, out);
}

void HistogramKernel::runTextHistogram(char *d_in, int n, unsigned int *out) {
    textHistogramKernel << < this->gridSize, this->blockSize >> > (d_in, n, out);
}