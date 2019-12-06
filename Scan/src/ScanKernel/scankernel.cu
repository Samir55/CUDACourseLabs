#include "scankernel.h"


__global__ void sum(long *d_out, int n, long *aux) {
    __shared__ long s_val;

    if (blockIdx.x == 0) {
        return;
    }

    if (threadIdx.x == 0)
        s_val = aux[blockIdx.x - 1];

    __syncthreads();

    int i = threadIdx.x + 2 * blockIdx.x * BLOCK_SIZE;

    if (i < n)
        d_out[i] += s_val;
    if (i + BLOCK_SIZE < n)
        d_out[i + BLOCK_SIZE] += s_val;

//    printf("%d %d\n", threadIdx.x, d_out[i]);
}

__global__ void scan(int *d_in, long *d_out, long *aux, int n) {
    // each thread is responsible for two elements
    int start = 2 * BLOCK_SIZE * blockIdx.x;
    int i = start + threadIdx.x; // first element
    int j = i + BLOCK_SIZE; // second element

    // Copy to the shared memory
    __shared__ long xy[BLOCK_SIZE * 2];  // each thread is responsible for two elements

    if (i < n)
        xy[threadIdx.x] = d_in[i];
    else
        xy[threadIdx.x] = 0;

    if (j < n)
        xy[threadIdx.x + BLOCK_SIZE] = d_in[j];
    else
        xy[threadIdx.x + BLOCK_SIZE] = 0;

    __syncthreads();

    // Phase one reduction
    int stride = 1;
    while (stride < BLOCK_SIZE) { // The stride should be less than or equal to the half length of the segment

        int index = (threadIdx.x + 1) * 2 * stride - 1; // We run on the odd

        if (index < BLOCK_SIZE * 2) { // Valid index in the segment
            xy[index] += xy[index - stride];
        }

        stride *= 2;

        __syncthreads();
    }

    // Phase two post reduction
    stride = BLOCK_SIZE;
    while (stride > 0) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;

        if (index + stride < 2 * BLOCK_SIZE) {
            xy[index + stride] += xy[index];
        }

        stride /= 2;
        __syncthreads();
    }

    // Save in the output (first phase in saving before running the second-phase sum kernel)
    // Why this method didn't work in matrix multiplication please check (why not using i and j)
    if (i < n) {
        d_out[i] = xy[threadIdx.x];
    }
    if (j < n) {
        d_out[j] = xy[threadIdx.x + BLOCK_SIZE];
    }


    aux[blockIdx.x] = xy[2 * BLOCK_SIZE - 1]; // copy the last element in the array.
}

void ScanKernel::run(int *d_in, long *d_out, long *aux, int n) {
    scan << < this->gridSize, this->blockSize >> > (d_in, d_out, aux, n);
    sum << < this->gridSize, this->blockSize >> > (d_out, n, aux);

}
