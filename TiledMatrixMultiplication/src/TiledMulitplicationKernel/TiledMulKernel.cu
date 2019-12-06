#include <device_launch_parameters.h>
#include "TiledMulKernel.h"

#define cudaCheck(stmt)                                                        \
    do {                                                                    \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                            \
            cudaLog(ERROR, "Failed to run stmt ", #stmt);                    \
            cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));    \
            return -1;                                                        \
            }                                                                \
        } while (0)                                                            \

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

__global__ void kernel(int *d_matrix_a, int *d_matrix_b, int *d_matrix_res, int n, int m, int l) {
    // Get the corresponding row and column to the result matrix element
    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE;

    int max_dim = max(n, max(m, l));

    if (row >= max_dim && col >= max_dim) {
        return;
    }

    // Assuming the tile width = block width and tile height = block height
    __shared__ int s_a_matrix[TILE_SIZE][TILE_SIZE];
    __shared__ int s_b_matrix[TILE_SIZE][TILE_SIZE];

    // Loop over the tiles
    int p_value = 0.0;

    int a_i, b_i;
    for (int i = 0; i < int((max_dim - 1) * 1.0 / TILE_SIZE + 1); i++) {
        // Phase 0 transfer the data that this thread must transfer (one cell from matrix A and one cell from matrix b).
        a_i = row * m + (i * TILE_SIZE + tx); // row * width + col
        b_i = (i * TILE_SIZE + ty) * l + col;

        // When to copy values
        if (a_i < n * m)
            s_a_matrix[ty][tx] = d_matrix_a[a_i];
        else
            s_a_matrix[ty][tx] = 0;

        if (b_i < m * l)
            s_b_matrix[ty][tx] = d_matrix_b[b_i];
        else
            s_b_matrix[ty][tx] = 0;

        __syncthreads();

        // Phase 1 after finishing copying the data,
        for (int k = 0; k < TILE_SIZE; k++) {
            p_value += s_a_matrix[ty][k] * s_b_matrix[k][tx];
        }

        __syncthreads();
    }

    // Phase 1 Storing the p_value which is in the output matrix borders
    if (row < n && col < l) {
        d_matrix_res[row * l + col] = p_value;
    }
}

void TiledMulKernel::run(int *d_matrix_a, int *d_matrix_b, int *d_matrix_res, int n, int m, int l) {
    kernel << < this->gridSize, this->blockSize >> > (d_matrix_a, d_matrix_b, d_matrix_res, n, m, l);
}
