#include <cmath>
#include "ScanKernel/scankernel.h"

#define MAX_ARR_SIZE 2048*65535
#define MAX_ELEMENT 100000

using namespace std;

int main() {
    int arr[10];
    int n = sizeof(arr) / sizeof(int);

    // Fill the array
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % MAX_ELEMENT;
    }

    long *h_out = new long[n];
    long *d_out = nullptr;
    long *d_aux = nullptr;
    int *d_in = nullptr;
    int grid_size = ceil(n * 1.0 / (2 * BLOCK_SIZE));

    // Reserve memory
    cudaMalloc((void **) &d_in, sizeof(int) * n);
    cudaMalloc((void **) &d_out, sizeof(long) * n);
    cudaMalloc((void **) &d_aux, sizeof(long) * grid_size); // needed for phase two

    // Copy to device
    cudaMemcpy(d_in, arr, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Run kernel
    ScanKernel scanKernel;

    scanKernel.setGridSize(1, 1, 1);
    scanKernel.setBlockSize(1, 1, 1);
    scanKernel.run(d_in, d_out, d_aux, n);

    // Copy back result
    cudaMemcpy(h_out, d_out, sizeof(long) * n, cudaMemcpyDeviceToHost);

    // Check the result with cpu result
    for (int i = 0; i < n; i++) {
        cout << h_out[i] << " ";
    }

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);

    return 0;
}