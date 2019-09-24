#include <cmath>
#include "ScanKernel/scankernel.h"

#define MAX_ARR_SIZE 2048*65535
#define MAX_ELEMENT 1000

using namespace std;

int main() {
    int arr[100000];
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
    scanKernel.setBlockSize(BLOCK_SIZE, 1, 1);
    scanKernel.run(d_in, d_out, d_aux, n);

    // Copy back result
    cudaMemcpy(h_out, d_out, sizeof(long) * n, cudaMemcpyDeviceToHost);

    // Check the result with cpu result
    long *cpu_res = new long[n];
    bool ok = true;

    for (int i = 0; i < n; i++) {
        cpu_res[i] = i == 0 ? arr[0] : cpu_res[i - 1] + arr[i];
        ok = cpu_res[i] != h_out[i];
    }

    cout << "Checking result with CPU... ";

    if (!ok)
        cerr << "FAIL\n";
    else
        cout << "\033[1;32mOK\033[0m\n";

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);

    return 0;
}