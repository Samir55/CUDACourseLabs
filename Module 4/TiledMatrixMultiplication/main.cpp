#include <iostream>
#include "MatrixMultiplication/MatrixMultiplication.h"

using namespace std;

int main() {
    MatrixMultiplication mat_mul;

    int *a;
    cudaMalloc((void **) &a, 16);

    int *res = mat_mul.multiply("../input/input.txt");

    auto dims = mat_mul.get_output_dim();
    for (int i = 0; i < dims.first; i++) {
        for (int j = 0; j < dims.second; j++) {
            cout << res[i * dims.second + j] << " ";
        }
        cout << endl;
    }


    return 0;
}