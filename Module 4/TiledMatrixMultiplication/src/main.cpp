#include <iostream>
#include "MatrixMultiplication/MatrixMultiplication.h"

using namespace std;

int main() {
    MatrixMultiplication mat_mul;
    cudaDeviceSynchronize();
//    mat_mul.buildLargeInput(6);
    mat_mul.multiply("../input/input1.txt");
    mat_mul.printEquation();
//    mat_mul.printElapsedTime();
    return 0;
}