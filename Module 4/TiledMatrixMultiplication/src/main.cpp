#include <iostream>
#include "MatrixMultiplication/MatrixMultiplication.h"

using namespace std;

int main() {
    MatrixMultiplication mat_mul;
    mat_mul.multiply("../input/input.txt");
    mat_mul.printEquation();
//    mat_mul.printElapsedTime();
    return 0;
}