#include <iostream>
#include "MatrixMultiplication/MatrixMultiplication.h"

using namespace std;

int main() {
    MatrixMultiplication mat_mul;
    mat_mul.multiply("../input/input435084.txt");
    mat_mul.printElapsedTime();
    return 0;
}