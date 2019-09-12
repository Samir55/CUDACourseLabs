//
// Created by aabdelreheem@EJAD.LOCAL on ١٢‏/٩‏/٢٠١٩.
//

#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H
#define MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H

#include <iostream>
#include <cmath>
#include "../TiledMulitplicationKernel/TiledMulKernel.h"
//#include "../BasicMulitplicationKernel/BasicMulKernel.h"

using namespace std;

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

class MatrixMultiplication {
private:
    int *h_matrix_a;
    int *h_matrix_b;
    int *h_matrix_res;

    int n{}, m{}, l{};


    void readMatrices(const string &file_path) {
        // Reading input TODO change from freopen to file input
        freopen(file_path.c_str(), "r", stdin);
//        freopen("../input/output.txt", "w", stdout);

        cin >> n >> m >> l;

        if (h_matrix_a != nullptr)
            delete h_matrix_a;

        if (h_matrix_b != nullptr)
            delete h_matrix_b;

        h_matrix_a = new int[n * m];
        h_matrix_b = new int[m * l];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cin >> h_matrix_a[i * n + j];
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < l; j++) {
                cin >> h_matrix_b[i * m + j];
            }
        }
    }

    void allocateGPUMemory() {
        cudaMalloc((void **) &d_matrix_a, sizeof(int) * n * m);
        cudaMalloc((void **) &d_matrix_b, sizeof(int) * m * l);
        cudaMalloc((void **) &d_matrix_res, sizeof(int) * n * l);
    }

    void transferInput() {
        cudaMemcpy(d_matrix_a, h_matrix_a, sizeof(int) * n * m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix_b, h_matrix_b, sizeof(int) * m * l, cudaMemcpyHostToDevice);
    }

    void runKernel(int type) {
        if (type == 1) {
            TiledMulKernel multiplyKernel;
            multiplyKernel.setBlockSize(16, 16, 1);
            multiplyKernel.setGridSize(int(ceil(n * 1.0 / BLOCK_SIZE)), int(ceil(l * 1.0 / BLOCK_SIZE)), 1);
            multiplyKernel.run(d_matrix_a, d_matrix_b, d_matrix_res, n, m, l);
        } else if (type == 2) {

        }

    }

    int *transferOutput() {
        h_matrix_res = new int[m * l];
        cudaMemcpy(h_matrix_res, d_matrix_res, sizeof(int) * m * l, cudaMemcpyDeviceToHost);
        return h_matrix_res;
    }

public:
    int *d_matrix_a;
    int *d_matrix_b;
    int *d_matrix_res;

    MatrixMultiplication() {
        h_matrix_a = h_matrix_b = h_matrix_res = d_matrix_a = d_matrix_b = d_matrix_res = nullptr;
    }

    int *multiply(const string &file_path) {
        this->readMatrices(file_path);

        this->allocateGPUMemory();

        this->transferInput();

        this->runKernel(1);

        return this->transferOutput();
    }

    pair<int, int> get_output_dim() {
        return {this->n, this->l};
    }

};


#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H
