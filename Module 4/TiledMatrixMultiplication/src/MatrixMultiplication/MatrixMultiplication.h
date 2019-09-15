//
// Created by aabdelreheem@EJAD.LOCAL on ١٢‏/٩‏/٢٠١٩.
//

#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H
#define MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H

#include <iostream>
#include <cmath>
#include "../TiledMulitplicationKernel/TiledMulKernel.h"
#include "../GPUTimer/GPUTimer.h"

using namespace std;

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

class MatrixMultiplication {
private:
    int *h_matrix_a;
    int *h_matrix_b;
    int *h_matrix_res;
    vector<vector<int>> cpu_res;

    int n{}, m{}, l{};


    void readMatrices(const string &file_path) {
        // Reading input
        freopen(file_path.c_str(), "r", stdin);

        cin >> n >> m >> l;

        if (h_matrix_a != nullptr)
            delete h_matrix_a;

        if (h_matrix_b != nullptr)
            delete h_matrix_b;

        h_matrix_a = new int[n * m];
        h_matrix_b = new int[m * l];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cin >> h_matrix_a[i * m + j];
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < l; j++) {
                cin >> h_matrix_b[i * l + j];
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
            int max_dim = max(n, max(m, l));
            multiplyKernel.setGridSize(int(ceil(max_dim * 1.0 / BLOCK_SIZE)), int(ceil(max_dim * 1.0 / BLOCK_SIZE)), 1);
            multiplyKernel.run(d_matrix_a, d_matrix_b, d_matrix_res, n, m, l);
        } else if (type == 2) {

        }

    }

    int *transferOutput() {
        h_matrix_res = new int[n * l];
        cudaMemcpy(h_matrix_res, d_matrix_res, sizeof(int) * n * l, cudaMemcpyDeviceToHost);
        return h_matrix_res;
    }

public:
    int *d_matrix_a;
    int *d_matrix_b;
    int *d_matrix_res;
    double elapsed_gpu_time;

    MatrixMultiplication() {
        h_matrix_a = h_matrix_b = h_matrix_res = d_matrix_a = d_matrix_b = d_matrix_res = nullptr;
    }

    int *multiply(const string &file_path) {

        this->readMatrices(file_path);

        this->allocateGPUMemory();

        this->transferInput();

        GPUTimer timer;
        timer.start();
        this->runKernel(1);
        timer.stop();
        elapsed_gpu_time = timer.elapsed();

        return this->transferOutput();
    }

    pair<int, int> get_output_dim() {
        return {this->n, this->l};
    }

    void printEquation() {
        cout << "Input Matrix A " << endl;
        cout << "==================== " << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cout << h_matrix_a[i * m + j] << " ";
            }
            cout << endl;
        }
        cout << "Input Matrix B " << endl;
        cout << "==================== " << endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < l; j++) {
                cout << h_matrix_b[i * l + j] << " ";
            }
            cout << endl;
        }


        cout << "Output Matrix " << endl;
        cout << "==================== " << endl << endl;

        bool check_with_cpu_result = true;
        cpuMultiplication();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                if (h_matrix_res[i * l + j] != cpu_res[i][j])
                    check_with_cpu_result = false;
                cout << h_matrix_res[i * l + j] << " ";
            }
            cout << endl;
        }

        cout << "Same as CPU result: " << (check_with_cpu_result ? "YES" : "NO") << endl << endl;
    }

    void printElapsedTime() {
        cout << "Input Matrix A size: " << n << " " << m << endl;
        cout << "Input Matrix B size: " << m << " " << l << endl;

        cout << "Elasped Time (GPU): " << elapsed_gpu_time << endl;
        double elapsed_cpu_time = cpuMultiplication();
        cout << "Elasped Time (CPU): " << elapsed_cpu_time << endl;
        cout << "Speed Up: " << elapsed_cpu_time / elapsed_gpu_time << endl;
    }

    double cpuMultiplication() {
        vector<vector<int>> a(n, vector<int>{});
        vector<vector<int>> b(m, vector<int>{});

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i].push_back(h_matrix_a[i * m + j]);
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < l; j++) {
                b[i].push_back(h_matrix_b[i * l + j]);
            }
        }

        cpu_res = vector<vector<int>>(n, vector<int>(l, 0));

        double start = clock();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                int p_val = 0;
                for (int k = 0; k < m; k++) {
                    p_val += a[i][k] * b[k][j];
                }
                cpu_res[i][j] = p_val;
            }
        }

        return (clock() - start) / CLOCKS_PER_SEC * 1000;
    }

    void buildLargeInput(int x) {
        freopen(("../input/input" + to_string(x) + ".txt").c_str(), "w", stdout);

        int n = 100;
        int m = 90;
        int l = 10;

        cout << n << " " << m << " " << l << endl;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cout << int(rand()) % 10 << " ";
            }
            cout << endl;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < l; j++) {
                cout << int(rand()) % 10 << " ";
            }
            cout << endl;
        }

    }


};


#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLICATION_H
