//
// Created by aabdelreheem@EJAD.LOCAL on ١٢‏/٩‏/٢٠١٩.
//

#ifndef MATRIXMULTIPLICATION_GPUTIMER_H
#define MATRIXMULTIPLICATION_GPUTIMER_H
/* Timer for GPU
 * GPUTimer.h
 *
 * Created on: March 3, 2018
 * 		Author: ahmedsamir
 */

#include <cuda_runtime_api.h>
#include <cuda.h>

class GPUTimer {
public:
    cudaEvent_t e_start;
    cudaEvent_t e_stop;

    GPUTimer() {
        cudaEventCreate(&e_start);
        cudaEventCreate(&e_stop);
    }

    void start() {
        cudaEventRecord(e_start, 0);
    }

    void stop() {
        cudaEventRecord(e_stop, 0);
    }

    double elapsed() {
        float elapsed;
        cudaEventSynchronize(e_stop);
        cudaEventElapsedTime(&elapsed, e_start, e_stop);
        return elapsed;
    }

    ~GPUTimer() {
        cudaEventDestroy(e_start);
        cudaEventDestroy(e_stop);
    }

};

#endif //MATRIXMULTIPLICATION_GPUTIMER_H
