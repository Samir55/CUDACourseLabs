#ifndef TEMPLATEPROJECT_UTILS_H
#define TEMPLATEPROJECT_UTILS_H

#include <iostream>
#include <cmath>
#include <opencv/cv.hpp>
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

#define cudaCheck(stmt)                                                        \
    do {                                                                    \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                            \
            cudaLog(ERROR, "Failed to run stmt ", #stmt);                    \
            cudaLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));    \
            return -1;                                                        \
            }                                                                \
        } while (0)                                                            \


class ImageHandler {
public:
    static float *toArray(const Mat &m) {
        int r = m.rows;
        int c = m.cols;
        int channels = m.channels();
        auto *arr = new float[r * c * channels];

        int ic = 0;
        auto data = m.data;
        for (int i = 0; i < r * c * channels; ++i) {
            arr[i] = float(data[i]);
        }

        return arr;
    }

    static Mat toMat(const float *arr, int r, int c, int channels) {
        Mat m(r, c, 0);

        int ic = 0;
        auto &data = m.data;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < channels; k++) {
                    data[ic] = (unsigned char) (arr[ic]);
                    ic++;
                }
            }
        }
        return m;
    }
};


#endif //TEMPLATEPROJECT_UTILS_H
