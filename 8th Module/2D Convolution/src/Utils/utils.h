//
// Created by user1 on ٩‏/٩‏/٢٠١٩.
//

#ifndef TEMPLATEPROJECT_UTILS_H
#define TEMPLATEPROJECT_UTILS_H

#include <iostream>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;

class ImageHandler {
public:
    static float *toArray(const Mat &m, int pad_size = 0) {
        int r = m.rows;
        int c = m.cols;
        int channels = m.channels();
        auto *arr = new float[r * (c + 2 * pad_size) * channels];

        int ic = 0;
        auto data = m.data;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; j++) {
                for (int p = 0; p < pad_size; p++) {
                    arr[ic++] = 0;
                    arr[ic++] = 0;
                    arr[ic++] = 0;
                }
                for (int k = 0; k < channels; k++) {
                    arr[ic++] = data[(i * c + j) * channels + k];
                }
                for (int p = 0; p < pad_size; p++) {
                    arr[ic++] = 0;
                    arr[ic++] = 0;
                    arr[ic++] = 0;
                }
            }
        }

        return arr;
    }

    static Mat toMat(const float *arr, int r, int c) {
        Mat m(r, c, 0);

        int ic = 0;
        auto &data = m.data;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                cout << arr[ic] << endl;
                data[ic] = int(arr[ic++]);
            }
        }
        return m;
    }
};


#endif //TEMPLATEPROJECT_UTILS_H
