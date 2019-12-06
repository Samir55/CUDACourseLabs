#ifndef TEMPLATEPROJECT_UTILS_H
#define TEMPLATEPROJECT_UTILS_H

void allocateMemory(int img_height, int img_width, float *&d_img_input, float *&d_img_output) {
    // allocating the needed gpu memory for the input and the output
    cudaMalloc((void **) &d_img_input, (long long) (sizeof(float) * img_height * img_width));
    cudaMalloc((void **) &d_img_output, (long long) (sizeof(float) * img_height * img_width));
}

class ImageHandler {
public:
    static float *toArray(const Mat &m) {
        int r = m.rows;
        int c = m.cols;
        auto *arr = new float[r * c];

        int ic = 0;
        auto data = m.data;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                arr[ic] = float(data[ic++]);
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
