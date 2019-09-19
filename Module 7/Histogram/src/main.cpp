#include <iostream>
#include "Histogram/Histogram.h"

using namespace std;

int main() {
    cudaDeviceSynchronize();
    Histogram histogram;
    string text = "ahmedsamirhamedabdelreheem";
    int nums[] = {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 5};
    int n = sizeof(nums) / sizeof(int);

    unsigned int *res = histogram.hist(nums, n);

    for (int i = 0; i <= 6; i++)
        cout << i << " " << "count " << res[i] << endl;

    return 0;
}