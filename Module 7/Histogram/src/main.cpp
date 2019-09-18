#include <iostream>
#include "Histogram/Histogram.h"

using namespace std;

int main() {
    cudaDeviceSynchronize();
    Histogram histogram;
    string text = "ahmedsamirhamedabdelreheem";

    unsigned int* res = histogram.textHist(text.c_str(), text.size());

    return 0;
}