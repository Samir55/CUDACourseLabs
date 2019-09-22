#include <iostream>
#include <map>
#include <assert.h>
#include "Histogram/Histogram.h"

using namespace std;

bool test_text() {
    char letters[] = "abcdefghijklmnopqrstuvwxyz";

    int freq[TEXT_NUM_BINS] = {};
    int n = rand() % (2 << 20);
    char *text = new char[n];

    // Generate an input array and run cpu calculation
    for (int i = 0; i < n; i++) {
        text[i] = letters[int(rand()) % 26];
        freq[text[i]]++;
    }

    // Run gpu calculation
    Histogram histogram;
    unsigned int *res = histogram.textHist(text, n);


    // Test
    for (int i = 0; i < TEXT_NUM_BINS; i++) {
        if (freq[i] != res[i])
            return false;
    }

    return true;
}

bool test_nums() {
    map<int, int> freq;
    int n = 100000;
    int *nums = new int[n];

    // Generate an input array and run cpu calculation
    for (int i = 0; i < n; i++) {
        nums[i] = int(rand()) % NUM_BINS;
        freq[nums[i]]++;
    }


    // Run gpu calculation
    Histogram histogram;
    unsigned int *res = histogram.hist(nums, n);


    // Test
    for (int i = 0; i < NUM_BINS; i++) {
        if (freq[i] != res[i])
            return false;
    }

    return true;
}

int main() {
    cudaDeviceSynchronize();

    assert(test_nums() == true);
    assert(test_text() == true);

    return 0;
}