#include <iostream>

void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0; // index into A
    int j = 0; // index into B
    int k = 0; // index into C

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    if (i == m) {
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}

int main() {
    int arr1[5] = {1, 2, 31, 41, 52};
    int arr2[4] = {5, 6, 7, 8};
    int arr3[9] = {};

    merge_sequential(arr1, 5, arr2, 4, arr3);

    // Correctly prints the elements inside arr3
    for (int i = 0; i < 9; ++i) {
        std::cout << arr3[i] << " ";
    }
}
