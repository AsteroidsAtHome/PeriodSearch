#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace std;

namespace math {

    template <typename T, typename A, typename B>
    T DotProduct(vector<T, A> const &a, vector<T, B> const &b) {
        T result;
        size_t size_a = a.size();
        size_t size_b = b.size();
        if (size_a != size_b) {
            throw std::invalid_argument("Number of rows of vector 'a' are not equal to the number of columns of vector 'b'!");
        }

        for (size_t i = 0; i < size_a; ++i)
        {
            T mid = a[i] * b[i];
            result = result + mid;
        }

        return result;
    }
}