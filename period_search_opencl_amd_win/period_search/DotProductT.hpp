#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace std;

namespace math {

    template <typename T, typename A, typename B>
    T DotProduct(vector<T, A> const &a, vector<T, B> const &b) {
        size_t size_a = a.size();
        size_t size_b = b.size();
        T result;

        for (size_t i = 0; i != a.size(); ++i)
        {
            try {
                result += a[i] * b[i];
            }
            catch (exception const& ex) {
                std::cerr << "Exception: " << ex.what() << endl;
            }
        }
        //T const result = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

        return result;
    }
}