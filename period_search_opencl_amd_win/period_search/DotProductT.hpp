#pragma once
#include <vector>
#include <stdexcept>

namespace math {

    template <typename T, typename A, typename B>
    T DotProduct(std::vector<T, A> const &vectorA, std::vector<T, B> const &vectorB) {
        T result;
        size_t const sizeA = vectorA.size();
        size_t const sizeB = vectorB.size();

        if (sizeA != sizeB) {
            throw std::invalid_argument("Number of rows of vector 'a' are not equal to the number of columns of vector 'b'!");
        }

        for(size_t i = 0; i < sizeA; ++i)
        {
            T mid = vectorA[i] * vectorB[i];
            result = i == 0 ? mid : result + mid;
        }

        return result;
    }
}