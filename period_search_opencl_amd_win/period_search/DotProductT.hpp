#pragma once
#include <vector>
#include <stdexcept>

namespace math
{
    template <typename T, typename A, typename B>
    T DotProduct(std::vector<T, A> const &vectorA, std::vector<T, B> const &vectorB) {
        T result;
        size_t const sizeA = vectorA.size();
        size_t const sizeB = vectorB.size();

        if (sizeA != sizeB) {
            throw std::invalid_argument("Number of rows of vector 'a' are not equal to the number of columns of vector 'b'!");
        }

        for (size_t i = 0; i < sizeA; ++i)
        {
            T mid = vectorA[i] * vectorB[i];
            result = i == 0 ? mid : result + mid;
        }

        return result;
    }

    inline double DotProduct3(const cl_double3& vectorA, const cl_double3& vectorB)
    {
        auto result = 0.0;
        for (size_t i = 0; i < 3; i++)
        {
            result += vectorA.s[i] * vectorB.s[i];
        }

        //return vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z;
        return result;
    }

    inline double DotProduct3(const cl_double3 *vectorA, const cl_double3 *vectorB)
    {
        auto result = 0.0;
        for(size_t i = 0; i < 3; i++)
        {
            result += vectorA->s[i] * vectorB->s[i];
        }

        //return vectorA->x * vectorB->x + vectorA->y * vectorB->y + vectorA->z * vectorB->z;
        return result;
    }
}