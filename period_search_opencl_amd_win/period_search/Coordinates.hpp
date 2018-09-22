#pragma once
#include <vector>
#include "DotProductT.hpp"

template <typename T>
struct Coordinates
{
    /* ecliptic astronomical tempocentric coordinates of the Sun in AU */
    std::vector<T> e0;

    /* ecliptic astrocentric coordinates of the Earth in AU */
    std::vector<T> e;


    explicit Coordinates(const size_t& size)
    {
        e0.resize(size);
        e.resize(size);
    }

    T DotProduct() const
    {
        T dotProduct = math::DotProduct<T>(e, e0);

        return dotProduct;
    }
};
