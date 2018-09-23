#pragma once
#include <vector>
#include "DotProductT.hpp"

/// <summary>
/// Struct of two vectors vector<T> e0 & vector<T> e for tempocentric and astrocentric coordinates
/// </summary>
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
