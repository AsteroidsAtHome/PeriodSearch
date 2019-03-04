#pragma once
#include <vector>
#include <CL/cl.h>
#include "DotProductT.hpp"

struct PairCoordinates
{
    // ecliptic astronomical tempocentric coordinates of the Sun in AU
    cl_double3 e0;

    // ecliptic astrocentric coordinates of the Earth in AU
    cl_double3 e;

    double DotProduct() const
    {
        return math::DotProduct3(e0, e);
    }
};

class PairDouble3
{
public:
    const cl_double3 *xx{}, *xx0{};

    PairDouble3(const cl_double3& x, const cl_double3& x0);
    ~PairDouble3();
};

inline PairDouble3::PairDouble3(const cl_double3& x, const cl_double3& x0)
{
    xx = &x;
    xx0 = &x0;
}

inline PairDouble3::~PairDouble3() = default;


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

// Normalized distance vectors
struct CoordinatesDouble3
{
    /* Normals of ecliptic astronomical tempocentric coordinates of the Sun in AU */
    std::vector<cl_double3> ee0;

    /* Normals of ecliptic astrocentric coordinates of the Earth in AU */
    std::vector<cl_double3> ee;


    explicit CoordinatesDouble3(const size_t& size)
    {
        ee0.resize(size);
        ee.resize(size);
    }

    PairDouble3 operator[](int& idx) const
    {
        PairDouble3 a(ee[idx], ee0[idx]);
        return a;
    }
};

