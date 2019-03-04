#pragma once
#include <vector>

template <typename T>
struct LightPoints
{
    /* JD time*/
    std::vector<T> time;

    /* Brightness */
    std::vector<T> brightness;


    /*explicit LightPoint(const size_t& size)
    {
        time.resize(size);
        brightness.resize(size);
    }*/
};

template <typename T>
struct InitialJd
{
    /* initial minimum JD */
    T min = 1e20;

    /* initial maximum JD */
    T max = -1e40;
};

template <typename T>
struct EllipsoidFunctionContext
{
    std::vector<int> indx;
    std::vector<T> fitvec;
    std::vector<T> er;
    std::vector<std::vector<T>> fmat;
    std::vector<std::vector<T>> fitmat;
};
