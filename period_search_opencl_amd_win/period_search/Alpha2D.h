#pragma once
#include <CL/cl.h>
#include <memory.h>
#include "declarations.hpp"

template<size_t N, size_t M>
class Alpha2D
{
public:
    Alpha2D();
    virtual ~Alpha2D();
    cl_double& get() {
        return *pAr;
    }

    cl_double& operator ()()
    {
        return *pAr;
    }
};


template <size_t N, size_t M>
Alpha2D<N, M>::Alpha2D()
{
    wt = N;
    ht = M;
    pAr = vector_double(wt * ht);
}


template <size_t N, size_t M>
Alpha2D<N, M>::~Alpha2D()
{
    deallocate_vector(pAr);
}

//template<size_t N, size_t M>
//inline cl_double & Alpha2D<N, M>::get()
//{
//    return *pAr;
//}

size_t wt, ht;
cl_double *pAr = nullptr;

