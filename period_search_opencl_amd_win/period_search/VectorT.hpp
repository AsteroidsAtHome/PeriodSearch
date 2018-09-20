#pragma once
#include <cmath>
#include "DotProductT.hpp"
#include "declarations.h"
#include <type_traits>

using namespace std;
namespace math {

    /// <summary>
    /// Accepts only integral and floating point types
    /// </summary>
    template <class T>
    class VectorT
    {
        //typename = typename enable_if<is_arithmetic<T>::value, T>::type
    private:
        vector<T> v;

    public:
        T &operator[](size_t index);
        T operator[](size_t index) const;
        T magnitude() const;
        VectorT(size_t size);
        virtual ~VectorT();
    };

    template <class T>
    T &VectorT<T>::operator[](size_t index)
    {
        return v[index];
    }

    template <class T>
    T VectorT<T>::operator[](size_t index) const
    {
        return v[index];
    }

    template <class T>
    T VectorT<T>::magnitude() const
    {
        T result = sqrt(DotProduct<T>(v, v));

        return result;
    }

    template <class T>
    VectorT<T>::VectorT(size_t size)
    {
        v.resize(size);
    }

    template <class T>
    VectorT<T>::~VectorT()
    {
    }

}