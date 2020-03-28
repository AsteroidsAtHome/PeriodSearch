template<typename T, size_t N, size_t M, size_t Z>
class Array3D
{
    Array3D(const Array3D& b)
    {
        if (this != &b)
            *this = b;
    }

public:
    Array3D(T(&p)[N][M][Z])
    {
        wt = N;
        ht = M;
        zt = Z;
        pAr = p[0][0];
    }

    /*~A()
    {
    if (pAr)
    delete[] pAr;
    }*/

    const T& operator ()(size_t x, size_t y, size_t z) const
    {
        return pAr[x*zt*ht +y*zt + z];
    }

    T& operator ()(size_t x, size_t y, size_t z)
    {
        return pAr[x*zt*ht + y*zt + z];
    }

    void set(size_t x, size_t y, size_t z, T value) const;

private:
    size_t wt, ht, zt;
    T *pAr = nullptr;
};

template <typename T, size_t N, size_t M, size_t Z>
void Array3D<T, N, M, Z>::set(size_t x, size_t y, size_t z, T value) const
{
    pAr[x*zt*ht + y*zt + z] = value;
}