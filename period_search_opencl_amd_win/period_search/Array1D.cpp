template<typename T, size_t N>
class Array1D
{
    Array1D(Array1D& b)
    {
        if (this != &b)
            *this = b;
    }

public:
    T *pAr = nullptr;
    Array1D(T(&p)[N]) //: wt(x)
    {
        wt = N;
        pAr = p[0];
    }

    /*~A()
    {
    if (pAr)
    delete[] pAr;
    }*/

    const T& operator ()(size_t x) const
    {
        return pAr[x];
    }

    T& operator ()(size_t x)
    {
        return pAr[x];
    }

    void set(size_t x, T value) const;

private:
    size_t wt;
};

template <typename T, size_t N>
void Array1D<T, N>::set(size_t x, T value) const
{
    pAr[x] = value;
}