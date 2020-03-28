template<typename T, size_t N, size_t M>
class Array2D
{
    Array2D(const Array2D& b)
    {
        if (this != &b)
            *this = b;
    }

public:
    Array2D(T(&p)[N][M]) //: wt(x), ht(y)
    {
        wt = N;
        ht = M;
        pAr = p[0];
    }

    /*~A()
    {
    if (pAr)
    delete[] pAr;
    }*/

    const T& operator ()(size_t x, size_t y) const
    {
        return pAr[x * ht + y];
    }

    T& operator ()(size_t x, size_t y)
    {
        return pAr[x * ht + y];
    }

    void set(size_t x, size_t y, T value) const;

private:
    size_t wt, ht;
    T *pAr = nullptr;
};

template <typename T, size_t N, size_t M>
void Array2D<T, N, M>::set(size_t x, size_t y, T value) const
{
    pAr[x * ht + y] = value;
}