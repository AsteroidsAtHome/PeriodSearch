#pragma once
#include <CL/cl.hpp>
#include <memory.h>
#include "declarations.hpp"


class Alpha
{
public:
    static cl_double **alpha;
    Alpha(size_t dim_x, size_t dim_y);
    virtual ~Alpha();


    cl_double Get(size_t x, size_t y) {
        return alpha[x][y];
    }

    void Set(size_t x, size_t y, cl_double value) {
        alpha[x][y] = value;
    }

private:
    static size_t _rows;
};


Alpha::Alpha(size_t rows, size_t columns)
{
    _rows = rows;
    alpha = matrix_double(rows, columns);
}


Alpha::~Alpha()
{
    deallocate_matrix_double(alpha, _rows);
}

cl_double **Alpha::alpha;
size_t Alpha::_rows;
