#include "malloc.h"
#include "Beta.h"
#include "globals.hpp"

Beta::Beta() {};

Beta::~Beta() {
    free(ptr);
}

void Beta::Init(int n) {
    if (length != n && n != 0) {
        if (ptr)
            free(ptr);
        ptr = (double *)malloc((n + 1) * sizeof(double));
        length = n;
    }
    else
    {
        ptr = (double *)malloc((n + 1) * sizeof(double));
        length = n;
    }
}

void Beta::set(int element, double value){
    ptr[element] = value;
}

double Beta::get(size_t x) {
    return ptr[x];
}
