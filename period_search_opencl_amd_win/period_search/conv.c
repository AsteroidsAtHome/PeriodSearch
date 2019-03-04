/* Convexity regularization function

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include "globals.h"
#include "declarations.hpp"

double conv(const int& nc, const int& ma, double *dres)
{
    int j;

    double res = 0;
    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    for (auto i = 0; i < Numfac; i++)
    {
        res += Area[i] * Nor[nc][i];
        for (j = 0; j < Ncoef; j++)
            dres[j] += Darea[i] * Dg[i][j] * Nor[nc][i];
    }

    return(res);
}
