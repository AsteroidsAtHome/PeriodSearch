/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <chrono>
#include <vector>
#include "globals.h"
#include "declarations.hpp"
#include "constants.h"
#include <iostream>
#include "LightPoint.h"
#include "globals.hpp"
#include <algorithm>
#include "MrqminContext.h"

using namespace std;
using namespace std::chrono;

int mrqmin(const CoordinatesDouble3& ndv,
    const LightPoints<double> &lightPoints,
    double sig[],
    std::vector<double> &a,
    const std::vector<int> &ia,
    const int &ma,
    double **covar,
    double **alpha,
    const int &mfit,
    struct MrqminContext &ctx)
{

    int j, k, l, err_code;
    //static int mfit, lastone, lastma; /* it is set in the first call*/
    const auto indexA = ma - 4 - Nphpar;
    const auto indexB = indexA + 1;

    if (Alamda < 0)
    {
        /* number of fitted parameters */
        Alamda = Alamda_start; /* initial alambda */
        curv(a);
        blmatrix(a[indexA], a[indexB]);
        Ochisq = mrqcof(ndv, lightPoints, sig, a, ia, ma, alpha, ctx.beta, mfit);
        std::copy(a.begin(), a.end(), ctx.atry.begin());  // TODO: test this!
        //for (j = 1; j <= ma; j++)
        //    atry[j] = a[j];
    }

    std::copy(ctx.beta.begin(), ctx.beta.end(), ctx.da.begin());
    for (j = 0; j < mfit; j++)
    {
        for (k = 0; k < mfit; k++)
            covar[j][k] = alpha[j][k];

        covar[j][j] = alpha[j][j] * (1 + Alamda);
        //da[j] = beta[j];
    }

    err_code = gauss_errc(covar, mfit, ctx.da);

    if (err_code != 0) return(err_code);

    j = 0;
    for (l = 1; l <= ma; l++)
    {
        if (ia[l - 1])
        {
            ctx.atry[l] = a[l] + ctx.da[j];
            j++;
        }
    }

    curv(ctx.atry);
    blmatrix(ctx.atry[indexA], ctx.atry[indexB]);
    Chisq = mrqcof(ndv, lightPoints, sig, ctx.atry, ia, ma, covar, ctx.da, mfit);

    if (Chisq < Ochisq)
    {
        Alamda = Alamda / Alamda_incr;
        for (j = 0; j < mfit; j++)
        {
            for (k = 0; k < mfit; k++)
                alpha[j][k] = covar[j][k];

            ctx.beta[j] = ctx.da[j];
        }

        std::copy(ctx.atry.begin(), ctx.atry.end(), a.begin());
        /*for (l = 1; l <= ma; l++)
            a[l] = atry[l];*/
    }
    else
    {
        Alamda = Alamda_incr * Alamda;
        Chisq = Ochisq;
    }

    return(0);
}

