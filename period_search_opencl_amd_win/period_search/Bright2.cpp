/* computes integrated brightness of all visible and illuminated areas
   and its derivatives

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <vector>
#include "globals.h"
#include "declarations.hpp"
#include "constants.h"
#include <valarray>
#include <algorithm>

double Bright2(const PairDouble3 &ndv, const double &time, const std::vector<double> &cg, cl_double *dyda, int ncoef)
{
    int i, j,
        incl[MAX_N_FAC], //array of indexes of facets to Area, Dg, Nor. !!!!!!!!!!!incl IS ZERO INDEXED
        incl_count = 0;

    double tmpdyda,
        e[4], e0[4],
        php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1],
           dbr[MAX_N_FAC], //IS ZERO INDEXED
        de[4][4], de0[4][4], dtm[4][4][4], tmat[4][4];

    //std::vector<cl_double3> tmat(4);

    double tmpdyda1 = 0, tmpdyda2 = 0, tmpdyda3 = 0;
    double tmpdyda4 = 0, tmpdyda5 = 0;

    const int ncoef0 = ncoef - 2 - Nphpar;
    const double cl = exp(cg[ncoef - 1]);                     /* Lambert */
    const double cls = cg[ncoef];                             /* Lommel-Seeliger */
    const double cosAlpha = math::DotProduct3(ndv.xx, ndv.xx0);
    const double alpha = acos(cosAlpha);
    for (i = 1; i <= Nphpar; i++)
        php[i] = cg[ncoef0 + i];

    phasec(dphp, alpha, php); /* computes also Scale */

    matrix(cg[ncoef0], time, tmat, dtm);

    double br = 0;
    /* Directions (and ders.) in the rotating system */

    for (i = 1; i <= 3; i++)
    {
        e[i] = 0;
        e0[i] = 0;
        e[i] = tmat[i][1] * ndv.xx->x + tmat[i][2] * ndv.xx->y + tmat[i][3] * ndv.xx->z;
        e0[i] = tmat[i][1] * ndv.xx0->x + tmat[i][2] * ndv.xx0->y + tmat[i][3] * ndv.xx0->z;

        for (j = 1; j <= 3; j++)
        {
            de[i][j] = dtm[j][i][1] * ndv.xx->x + dtm[j][i][2] * ndv.xx->y + dtm[j][i][3] * ndv.xx->z;
            de0[i][j] = dtm[j][i][1] * ndv.xx0->x + dtm[j][i][2] * ndv.xx0->y + dtm[j][i][3] * ndv.xx0->z;
        }
    }

    for (i = 0; i < Numfac; i++)
    {
        const auto lmu = e[1] * Nor[0][i] + e[2] * Nor[1][i] + e[3] * Nor[2][i];
        const auto lmu0 = e0[1] * Nor[0][i] + e0[2] * Nor[1][i] + e0[3] * Nor[2][i];

        if ((lmu > TINY) && (lmu0 > TINY))
        {
            // INNER_CALC_DSMU
            const auto dnom = lmu + lmu0;
            const auto s = lmu * lmu0 * (cl + cls / dnom);
            br = br + Area[i] * s;

            incl[incl_count] = i;
            dbr[incl_count++] = Darea[i] * s;

            const auto dsmu = cls * pow(lmu0 / dnom, 2) + cl * lmu0;
            const auto dsmu0 = cls * pow(lmu / dnom, 2) + cl * lmu;

            // end of inner_calc_dsmu

            // INNER_CALC
            const auto sum1 = Nor[0][i] * de[1][1] + Nor[1][i] * de[2][1] + Nor[2][i] * de[3][1];
            const auto sum10 = Nor[0][i] * de0[1][1] + Nor[1][i] * de0[2][1] + Nor[2][i] * de0[3][1];
            const auto sum2 = Nor[0][i] * de[1][2] + Nor[1][i] * de[2][2] + Nor[2][i] * de[3][2];
            const auto sum20 = Nor[0][i] * de0[1][2] + Nor[1][i] * de0[2][2] + Nor[2][i] * de0[3][2];
            const auto sum3 = Nor[0][i] * de[1][3] + Nor[1][i] * de[2][3] + Nor[2][i] * de[3][3];
            const auto sum30 = Nor[0][i] * de0[1][3] + Nor[1][i] * de0[2][3] + Nor[2][i] * de0[3][3];

            tmpdyda1 = tmpdyda1 + Area[i] * (dsmu * sum1 + dsmu0 * sum10);
            tmpdyda2 = tmpdyda2 + Area[i] * (dsmu * sum2 + dsmu0 * sum20);
            tmpdyda3 = tmpdyda3 + Area[i] * (dsmu * sum3 + dsmu0 * sum30);
            tmpdyda4 = tmpdyda4 + lmu * lmu0 * Area[i];
            tmpdyda5 = tmpdyda5 + Area[i] * lmu * lmu0 / (lmu + lmu0);
            // end of inner_calc
        }
    }

    /* Derivatives of brightness w.r.t. g-coeffs */
    for (i = 0; i < ncoef0 - 3; i++)
    {
        tmpdyda = 0;
        for (j = 0; j < incl_count; j++)
            tmpdyda = tmpdyda + dbr[j] * Dg[incl[j]][i];
        dyda[i] = Scale * tmpdyda;
    }
    /* Ders. of brightness w.r.t. rotation parameters */
    dyda[ncoef0 - 3 + 1 - 1] = Scale * tmpdyda1;
    dyda[ncoef0 - 3 + 2 - 1] = Scale * tmpdyda2;
    dyda[ncoef0 - 3 + 3 - 1] = Scale * tmpdyda3;

    /* Ders. of br. w.r.t. cl, cls */
    dyda[ncoef - 1 - 1] = tmpdyda4 * Scale * 0.1;
    dyda[ncoef - 1] = tmpdyda5 * Scale;

    /* Ders. of br. w.r.t. phase function params. */
    for (i = 1; i <= Nphpar; i++)
        dyda[ncoef0 + i - 1] = br * dphp[i];

    /* Scaled brightness */
    br *= Scale;

    return(br);
}
