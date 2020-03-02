/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <CL/cl.hpp>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.hpp"
#include "constants.h"

double bright_old(double ee[], double ee0[], double t, double cg[], cl_double dyda[], int ncoef)
{
    int ncoef0, i, j, k,
        incl[MAX_N_FAC], //array of indexes of facets to Area, Dg, Nor. !!!!!!!!!!!incl IS ZERO INDEXED
        incl_count = 0;

    double cos_alpha, br, cl, cls, alpha, dnom, tmpdyda,
        e[4], e0[4],
        php[N_PHOT_PAR + 1], dphp[N_PHOT_PAR + 1], s,
        dbr[MAX_N_FAC], //IS ZERO INDEXED
        de[4][4], de0[4][4], tmat[4][4],
        dtm[4][4][4];

    double tmpdyda1 = 0, tmpdyda2 = 0, tmpdyda3 = 0;
    double tmpdyda4 = 0, tmpdyda5 = 0;

    ncoef0 = ncoef - 2 - Nphpar;
    cl = exp(cg[ncoef - 1]); /* Lambert */
    cls = cg[ncoef];       /* Lommel-Seeliger */
    cos_alpha = host_dot_product(ee, ee0);
    alpha = acos(cos_alpha);
    for (i = 1; i <= Nphpar; i++)
        php[i] = cg[ncoef0 + i];

    phasec(dphp, alpha, php); /* computes also Scale */

    matrix(cg[ncoef0], t, tmat, dtm);

    br = 0;
    /* Directions (and ders.) in the rotating system */
    for (i = 1; i <= 3; i++)
    {
        e[i] = 0;
        e0[i] = 0;
        for (j = 1; j <= 3; j++)
        {
            e[i] = e[i] + tmat[i][j] * ee[j];
            e0[i] = e0[i] + tmat[i][j] * ee0[j];
            de[i][j] = 0;
            de0[i][j] = 0;
            for (k = 1; k <= 3; k++)
            {
                de[i][j] = de[i][j] + dtm[j][i][k] * ee[k];
                de0[i][j] = de0[i][j] + dtm[j][i][k] * ee0[k];
            }
        }
    }

    /*Integrated brightness (phase coeff. used later) */
    double lmu, lmu0, dsmu, dsmu0;
    for (i = 0; i < Numfac; i++)
    {
        lmu = e[1] * Nor[0][i] + e[2] * Nor[1][i] + e[3] * Nor[2][i];
        lmu0 = e0[1] * Nor[0][i] + e0[2] * Nor[1][i] + e0[3] * Nor[2][i];
        /*lmu = e[1] * Nor[i][1] + e[2] * Nor[i][2] + e[3] * Nor[i][3];
        lmu0 = e0[1] * Nor[i][1] + e0[2] * Nor[i][2] + e0[3] * Nor[i][3];*/
        if ((lmu > TINY) && (lmu0 > TINY))  // if (icmp) 'avx'
        {
            // INNER_CALC_DSMU
            dnom = lmu + lmu0;
            s = lmu * lmu0 * (cl + cls / dnom);
            //br = Area[i] * s;
            br = br + Area[i] * s;
            //
            incl[incl_count] = i;
            dbr[incl_count++] = Darea[i] * s;
            //
            dsmu = cls * pow(lmu0 / dnom, 2) + cl * lmu0;
            dsmu0 = cls * pow(lmu / dnom, 2) + cl * lmu;

            // end of inner_calc_dsmu

            // INNER_CALC
            /*double sum1 = 0, sum2 = 0, sum3 = 0;
            double sum10 = 0, sum20 = 0, sum30 = 0;*/

            double sum1  = Nor[0][i] *  de[1][1] + Nor[1][i] *  de[2][1] + Nor[2][i] *  de[3][1];
            double sum10 = Nor[0][i] * de0[1][1] + Nor[1][i] * de0[2][1] + Nor[2][i] * de0[3][1];
            double sum2  = Nor[0][i] *  de[1][2] + Nor[1][i] *  de[2][2] + Nor[2][i] *  de[3][2];
            double sum20 = Nor[0][i] * de0[1][2] + Nor[1][i] * de0[2][2] + Nor[2][i] * de0[3][2];
            double sum3  = Nor[0][i] *  de[1][3] + Nor[1][i] *  de[2][3] + Nor[2][i] *  de[3][3];
            double sum30 = Nor[0][i] * de0[1][3] + Nor[1][i] * de0[2][3] + Nor[2][i] * de0[3][3];

            /*for (j = 1; j <= 3; j++)
            {
                sum1 = sum1 + Nor[j][i] * de[j][1];
                sum10 = sum10 + Nor[j][i] * de0[j][1];
                sum2 = sum2 + Nor[j][i] * de[j][2];
                sum20 = sum20 + Nor[j][i] * de0[j][2];
                sum3 = sum3 + Nor[j][i] * de[j][3];
                sum30 = sum30 + Nor[j][i] * de0[j][3];
            }*/

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
    /*dyda[ncoef0 - 3 + 1] = Scale * tmpdyda1;
    dyda[ncoef0 - 3 + 2] = Scale * tmpdyda2;
    dyda[ncoef0 - 3 + 3] = Scale * tmpdyda3;*/

    /* Ders. of br. w.r.t. cl, cls */
    dyda[ncoef - 1 - 1] = tmpdyda4 * Scale * 0.1;
    dyda[ncoef - 1] = tmpdyda5 * Scale;

    //dyda[ncoef - 1 - 1] = Scale * tmpdyda4 * cl;
    //dyda[ncoef - 1] = Scale * tmpdyda5;

    /* Ders. of br. w.r.t. phase function params. */
    for (i = 1; i <= Nphpar; i++)
        dyda[ncoef0 + i - 1] = br * dphp[i];

    /* Scaled brightness */
    br *= Scale;

    return(br);
}
