/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyFma.hpp"

#if defined(__GNUC__)
__attribute__((target("avx,fma")))
#endif
void CalcStrategyFma::curv(double cg[])
{
    int i, m, l, k;

    for (i = 1; i <= Numfac; i++)
    {
        double g = 0;
        int n = 0;
        //m=0
        for (l = 0; l <= Lmax; l++)
        {
            double fsum;
            n++;
            fsum = cg[n] * Fc[i][0];
            g = g + Pleg[i][l][0] * fsum;
        }
        //
        for (m = 1; m <= Mmax; m++)
            for (l = m; l <= Lmax; l++)
            {
                double fsum;
                n++;
                fsum = cg[n] * Fc[i][m];
                n++;
                fsum = fsum + cg[n] * Fs[i][m];
                g = g + Pleg[i][l][m] * fsum;
            }
        g = exp(g);
        Area[i - 1] = Darea[i - 1] * g;

        __m256d avx_g = _mm256_set1_pd(g);
        int cyklus = (n >> 2) << 2;
        for (k = 1; k <= cyklus; k += 4)
        {
            __m256d avx_pom = _mm256_loadu_pd(&Dsph[i][k]);
            avx_pom = _mm256_mul_pd(avx_pom, avx_g);
            _mm256_store_pd(&Dg[i - 1][k - 1], avx_pom);
        }
        if (k <= n) Dg[i - 1][k - 1] = g * Dsph[i][k]; //last odd value
        if (k + 1 <= n) Dg[i - 1][k - 1 + 1] = g * Dsph[i][k + 1]; //last odd value
        if (k + 2 <= n) Dg[i - 1][k - 1 + 2] = g * Dsph[i][k + 2]; //last odd value
    }

    // For Unit tests
    /*printf("\nDg[%d][%d:\n", 288, 24);
    for(int q = 0; q <= 16; q++)
    {
        printf("_dg_%d[] = { ", q);
        for(int p = 0; p <= 288; p++)
        {
            printf("%.30f, ", Dg[p][q]);
        }
        printf("};\n");
    }

    printf("\nArea[%d]:\n", 288);
    printf("_area[] = { ");
    for(int p = 0; p <= 288; p++)
    {
        printf("%.30f, ", Area[p]);
    }
    printf("};\n");*/
}